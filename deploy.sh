#!/usr/bin/env bash
# Goes in: scripts/deploy.sh   (run from anywhere: scripts/deploy.sh <commit-sha>)
#
# Blue/green deploy. The new image starts in the idle slot while the live slot
# keeps serving. NGINX is switched only after the new container is healthy.
# Rollback = run this script again with an older commit SHA.
set -euo pipefail

# Always work from the project root, even if called from another folder.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${1:?usage: scripts/deploy.sh <image_tag (commit SHA)>}"
IMAGE_REPO="${IMAGE_REPO:-muhammadahsan513/fraud-detect-ecom}"
IMAGE="${IMAGE_REPO}:${TAG}"
NETWORK="${NETWORK:-app-network}"
NGINX_SERVICE="${NGINX_SERVICE:-nginx}"        # service name in compose.yaml
NGINX_PORT="${NGINX_PORT:-80}"                 # host port NGINX is published on
STATE_FILE="${STATE_FILE:-./state.json}"
UPSTREAM_FILE="./nginx/active/upstream.conf"
APP_PORT=8000
HEALTH_TIMEOUT=60
DRAIN_SECONDS=30                               # max time the old slot gets to finish requests
# Same runtime settings your old compose "api" service had: .env file + log rotation.
DEFAULT_RUN_ARGS="--env-file .env --log-driver json-file --log-opt max-size=10m --log-opt max-file=3"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-$DEFAULT_RUN_ARGS}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [deploy] $*"; }

nginx_running() { [[ -n "$(docker compose ps -q --status running "$NGINX_SERVICE" 2>/dev/null)" ]]; }
nginx_exec()    { docker compose exec -T "$NGINX_SERVICE" "$@"; }

# ---------- 1. prepare ----------
log "step 1: preparing (tag=$TAG)"
mkdir -p ./nginx/active
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

# ---------- 2. pull the image ----------
log "step 2: pulling $IMAGE"
if ! docker pull "$IMAGE"; then
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "WARNING: pull failed, using the image that already exists locally"
  else
    log "ERROR: cannot pull or find $IMAGE"
    exit 1
  fi
fi

# ---------- 3. which slot is live? ----------
LIVE=""; OLD_TAG=""
if [[ -f "$STATE_FILE" ]]; then
  LIVE="$(sed -n 's/.*"live_slot": *"\([a-z]*\)".*/\1/p' "$STATE_FILE")"
  OLD_TAG="$(sed -n 's/.*"image_tag": *"\([^"]*\)".*/\1/p' "$STATE_FILE")"
fi
if [[ "$LIVE" == "blue" ]]; then IDLE="green"; else IDLE="blue"; fi
IDLE_NAME="api-${IDLE}"
LIVE_NAME=""; [[ -n "$LIVE" ]] && LIVE_NAME="api-${LIVE}"
log "step 3: live=${LIVE:-none} (${OLD_TAG:-none}), idle=$IDLE -> deploying $TAG to $IDLE_NAME"

# ---------- 4. start the new version in the idle slot ----------
log "step 4: starting $IDLE_NAME"
docker rm -f "$IDLE_NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$IDLE_NAME" --network "$NETWORK" \
  --restart unless-stopped \
  --health-interval=2s --health-timeout=5s \
  --health-start-period=20s --health-retries=3 \
  $EXTRA_RUN_ARGS \
  "$IMAGE" >/dev/null

# ---------- 5. wait for healthy (max 60s) ----------
log "step 5: waiting up to ${HEALTH_TIMEOUT}s for $IDLE_NAME to become healthy"
deadline=$((SECONDS + HEALTH_TIMEOUT))
status="starting"
while (( SECONDS < deadline )); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$IDLE_NAME" 2>/dev/null || echo missing)"
  [[ "$status" == "healthy" ]] && break
  [[ "$status" == "unhealthy" || "$status" == "missing" ]] && break
  sleep 2
done

if [[ "$status" != "healthy" ]]; then
  log "FAILED: $IDLE_NAME status is '$status' (timeout or unhealthy). Container logs:"
  docker logs "$IDLE_NAME" 2>&1 | tail -n 100 || true
  docker rm -f "$IDLE_NAME" >/dev/null 2>&1 || true
  log "NGINX and state file were NOT touched. ${LIVE_NAME:-No old slot} keeps serving."
  exit 1
fi
log "$IDLE_NAME is healthy"

# ---------- 6. switch NGINX to the new slot ----------
fail_and_restore() {
  log "FAILED: $1. Restoring the old NGINX config."
  if [[ -f "${UPSTREAM_FILE}.bak" ]]; then
    cp "${UPSTREAM_FILE}.bak" "$UPSTREAM_FILE"
    if nginx_running; then nginx_exec nginx -s reload || true; fi
  fi
  docker logs "$IDLE_NAME" 2>&1 | tail -n 50 || true
  docker rm -f "$IDLE_NAME" >/dev/null 2>&1 || true
  exit 1
}

log "step 6: pointing NGINX at $IDLE_NAME"
rm -f "${UPSTREAM_FILE}.bak"
[[ -f "$UPSTREAM_FILE" ]] && cp "$UPSTREAM_FILE" "${UPSTREAM_FILE}.bak"
echo "upstream api { server ${IDLE_NAME}:${APP_PORT}; }" > "$UPSTREAM_FILE"

if nginx_running; then
  nginx_exec nginx -t        || fail_and_restore "nginx -t rejected the new config"
  nginx_exec nginx -s reload || fail_and_restore "nginx reload failed"
  log "NGINX config tested and reloaded"
else
  log "NGINX is not running (first deploy), starting it"
  docker compose up -d --no-deps "$NGINX_SERVICE" || fail_and_restore "could not start NGINX"
fi

# ---------- 7. smoke test through NGINX ----------
log "step 7: smoke test via http://localhost:${NGINX_PORT}/health"
ok=0
for _ in 1 2 3 4 5; do
  if curl -fsS "http://localhost:${NGINX_PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
[[ $ok -eq 1 ]] || fail_and_restore "smoke test through NGINX failed"

# ---------- 8. record state, then retire the old slot ----------
printf '{"live_slot": "%s", "image_tag": "%s"}\n' "$IDLE" "$TAG" > "$STATE_FILE"
log "step 8: state saved ($IDLE, $TAG)"

if [[ -n "$LIVE_NAME" ]]; then
  log "stopping old slot $LIVE_NAME (up to ${DRAIN_SECONDS}s to finish in-flight requests)"
  sleep 3
  docker stop -t "$DRAIN_SECONDS" "$LIVE_NAME" >/dev/null 2>&1 || true
  docker rm "$LIVE_NAME" >/dev/null 2>&1 || true
fi

log "SUCCESS: $IDLE is live with $TAG"