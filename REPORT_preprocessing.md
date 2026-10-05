# Preprocessing & modeling-readiness report — e-commerce order anomaly detection

Scope: `eda.ipynb` + `orders-sheet.csv` (150,000 × 22). Every number below was recomputed in `prep_check.ipynb` (scratch notebook, executed; `eda.ipynb` untouched, no final model trained).

> **Note on "your summary".** The placeholder `[PASTE YOUR "7. Summary" SECTION HERE]` was never filled in, so I verified the **"7. Summary" section that is already inside `eda.ipynb`** (claims 1–10 + "Transformations needed"). If you meant a different summary, tell me.
>
> **Note on the task sheet (`task-friday.docx`).** Tasks 2–3 serve the model through a FastAPI endpoint that scores **one order at a time**. That constrains this design: anything needing other rows at inference (customer aggregates, rolling windows) is a liability. I flag where it matters.

---

## 0. TL;DR — what you must know before coding

1. **Most of your summary checks out numerically** (duplicates, ID collisions, missing %, formula match, label cleanup). Details and 8 corrections in §1.
2. **The data contains five tight, rule-breaking clusters ≈ 2,695 rows (1.80 %)**, each ≈ 540 rows (see §1.3). They look planted/synthetic. I call them *rule-derived pseudo-labels*: useful as a sanity check, **not ground truth**, and partly circular with the features (§7).
3. **The residual (`order_value` vs formula) only sees 4 of the 5 clusters (79.5 %).** The fifth — **platform_fee inflated 0.39–3.6× the subtotal (537 rows)** — has a perfectly consistent order total, so the residual is blind to it. Your EDA found the fee ratios but never noticed this cluster is formula-clean.
4. **`account_age_days` is informative — your summary says otherwise.** 536 of the anomalies are accounts aged **0–7 days**; 26.8 % of orders from accounts ≤ 30 days are in a cluster vs 1.44 % elsewhere. Summary item 9 ("account-age buckets show almost the same median value") is **wrong**: median order value for ≤ 30-day accounts is 81.32 vs 51.45 overall.
5. **Customer aggregates, hour/dow, `customer_age`, `geo_mismatch`, missing-flags, rare-combo frequency: no usable signal** (§1.4). Dropping them keeps the model small *and* keeps the API stateless.
6. **A plain Isolation Forest on well-engineered features reaches AP ≈ 0.87–0.98 vs 0.25 on raw columns** (pseudo-label caveat), but is **weak on fee-inflated rows and on single-feature anomalies** (§7.4). Plan for that.

---

## 1. Verification of your findings

### 1.1 Claims verified (✓), corrected (✗), or incomplete (~)

| # | EDA claim | Verdict | Evidence from the data |
|---|---|---|---|
| 1 | 150,000 × 22; 637 days; missing only in 5 cols (1.2 / 0.6 / 0.4 / 1.8 / 0.9 %) | ✓ | `customer_age` 1,802; `discount_pct` 900; `shipping_cost` 601; `device_type` 2,703; `ip_country` 1,347 |
| 2 | `order_id` not unique; collisions, not multi-line orders; 350 exact dups | ✓ | 74,002 unique IDs / 150,000 rows; 120,477 rows share an ID. IDs span `ORD0000002–ORD0092000`; if 150k IDs were drawn *at random* from 92,000 you'd expect ≈ 73,982 unique — observed 74,002. **IDs are random labels with no meaning.** Only 14 extra row-pairs share both `order_id` and `customer_id` (chance). 350 dups = 700 rows involved |
| 3 | Dirty labels: 20→10 categories, `paypal`, `web ` | ✓ | raw→clean unique: category 20→10, payment 6→5, channel 4→3. Dirty rows: ~299 `web `, 301 `paypal`, 28–45 per lower-case category |
| 4a | Formula holds for "~97.6 % of rows" | ~ **wrong denominator** | 98.56 % of the 148,502 *evaluable* rows (146,360). 97.57 % of **all** rows counts the 1,498 NaN-residual rows as failures |
| 4b | `platform_fee` not in total | ✓ | adding/subtracting matches 0 % |
| 4c | 2,142 violators (1.4 %), ~70 % with value higher than expected, median gap ≈ 2.5k | ✓ | 2,142 = 1.43 %; 70.1 % positive; median \|residual\| = 2,563. Smallest \|residual\| = 0.96: only **1** row lies between 0.01 and 1 → tolerance 0.01 is safe, violations are not rounding noise |
| 5 | Heavy tails; skew > 1 for money/qty → log | ✓ (nuance) | skew raw → log1p: quantity 14.7→2.08, unit_price 11.2→0.23, shipping 19.5→−0.27, tax 9.0→0.67, fee 7.5→0.68, order_value 23.2→1.49. `discount_pct` skew is only 1.75 and the mass is at 0 / 5–50 → **don't log it** |
| 6 | ~500 orders > 5× product median price; some products max/min > 1000 | ✓ | 513 rows > 5×; 336 rows < ⅕; 4 products with max/min > 1000. Clean rows stay in log-ratio −0.24…+0.21 (0.1 %–99.9 %) |
| 7 | Tax ≈ 10 %, fee ≈ 11–12 % of subtotal | ✓ but misleading for tax | fee_rate median 0.1153 (identical across channels: 0.1153/0.1152/0.1153) – so the notebook's "fee varies by channel" worry is moot. **Tax rate is not "≈10 %": it's uniform on {0, 1 %, …, 20 %}** (each ≈ 5 % of rows), mean 10 %. A tax_rate of 3 % isn't odd; only > 20 % is |
| 8 | Geo mismatch ≈ 81 %, weak | ✓ | 81.1 %. Anomaly rate mismatch vs match 1.89 % vs 1.39 % (explained entirely by cluster E, where mismatch = 100 %) |
| 9a | Hour/dow ≈ uniform; 25 % at 00–05 h | ✓ | 24.8 %; χ² vs anomaly status p = 0.90 / 0.67 |
| 9b | "account-age buckets and order status show almost the same median value, except ≤30d whose *mean* is much higher" | ✗ | ≤ 30 d accounts: **median** 81.32 vs 51.45 overall (clean-only 52.13). The elevated median *is* cluster E: 536 rows with account age 0–7 d. Anomaly rate ≤ 30 d = 26.8 % vs 1.44 % |
| 9c | Burst activity minimal (max 3/h) | ✓ | median gap between a customer's orders ≈ 1,111 h |
| 10 | Most extreme orders "concentrated in a small set of customers … fraud ring" | ✗ **over-interpreted** | 17,997 customers, 8.3 orders each. Customers with ≥ 2 pseudo-anomalies: **193 observed vs 183 expected if independent.** Top-20 customers hold 6.2 % of spend. Top spenders are simply customers who drew one A-cluster order (value up to 200k) |
| – | "Transformations": impute `discount_pct` 0/median + flag, `shipping_cost` median + flag, 'Unknown' for device/ip | ~ | Flags not justified: see §3 |
| – | "one-hot for … country, category …" | ~ | Not needed for Isolation Forest — those columns carry no signal (§5) |

### 1.2 Things you overlooked or got wrong (summary)

1. **Fee-inflated cluster F (537 rows) is invisible to the residual** (§0.3, §1.3).
2. **Account age matters** — cluster E (§0.4).
3. Tax rate is uniform 0–20 %, not "≈ 10 %".
4. Formula-match denominator (97.57 vs 98.56 %).
5. Customer "ring" interpretation is not supported.
6. `formula_ok` as written in the notebook (`residual.abs() <= 0.01`) is **False for NaN residuals** — 1,498 rows counted as "not OK" (the notebook prints both numbers, but any model using `formula_ok` would treat missing-discount rows as violators).
7. **`customer_age`/`account_age_days` aren't constant per customer**: `customer_age` and country are (nunique = 1), but `account_age_days` takes 2–3 values for 532 of 17,997 customers and does not grow with time (corr ≈ −0.03). It's a static attribute with noise, not "account age at order date" — don't derive "age at order" from it.
8. Two metrics in the notebook (`orders_last_1h/24h`) are computed on the *whole* dataset (look-ahead-free but cross-row); fine for EDA, not usable by a stateless API — and signal is nil anyway.

### 1.3 The five pseudo-label clusters (reverse-engineered)

Disjoint, ≈ 540 rows each, total **2,695 = 1.80 %**. All are *far* from normal on at least one rule:

| Cluster | n | Signature | Residual sees it? |
|---|---|---|---|
| **A** bulk / high price | 539 | qty 25–99 (clean max = 10; **no** rows with qty 11–20), unit price 2–620× product median, `order_value` median 59.7k | yes (98.7 % violate) |
| **B** extreme discount | 543 | discount 70–95 % (clean max 50), `order_value` ≈ 134× formula (median) | yes (99.8 %) |
| **C** shipping / low price | 540 | shipping 153–899 (clean max 23.75) and, in 480 rows, unit price < ½ product median (down to 0.4 %); `order_value` ≈ 0.11× formula | yes (98.7 %) |
| **E** new-account value inflation | 536 | account age 0–7 d, 100 % geo-mismatch, inputs normal, `order_value` 930–6,989 = ×1.7–×640 formula | yes (100 %) — only via residual / value ratio |
| **F** fee inflated | 537 | `platform_fee` / subtotal = 0.39–3.6 (clean range **0.069–0.170**, a hard gap 0.17→0.39), everything else normal, formula OK | **no (0 %)** |

Clean-row reference ranges (useful as sanity bounds): fee_rate 0.069–0.170; tax_rate 0–0.201; shipping ≤ 23.75; quantity ≤ 10; discount ≤ 50 %; log(price/product median) ∈ [−0.24, 0.21]; order_value ≤ 5,902.

**Honesty warning for the README:** I derived these clusters by looking for rule breaks. If you evaluate your model against them you are partly measuring the features you built from the same rules (§7.2). Call them "rule-derived pseudo-labels", never "ground truth/fraud".

### 1.4 Signals that are *not* there (tested against pseudo-labels)

| Candidate | Result |
|---|---|
| payment, channel, device, status, coupon, category | χ² p = 0.51, 0.47, 0.36, 0.29, 0.26, 0.99 |
| customer_country | p = 0.05 (borderline, no consistent pattern) |
| hour / dow | p = 0.90 / 0.67 |
| `customer_age` | mean 46.8 anomalous vs 46.6 normal |
| missingness | anomaly rate missing vs present: customer_age 1.94 / 1.79 %, device 1.48 / 1.80 %, ip_country 1.86 / 1.80 %, discount 1.22 / 1.80 %, shipping 1.33 / 1.80 % (the lower rates for discount/shipping are structural: cluster B needs a discount, C needs shipping) |
| rare channel×payment×country combos | only 150 combos, smallest = 63 rows (none < 30); anomaly rate in rarest decile 2.01 % vs 1.77 % |
| customer aggregates | anomalies independent across customers (193 vs 183); `n_orders` vs anomaly rate flat (1.43–1.57 %) |
| category-relative value | category medians of order_value span only 45.9–54.1 |

---

## 2. Data cleaning

### 2.1 Label canonicalisation
**What:** strip whitespace, fixed mapping (not "most frequent spelling", which is data-dependent and unreproducible at inference).
**Why:** a typo like `paypal` (301 rows) or `web ` (299 rows) would be a rare category an anomaly model flags for the wrong reason.
**Code:**
```python
CAT_COLS = ["customer_country","ip_country","category","payment_method","sales_channel","device_type","order_status","coupon_used"]
FIXED_MAP = {"payment_method": {"paypal": "PayPal"}, "sales_channel": {"web": "Web"}}
def clean_labels(df):
    df = df.copy()
    for c in CAT_COLS:
        v = df[c].astype("string").str.strip()
        if c == "category": v = v.str.title()                      # 'toys' -> 'Toys' (all 10 are single words)
        elif c in FIXED_MAP: v = v.str.lower().map(FIXED_MAP[c]).fillna(v)
        df[c] = v
    return df
```
Result verified: category 10, payment 5 {Apple Pay, Bank Transfer, Card, Google Pay, PayPal}, channel 3 {Marketplace, Mobile App, Web}; identical to the notebook's canonicalisation. **Risk if skipped:** 10 phantom categories + 2 phantom labels. **Inference:** unseen/dirty labels must not crash (apply the same function, never `.astype('category')` with fixed levels).

### 2.2 The 350 exact duplicates
**Evidence:** 350 rows duplicate an earlier row on *all* 22 columns including the minute-level timestamp (700 rows involved; no ID appears > 2×). Anomaly rate among them 1.43 % ≈ base rate (4 of 350 in B, 1 in C) → not a "replay attack" pattern, just pipeline duplication.
**Recommendation (train):** drop with `keep="first"` **before fitting** (they would double-weight 350 points and bias lookup medians). **Recommendation (inference/output):** do **not** drop — score every input row so output row count = input row count; optionally add an `is_duplicate` column. **Alternative:** keep duplicates and flag them as suspicious → rejected, 345 of 350 are otherwise perfectly ordinary.
```python
dup = raw.drop(columns="row_id").duplicated(keep="first").values   # row_id must be excluded!
train_raw = raw[~dup]
```
**Risk if skipped:** minor (0.23 % of rows); more important is the *opposite* error — dropping by `order_id` would delete ~76,000 legitimate rows.

### 2.3 Safe row identifier
`order_id` is not a key (only 74,002 distinct). Do not group/merge on it.
```python
raw.insert(0, "row_id", np.arange(len(raw)))      # position in the input file
# output columns: row_id, order_id, anomaly_score, status
```
Output must contain `order_id`, `anomaly_score`, `status` (your requirement) — **add `row_id`** as the first column so scores trace back. For the API (single order), `row_id` is the caller's optional request id. (`customer_id + product_id + timestamp` is *almost* unique: 700 exact repeats = the duplicates — but a synthetic index is simpler.)

### 2.4 Invalid / impossible values
Checked: no negatives, zero prices, quantities ≤ 0, orders ≤ 0, ages < 18 or > 100, unparseable timestamps; `customer_age` all whole numbers. Legit zeros: discount 74,185, shipping 7,770, tax 159, account age 77. So **no row is dropped as "invalid"** — the extreme values (qty 99, discount 95, shipping 899, value 200k) are the signal, **do not clip them away before modelling**. Do clip *inside the engineered features* only to avoid `log(0)`/division by zero (§4). At the API: validate `quantity ≥ 1`, `unit_price > 0`, `0 ≤ discount_pct < 100`, `order_value > 0` (return 422), but allow extreme values.

---

## 3. Missing values

| Column | n (%) | Strategy | Missing flag? | Informative? |
|---|---|---|---|---|
| `customer_age` | 1,802 (1.2) | **not used** by the model (§4: no signal). If kept for display, median (47) | no | no: 1.94 vs 1.79 % |
| `device_type` | 2,703 (1.8) | not used; if one-hot later: `"Unknown"` level | no | no: 1.48 vs 1.80 % |
| `ip_country` | 1,347 (0.9) | not used (geo dropped) | no | no |
| `discount_pct` | 900 (0.6) | **0** (mode: 49.5 % of rows are 0) | **no model flag**; instead set residual features to 0 for these rows (below) | no (structurally cannot be B) |
| `shipping_cost` | 601 (0.4) | training median (6.51) | same | no (structurally cannot be C) |

**Key interaction (not in the summary):** `residual` is NaN for **1,498** rows (900 + 601 − 3 both). Naively imputing then computing the residual creates *fake violations* (e.g. true discount 15 % imputed as 0). **Rule:** when discount or shipping was imputed, set `log_value_ratio = 0` (no evidence) — implemented via `res_ok`. Cost: ≈ 1.0 % of rows lose the residual check; expected loss of true E-type rows ≈ 1,498 × 0.36 % ≈ 5.
**Alternative (not implemented):** recover the missing value from the formula — implied discounts of consistent rows fall on the clean grid {0, 5, 10, 15 … 50 %} (e.g. 453 of the checked rows imply 0 %, 98 imply 10 %), so one can snap-and-test. More code, ≈ 5 rows of gain → not worth it.
**Why no missing flags:** a 98.8 %-zero feature adds a mostly-constant dimension; Isolation Forest picks features uniformly at random per split, so every uninformative feature wastes splits (measured: adding `customer_age` or `hour` reduced AP from 0.871 to 0.852 / 0.847, `geo_mismatch` to 0.842, §4.3). **Risk if skipped:** negligible for flags; large for the fake-violation problem above.

---

## 4. Feature engineering

### 4.1 Proposed features (all computed row-wise; lookups from train only)

Notation: `sub = qty·price·(1−disc/100)` (clipped ≥ 0.01), `exp = sub + shipping + tax`, `pm` = product median unit price.

| Feature | Formula | Targets | Anomalous = | In final set? |
|---|---|---|---|---|
| `log_quantity` | log1p(qty) | bulk orders (A) | high (clean ≤ log 11 = 2.4; A 3.3–4.6) | ✅ |
| `log_unit_price` | log(price) | extreme pricing (A, C) | high/low | ✅ |
| `discount_pct` | raw, imputed 0 | discount abuse (B) | > 50 | ✅ (don't log) |
| `log_shipping` | log1p(shipping) | shipping abuse (C) | high (clean ≤ 23.75; C ≥ 153) | ✅ |
| `log1p_tax_rate` | log1p(tax/sub) | tax tamper / tiny subtotal (B,C) | > 0.2 | ✅ |
| `log_fee_rate` | log(fee/sub) | **fee inflation (F)**, tampered subtotals (A–C) | clean −2.67…−1.77; F ≥ −0.94 | ✅ |
| `price_vs_product` | log(price / pm) | price tampering (A: ln 2–620, C: ln 0.004–0.5) | \|x\| > 0.3 (clean ±0.24) | ✅ |
| `log_order_value` | log(order_value) | huge / tiny totals (A,B,E) | high | ✅ |
| **residual** `log_value_ratio` | log(order_value / exp), 0 if imputed | tampered totals (A,B,C,E) | ≠ 0 (clean: 0.00 ± 0.00) | ✅ (relative, symmetric) |
| `res_slog` (signed residual) | sign(r)·log1p(\|r\|), r = value − exp | same, in money | ≠ 0 | ❌ ρ = 0.99 with value_ratio; adding it: AP 0.871→0.856 |
| `abs_res_log` (abs residual) | log1p(\|r\|) | same | > 0 | ❌ redundant |
| `formula_ok` | \|r\| ≤ 0.01 (98.57 %) | binary version of the residual | 0 | ❌ as model input (derived from same info; 98.6/1.4 flag). ✅ as **explanation column** |
| `log_acct_age` | log1p(account_age_days) | new-account abuse (E) | very low (0–7 d) | ✅ |
| `log_ship_ratio` | log1p(ship / sub) | shipping out of proportion | high | ⚠️ in FULL set; **candidate to drop** (§7.4: AP 0.956→0.977 when removed with max_samples 4096) |
| `log_fee_to_value` | log(fee / order_value) | fee vs *charged* total (A,B,E) | very low, or > 0.3 | ✅ (collinear with fee_rate, see §4.4) |
| effective discount | `1 − (value − ship − tax)/(qty·price)` | stated vs charged discount | ≠ disc/100 | ❌ identical information to `log_value_ratio`; stated discount alone ≡ `discount_pct` |
| `log_value_vs_cat` | log(value / category median value) | context-relative value | high | ❌ ρ = 1.00 with `log_order_value` (category medians vary only 45.9–54.1) |
| customer aggregates (orders, spend, #devices…) | groupby customer | rings/resellers | – | ❌ no signal (193 vs 183), cross-row ⇒ breaks stateless API, self-inclusion leakage |
| `geo_mismatch` | customer_country ≠ ip_country | account takeover | 1 | ❌ 81 % base rate; 100 % in E but E is already found by account age + value; adding it **reduced** AP 0.871→0.842 |
| hour, dow | from timestamp | night orders | – | ❌ uniform, p = 0.90/0.67 |
| `customer_age` | raw | – | – | ❌ no signal |
| missing flags | `isna()` | – | – | ❌ §3 |
| rare-combo frequency | count(channel, payment, country) | unusual combos | low count | ❌ 150 combos, min 63, no signal (2.01 vs 1.77 %) |

### 4.2 Customer / product aggregates, leakage and inference
* **Product median price (`pm`) is the one aggregate worth having**: 1,200 products, ≥ 87 rows each; it's *the* reference that turns a price into a deviation. Leakage: the anomalous rows (A, C ≈ 1,000 rows, 513+336 off by > 5×) are 0.9 % of ~125 rows/product → one product has ~1 contaminated row, so the **median is robust**; no leave-one-out needed. Compute on **training rows after dedupe**, store as a dict `product_id → median`, ship it inside the model bundle. At inference: `dict.get`, fallback to category median price, then global median (verified: an unseen `product_id` scores without error).
* **Customer aggregates**: not recommended. If a course requirement forces them: compute **leave-one-out** (exclude the current row) in train, store per-customer sums/counts in the bundle, and at inference *add* the new order to the stored history only for that call — and accept that an unseen customer has no history. Given 8.3 orders/customer, they're noisy and the data shows no signal.
* **Anything computed over time windows** (orders in last hour): not usable in a one-row API; skip.

### 4.3 Feature ablation (Isolation Forest, 300 trees, max_samples 256, seed 0, vs pseudo-labels; K = 2,695; **caveat: the pseudo-labels are rule-derived, so engineered rule-style features look good by construction**)

| Feature set | AP | P@K | recall@K A / B / C / E / F |
|---|---|---|---|
| V0 raw numerics (9 cols, no engineering) | 0.251 | 0.252 | 1.00 / .12 / .06 / .04 / .05 |
| V1 log1p of raw numerics | 0.366 | 0.408 | 1.00 / .10 / .06 / .86 / .02 |
| V2 7 stated-input features (no order_value, no acct age) | 0.649 | 0.614 | 1.00 / .99 / .95 / .01 / .11 |
| V3 NO_VALUE (V2 + acct age + ship_ratio) | 0.744 | 0.664 | 1.00 / .98 / 1.00 / .22 / .11 |
| V5 V2 + log_order_value + value_ratio | 0.843 | 0.796 | 1.00 / 1.00 / .98 / .94 / .06 |
| V6 = V5 + log_acct_age | 0.852 | 0.809 | 1.00 / 1.00 / .98 / 1.00 / .06 |
| **FULL (12)** | **0.871** | **0.816** | 1.00 / 1.00 / 1.00 / 1.00 / .08 |
| FULL + geo_mismatch / customer_age / hour | 0.842 / 0.852 / 0.847 | | worse |
| FULL + res_slog / + log_value_vs_cat (collinear) | 0.856 / 0.850 | | worse |
| FULL − fee features | 0.814 | | F = 0.00 |

Raw account age (V6r) works almost as well as log (0.847 vs 0.852) and a `≤7 days` flag as well (0.860) — **log1p is simplest**; note IF is *not* invariant to monotone transforms (random split points are uniform in [min,max], so a 0–7-day pocket in a 0–2,999 range is almost never isolated).

### 4.4 Correlation / redundancy
Spearman |ρ| ≥ 0.7 among candidates: `log_order_value`↔`log_value_vs_cat` **1.00**; `log_value_ratio`↔`res_slog` **0.99**; `log_unit_price`↔`log_order_value` 0.79; `log_fee_rate`↔`log_fee_to_value` 0.74. Linear R² of each FULL feature on the other 11: quantity 0.99, unit_price 0.99, fee_rate 0.99, order_value 0.99, fee_to_value 0.99, value_ratio 0.98, ship_ratio 0.97 — high because the ratio features are *algebraic combinations* in log-space (log fee_rate = log fee − log qty − log price − log(1−disc); fee_to_value = log fee − log value). For Isolation Forest this isn't a correctness problem (no matrix inversion), but redundant groups get more split attention — and it **breaks Mahalanobis-type models** (Elliptic Envelope needs a non-singular covariance; it did run on FULL12 here, but I did not verify its covariance conditioning, so don't rely on that). For the second model use the reduced set (§7.5).

### 4.5 Features to drop (and why)
`order_id` (random labels, §1), `customer_id` / `product_id` / `product_name` (identifiers; product only via the median lookup; `product_name` is 1-to-1 with `product_id`), `order_timestamp` raw (uniform; hour/dow no signal), `customer_age`, `customer_country` / `ip_country` / `geo_mismatch`, `device_type`, `payment_method`, `sales_channel`, `order_status`, `coupon_used`, `category` (no signal: χ² p 0.26–0.99), raw `tax_amount` / `platform_fee` / `shipping_cost` / `quantity` / `unit_price` / `order_value` (replaced by logs/ratios), `res_slog`, `abs_res_log`, `formula_ok`, `log_value_vs_cat`, missing flags, customer aggregates, rare-combo frequency.
**Important caveat:** "no signal" holds *for this dataset's clusters*. The task sheet says to be ready to justify features: the defensible statement is "tested against rule-derived pseudo-labels, categorical context columns showed no association (p-values in §1.4), so they were left out to avoid diluting Isolation Forest's random splits".

### 4.6 Recommended final feature list
**FULL (12):** `log_quantity, log_unit_price, discount_pct, log_shipping, log1p_tax_rate, log_fee_rate, price_vs_product, log_order_value, log_value_ratio, log_acct_age, log_ship_ratio, log_fee_to_value`.
**Preferred after §7.4 tuning (11):** same minus `log_ship_ratio`, trained with `max_samples=4096`.
**NO_VALUE (9, for the non-circular ablation):** the 7 stated-input features + `log_acct_age` + `log_ship_ratio`.

---

## 5. Transformations and scaling

| Column / feature | Transform | Why (numbers) |
|---|---|---|
| quantity | `log1p` | skew 14.7 → 2.08 (≈ 99.6 % of rows ≤ 10) |
| unit_price | `log` | skew 11.2 → 0.23 |
| shipping_cost | `log1p` | 19.5 → −0.27 |
| tax, fee, order_value (raw) | not used raw; **ratios** (`tax/sub`, `fee/sub`, `fee/value`) then `log`/`log1p`; `log(order_value)` | removes scale; clean fee_rate 0.069–0.17, tax_rate ≤ 0.2 |
| discount_pct | none | skew 1.75, discrete grid {0,5,…,50} + B cluster 70–95 |
| account_age_days | `log1p` | uniform 0–2,999 → left skew −2.1 but gives the 0–7 d pocket 26 % of the range |
| residual | **log-ratio** `log(value/exp)` (not signed-log of money) | scale-free; ±; 0 for clean rows. Signed-log `sign·log1p(\|r\|)` has ρ = 0.99 → redundant; keep it only if you want "money at stake" for explanations |
| clipping | `fee/sub`: clip lower 1e-4; `sub` ≥ 0.01; `fee/value` lower 1e-6 | prevents `log(0)`, ±inf. **No upper clipping** — extremes are the signal |

**Encoding:** none needed — no categorical column is in the final set (§1.4). If you add any later: one-hot for `payment_method` (5), `sales_channel` (3), `device_type` (4, + Unknown), `order_status` (5), `coupon_used` (2); one-hot also for `category` (10) and countries (10 each) — all are < 11 levels, so **frequency encoding is unnecessary and would invent an ordinal** relation. Only IDs would need frequency/aggregate encoding, and they're dropped.

**Scaling:** **Isolation Forest does not need scaling** (splits are per-feature thresholds); log transforms matter (above), scaling doesn't. **Second model:** `RobustScaler` (median/IQR) after the log transforms, not `StandardScaler` — mean/std are dragged by the 1.8 % outliers (e.g. `log_value_ratio` std is inflated by ±7 outliers while clean rows sit at 0.00). Caveat: `log_value_ratio` has IQR = 0 → sklearn leaves scale at 1 (measured IQR printed as 1.0), so it is effectively unscaled; discount_pct has IQR 15. Fine; just don't expect equal footing.

---

## 6. Pipeline design

**One bundle** = `Pipeline([("features", OrderFeatures()), ("iforest", IsolationForest(...))])` + metadata. `OrderFeatures` is a custom `BaseEstimator, TransformerMixin`: `fit()` learns lookups from **deduplicated train rows only**; `transform()` is row-wise & stateless otherwise (so batch scoring and a one-row API call are identical). Tested in `prep_check.ipynb` §4/§6:

```python
class OrderFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features=tuple(FULL)): self.features = features
    def fit(self, df, y=None):
        d = clean_labels(df)
        self.prod_med_price_ = d.groupby("product_id").unit_price.median().to_dict()
        self.cat_med_price_  = d.groupby("category").unit_price.median().to_dict()
        self.global_med_price_ = float(d.unit_price.median())
        self.ship_median_ = float(d.shipping_cost.median())
        return self
    def transform(self, df):
        d = clean_labels(df)
        disc = d.discount_pct.fillna(0.0); ship = d.shipping_cost.fillna(self.ship_median_)
        res_ok = d.discount_pct.notna() & d.shipping_cost.notna()
        q, p = d.quantity, d.unit_price
        sub = (q * p * (1 - disc / 100)).clip(lower=0.01)
        expected = sub + ship + d.tax_amount
        pm = d.product_id.map(self.prod_med_price_).fillna(d.category.map(self.cat_med_price_)).fillna(self.global_med_price_)
        X = pd.DataFrame(index=d.index)
        X["log_quantity"] = np.log1p(q);            X["log_unit_price"] = np.log(p)
        X["discount_pct"] = disc;                   X["log_shipping"] = np.log1p(ship)
        X["log1p_tax_rate"] = np.log1p(d.tax_amount / sub)
        X["log_fee_rate"] = np.log((d.platform_fee / sub).clip(lower=1e-4))
        X["log_ship_ratio"] = np.log1p(ship / sub)
        X["price_vs_product"] = np.log(p / pm)
        X["log_order_value"] = np.log(d.order_value)
        X["log_fee_to_value"] = np.log((d.platform_fee / d.order_value).clip(lower=1e-6))
        X["log_value_ratio"] = np.where(res_ok, np.log(d.order_value / expected), 0.0)
        X["log_acct_age"] = np.log1p(d.account_age_days)
        return X[list(self.features)]
```
Placing the dedupe **outside** the pipeline (`train_raw = raw[~dup]`) keeps `transform` row-preserving.

**What to save** (`joblib.dump(bundle, "model.joblib")`, bundle size ≈ 2.9 MB mostly the 1,200-entry lookup):
* `pipeline` (fitted `OrderFeatures` with `prod_med_price_`, `cat_med_price_`, `global_med_price_`, `ship_median_`, + the fitted `IsolationForest`)
* `feature_order` (list; `transform` already returns it in order — also expose it for `/model-info`)
* `threshold` (score cutoff; see §7.3) and `contamination` used
* `sklearn`/`pandas` versions, training date, n_train rows, git hash/`model_version` string
* the label maps are in code (`FIXED_MAP`), which must be importable by `detect_anomalies.py`/the API — put `OrderFeatures`, `clean_labels` in one shared module (e.g. `features.py`), otherwise `joblib.load` fails with "can't get attribute".

**Verified:** reload gives identical scores on 150k rows (`np.allclose` True), a single-row call equals the batch result, an unseen product + dirty label (`"automotive "`) scores without error.
`detect_anomalies.py` must: read CSV → add `row_id` → `bundle["pipeline"].score_samples(df)` → `anomaly_score = -score_samples` (higher = more anomalous; **document the sign**) → `status = np.where(score > threshold, "suspicious", "normal")` → write `row_id, order_id, anomaly_score, status`. Input validation (required columns, numeric coercion, NaN handling) goes in front, not inside the transformer.

---

## 7. Modeling plan

### 7.1 Isolation Forest hyper-parameters (FULL set, seed 0)
| n_estimators | max_samples | AP | P@K | time |
|---|---|---|---|---|
| 100 / 300 / 600 | 256 | 0.859 / 0.871 / 0.867 | 0.81 | 0.6 / 1.8 / 3.6 s |
| 100 / 300 / 600 | 1024 | 0.912 / 0.937 / 0.922 | 0.83–0.85 | 0.8 / 2.3 / 4.7 s |
| 100 / 300 / 600 | **4096** | 0.959 / 0.956 / 0.964 | 0.87–0.88 | 0.9 / 2.7 / 5.4 s |

**Recommendation:** `n_estimators=300` (more trees adds nothing), **`max_samples=4096`** (larger subsamples matter *here* because each anomaly cluster is ~540 rows: with 256-row subsamples ≈ 1 anomaly per tree). Default `256` is the textbook value, but the data shows it under-detects cluster F. Alternative: `max_samples="auto"` (=256).
Seed stability (5 seeds, 256): Spearman vs seed 0 = 0.975–0.982; top-K Jaccard 0.90–0.92 (FULL) and 0.81–0.82 (NO_VALUE, whose top-K is less clear-cut). AP per seed 0.868–0.876.

### 7.2 Circularity — the important caveat
`log_value_ratio` is built from `order_value` vs the formula; the residual flags 2,142 of the 2,695 pseudo-anomalies (79.5 %). **Any "% of residual violators found" check is evaluating the feature, not the model.** Also my pseudo-labels use rules (qty > 20, discount > 50, shipping > 100, fee_rate > 0.3) that mirror the features. Treat all P@K / AP numbers above as *consistency* checks. Honest checks below.

### 7.3 Threshold (normal vs suspicious)
* sklearn's default (`contamination="auto"`, offset −0.5) flags **7,155 rows (4.8 %)** — far above the ~1.8 % structured contamination. Don't use it.
* The score histogram is **bimodal** (FULL, seed 0): bulk 0.35–0.55; valley at **0.575–0.60 (only 66 rows)**; second mode 0.60–0.75. Clean max score 0.619. A valley threshold ≈ 0.58 is defensible and data-driven.
* Trade-off table (top-k flagged): 1.0 % → precision 1.00 / recall 0.56; 1.33 % → 1.00 / 0.74; 1.60 % → 0.91 / 0.81; **1.80 % → 0.82 / 0.82**; 2.13 % → 0.70 / 0.83; 2.67 % → 0.57 / 0.84. (Recall plateaus ~0.82 because F rows score like normal orders under max_samples=256 — median F score 0.491 vs clean 99th percentile 0.526; with 4096 recall at 1.8 % is **0.875**, F 0.37.)
* **Recommendation:** `threshold = quantile of *training* scores at 1 − contamination` with `contamination = 0.018` (justified by 1.43 % formula violators + 0.36 % fee outliers = 1.8 %), **store the numeric threshold in the bundle** (never recompute on inference data — a batch with 30 % anomalies would silently shift it). Cross-check against the histogram valley. **Alternative:** two bands — "suspicious" > valley, "review" between 98th–99th percentile.
* Selecting `contamination` ≠ tuning on labels: you don't have labels. State this in the README.

### 7.4 Non-circular sanity checks (run in `prep_check.ipynb`)
1. **Ablation without `order_value`/residual (`NO_VALUE`)** — cannot "see" the residual: finds A/B/C 0.98–1.00 at K but E only 0.22 and F 0.11. Rank correlation with FULL 0.967, top-K overlap 0.79. Reading: the model reproduces violators in A–C from *inputs alone* (plausible, not leakage); E genuinely requires the total.
2. **Seed stability** (above).
3. **Synthetic injection into a copy of 5,000 clean rows** (never seen in training; threshold = 98.2th training percentile; control FPR 0.12 %, n = 800). Recall by anomaly type — *exposes the model's real weak spot*:

| Injection | Residual sees it? | IF 256 | IF 4096 | Elliptic Env. | max\|robust z\| > 6 |
|---|---|---|---|---|---|
| S1 quantity ×10 (fee/tax scaled, formula-consistent) | no | 0.06 | 0.17 | 0.04 | 0.46 |
| S2 unit_price ×6 (consistent) | no | 0.05 | 0.12 | 0.00 | 1.00 |
| S3 platform_fee ×8 only | no | 0.07 | 0.50 | 0.00 | 0.84 |
| S4 discount 70 % (consistent) | no | 0.52 | 0.81 | 0.08 | 1.00 |
| S5 shipping 150 (consistent) | no | 0.36 | 0.67 | 0.33 | 0.65 |
| S6 account age 1–5 d only | no | 0.08 | 0.16 | 0.00 | 1.00 |
| S7 order_value ×15 only | yes | 0.85 | 0.99 | 1.00 | 1.00 |
| control (clean) FPR | | 0.00 | 0.00 | 0.00 | 0.01 |

**Honest conclusion:** the Isolation Forest finds anomalies that deviate on **several features at once** (the five planted clusters each break 3–5 features), but is **insensitive to a single-feature deviation** among 12 features (S1, S2, S6 ≈ 5–17 % recall). `max_samples=4096` roughly doubles-to-triples this; a simple **max-robust-z rule layer** catches single-feature deviations well but has more false positives (clean 99.9th pct of max\|z\| = 9.1, max 17.0). Suggest in the README: IF = primary score; list per-feature deviations (§8) for context. Don't claim "detects any unusual order".
4. **Manual inspection of top-N** (50 highest, 50 near the threshold, 50 random "normal"): do the explanations (§8) look like real rule breaks? In FULL/256 the 495 rows in the top-K that are *not* pseudo-labelled are **not** fee-inflated: they are legitimate tiny orders (median subtotal 2.85, quantity 1, median discount 40 %, fee_rate 0.069–0.161 i.e. normal, 15 % from accounts ≤ 30 days; scores 0.546–0.619). They rank high because `log_fee_to_value` / `log_order_value` are extreme for very small totals — a genuine false-positive source to inspect and discuss in the README.
5. **Fee cluster F recall by configuration** (F recall@K): FULL/256 0.08 → FULL/4096 0.37 → **FULL−ship_ratio/4096 0.58** (AP 0.977, P@K 0.917, false positives 225 vs 495). Further work: try removing `log_fee_to_value` or `log_order_value`, or adding a hard rule `fee_rate > 0.2` (clean max 0.170) as a documented business rule.

### 7.5 Comparison model
* **Elliptic Envelope** (robust-scaled FULL, fit on a 30k sample): AUC 0.991, AP 0.860, top-K overlap with IF 0.83 — comparable on the planted clusters, **fails on single-feature synthetic anomalies** (recall 0–0.33; only S7, a value-only change, reaches 1.00), needs the reduced non-collinear feature set to be well-posed. Fast (5 s).
* **LOF**: *masked* with the usual `k`: `k=35`, AUC **0.434**, AP 0.098 (worse than random!) because each anomaly cluster has ≈ 100 members in a 30k sample, bigger than `k`; with `k=300` AUC 1.000, AP 0.998 — impressive but depends on `k` > cluster size and on my cluster-based pseudo-labels; novelty-mode LOF scoring cost grows with training sample size (32 s at k = 300 on 30k). Needs robust scaling.
* **When worth it:** use **one** comparison model, **Elliptic Envelope on robust-scaled reduced features** (cheap, interpretable Mahalanobis distance) or LOF (k≈ 300), to show agreement (top-K overlap 0.83 for EE) and disagreement. Not required for the deliverable; skip if time-limited. Use `RobustScaler`, not `StandardScaler` (§5).

---

## 8. Explaining flagged orders

**Method (recommended): robust per-feature deviation from training median/IQR**, printed for the top-3 features per order. Cheap, deterministic, works from a single row (API-friendly), and uses the same features as the model.
```python
med = X_train.median(); scale = np.maximum((X_train.quantile(.75) - X_train.quantile(.25)) / 1.349, 0.01)
z = (X_row - med) / scale                       # robust z per feature
reasons = z.abs().sort_values(ascending=False).head(3)
```
The `0.01` floor is needed because `log_value_ratio` has IQR = 0 (clean rows are exactly 0.00) — z-values for it come out in the hundreds; report "value ≈ ×N the formula" in words instead.
**SHAP** (`shap.TreeExplainer(pipe[1])`) works on the fitted IsolationForest here (verified, v0.52); its top features agreed only partly with the z method (e.g. it picked `log_fee_to_value` for A and B where z picked price/value ratio) because collinear features share credit. Use SHAP as a second opinion; stay with robust z as the primary explanation.

**Five examples** (from the run; scores = −score_samples with FULL/256; threshold 0.557):

| Cluster | row_id / order_id | Order | Score | Why flagged (top deviations) |
|---|---|---|---|---|
| A bulk | 130921 / ORD0041287 | qty 33 @ 1,621.25 (product median 40.17), 0 % discount, value 59,736 vs formula 53,509 | 0.718 | price 40× the product's norm (z 25.7); fee 3.81 on a 53k subtotal (fee_rate z −23.8) |
| B discount | 34251 / ORD0039045 | qty 1 @ 5.77, discount 94 %, stated total 3,352.80 vs formula 3.75 | 0.762 | charged ≈ ×894 the discounted formula (value-ratio z 680); fee 0.37 vs a 3.3k total |
| C shipping/low price | 110845 / ORD0004365 | qty 10 @ 1.96 (product median 167.66), shipping 350.26, tax 133.51 | 0.747 | price 1.2 % of product norm (z −30.9); tax rate z 35.5; value 1,192 vs formula 496 |
| E new account | 77885 / ORD0043376 | acct age 4 d, qty 1 @ 2.64, total 3,036.85 vs formula 11.46 | 0.716 | total ×265 the formula (z 558); geo mismatch; account 4 days old |
| F fee | 130755 / ORD0058976 | qty 1 @ 26.51, 50 % discount, fee 13.55 on a 13.26 subtotal, total 18.84 (formula OK) | 0.618 | fee_rate z 7.4, fee/value z 6.2; **formula-consistent, so the residual says nothing** — a good README example of why `fee_rate` exists |

These are the first (E, B, C) or best-scoring rows of each cluster in my run; re-pick after you train your final model. Also present 2–3 "near threshold" and 2 clean-but-extreme orders (e.g. clean order values up to 5,902: large qty × price, perfectly consistent) to show the model isn't just "big = suspicious".

---

## 9. README outline
1. **Problem & data** — 150k rows, no label, what "suspicious" means here (rule-consistent anomalies, not proven fraud).
2. **Data findings that drove design** — non-unique `order_id` (+ random-draw evidence), dirty labels, 350 duplicates, formula & residual, five rule-break clusters, columns with no signal. Include the "Verification" table.
3. **Cleaning** — mapping, dedupe policy, `row_id`, missing-value policy (incl. the residual-NaN rule).
4. **Features** — the final table (formula, behaviour targeted, expected direction, evidence) + dropped features with reasons.
5. **Model** — Isolation Forest, hyper-parameters (max_samples reasoning), threshold method, comparison model result.
6. **Validation** — circularity caveat, NO_VALUE ablation, seed stability, synthetic injection table, limitations (single-feature deviations, F cluster).
7. **Findings** — number flagged, breakdown by anomaly pattern, 5 explained orders, what would be investigated manually.
8. **How to run** — `python train.py --data orders-sheet.csv --out model.joblib`; `python detect_anomalies.py --model model.joblib --input … --output …` (columns `row_id, order_id, anomaly_score, status`; sign convention).
9. **Reproducibility** — versions, seeds, bundle contents.
10. **Limitations & next steps** — API statelessness, drift (product price medians go stale), retraining.

---

## 10. Final checklist (ordered)
1. Create `features.py` with `clean_labels`, `FIXED_MAP`, `OrderFeatures`, `FULL` (copy from `prep_check.ipynb` §2/§4). Unit-test: dirty labels, unseen product, NaN discount/shipping, 1 row vs batch.
2. `train.py`: read CSV → add `row_id` → `dup = raw.drop(columns="row_id").duplicated()` → `train_raw = raw[~dup]` → fit `Pipeline([OrderFeatures, IsolationForest(n_estimators=300, max_samples=4096, random_state=42)])`.
3. Drop `log_ship_ratio` (11 features) or confirm with the ablation that it helps you; re-run the §7.4 tests with your final config.
4. Compute training scores → `threshold = np.quantile(train_scores, 1 − 0.018)`; compare with the histogram valley; save `joblib` bundle with metadata (§6).
5. Print: number flagged, score summary, flagged rate by cluster rule (labelled "rule-derived").
6. `detect_anomalies.py`: load bundle; validate columns/dtypes; add `row_id`; score all rows (do **not** drop duplicates); write `row_id, order_id, anomaly_score, status`.
7. Run the NO_VALUE ablation, 5-seed stability, synthetic injection and the "top-50 / near-threshold / random-normal" manual review; record the numbers for the README.
8. Add explanations (robust z top-3 per flagged row; optional SHAP); pick 5 orders incl. one formula-clean fee case and one extreme-but-normal order.
9. Optional comparison model (Elliptic Envelope on robust-scaled reduced set, or LOF k≈300).
10. Write README per §9; keep the pseudo-label caveat visible.
11. For Tasks 2–3: expose `feature_order`, `threshold`, `model_version` from the bundle in `/model-info`; reuse `features.py` unchanged inside the API; pin `scikit-learn`/`pandas` versions in `requirements.txt` (the pickle depends on them).

---
*Reproduce: `prep_check.ipynb` (executed). Environment used: Python 3.14, pandas 3.0.6, scikit-learn 1.9.1, shap 0.52.*
