# Unusual-order detection (unsupervised) — Task 1

Isolation Forest on 11 engineered, row-wise order features. It gives every order an `anomaly_score` and a
`normal` / `suspicious` status, without any fraud label.

| File | Purpose |
|---|---|
| `features.py` | shared module: model constants (seed, contamination, n_estimators, max_samples, default feature set), feature lists, `read_orders` + `validate_orders`, `clean_labels`, `FIXED_MAP`, `OrderFeatures` transformer, `anomaly_scores`, robust-z explanations |
| `train.py` | fits the pipeline + threshold from the CLI, writes `model.joblib` |
| `detect_anomalies.py` | loads `model.joblib` and scores a CSV; writes `row_id, order_uid, order_id, anomaly_score, status` |
| `score_order.py` | look up the score + reasons of one order (`--uid`, `--row-id`, `--order-id`), a whole product (`--product`) or a new order (`--json`) |
| `MANUAL.md` | step-by-step manual for finding the score of any order or product |
| `model.joblib` | saved bundle (pipeline + threshold + metadata) |
| `tests/test_features.py` | unit tests |
| `validation.ipynb` | evidence: feature-set check, threshold, ablation, seeds, injection, manual review, explanations |
| `REPORT_preprocessing.md`, `prep_check.ipynb`, `eda.ipynb` | Phase 1–2 analysis this design is based on |
| `scored_orders.csv`, `flagged_explanations.csv` | outputs (all orders; flagged orders with top-3 reasons) |

---

## 1. Problem and data

`orders-sheet.csv`: 150,000 orders × 22 columns, covering 637 days, 17,997 customers and 1,200 products. There is **no fraud label**.
"Suspicious" here means *statistically unusual relative to the bulk of orders*. It is a candidate for manual review, not
proven fraud. The model must later be served by an API that scores **one order at a time** (Task 2), so all inference
is stateless and row-wise.

## 2. Data findings that drove the design

All numbers were recomputed in `prep_check.ipynb` and re-checked in `validation.ipynb` §1.

| Finding | Evidence | Design consequence |
|---|---|---|
| `order_id` is **not unique** | 74,002 distinct IDs in 150,000 rows. The IDs span 2–92,000; drawing 150k at random from that range gives ≈ 73,982 unique, so the IDs are random labels | never group/merge on it; add `row_id` (position in file) |
| 350 exact duplicate rows | identical on all 22 columns incl. the minute-level timestamp; anomaly rate 1.43 % ≈ base rate | drop **only for training**; score every row at inference |
| Dirty labels | category 20 → 10 spellings, `paypal`, `web ` | fixed mapping, never "most frequent spelling" |
| Order formula | `order_value = qty·price·(1−disc/100) + shipping + tax` holds for 98.56 % of the 148,502 evaluable rows; `platform_fee` is **not** in the total | residual feature `log_value_ratio` |
| 2,142 formula violators (1.43 %) | median \|gap\| 2,563; only 1 row between 0.01 and 1, so these are not rounding errors | violations are signal |
| Five tight rule-breaking clusters ≈ 540 rows each (2,695 = 1.80 %) | A bulk/high price, B discount 70–95 %, C shipping 153–899 / price ≪ product median, E new account (0–7 d) + inflated total, F platform fee 0.39–3.6 × subtotal | used as **rule-derived pseudo-labels** for consistency checks only (see §6) |
| Cluster F is **formula-clean** | 0 of 537 violate the formula | the residual cannot see it → `log_fee_rate` is required |
| Account age matters | 26.8 % of orders from accounts ≤ 30 d are in a cluster vs 1.44 % otherwise | `log_acct_age` |
| Heavy tails | skew raw → log: quantity 14.7 → 2.1, price 11.2 → 0.2, shipping 19.5 → −0.3, value 23.2 → 1.5 | log transforms (Isolation Forest is *not* invariant to them) |
| No usable signal | payment, channel, device, status, coupon, category χ² p = 0.26–0.99; hour/dow p = 0.90/0.67; customer_age; geo mismatch (81 % base rate); missing-flags; customer aggregates (customers with ≥ 2 pseudo-anomalies: 193 observed vs 183 expected) | left out, which also keeps the API stateless |

## 3. Cleaning

* **Labels.** Whitespace is stripped from all categoricals, `category` is title-cased, and a fixed map is applied:
  `payment_method: paypal→PayPal`, `sales_channel: web→Web`. Unseen or dirty labels pass through and never crash; there
  are no fixed category levels.
* **Row identity.** `row_id = 0..n−1` (position in the input file) is the first output column. `order_uid` is a unique,
  content-based id: `"U"` + 12 hex characters of a SHA-1 hash of the 22 raw fields. Exact duplicates get `-2`, `-3`.
  It is the same for an order in any file or position, and also when the order is scored alone. 150,000 distinct values (vs 74,002 `order_id`s).
* **Duplicates.** `raw.drop(columns="row_id").duplicated(keep="first")` finds 350 duplicates, which are dropped before
  fitting (`row_id` must be excluded, otherwise nothing is a duplicate). At inference every row is scored, so output rows = input rows.
* **Invalid values.** None are present in the data. `validate_orders` (`features.py`, used by both `train.py` and `detect_anomalies.py`) validates the input *in front of* the pipeline:
  required columns, numeric coercion, `quantity ≥ 1`, `unit_price > 0`, `0 ≤ discount < 100`, `order_value > 0`, and
  non-negative shipping/tax/fee/account age. **Extreme values are allowed and never clipped** (qty 99, discount 95 %, value 200k are the signal).
* **Missing values.**

  | Column | Missing | Handling |
  |---|---|---|
  | `discount_pct` | 900 | 0 (the mode, 49.5 % of rows) |
  | `shipping_cost` | 601 | training median (6.51), stored in the pipeline |
  | `customer_age`, `device_type`, `ip_country` | 1,802 / 2,703 / 1,347 | not model inputs |

  **Residual rule ("res_ok"):** if discount or shipping was imputed, `log_value_ratio = 0`. Imputing and then
  computing the residual would create fake formula violations (a true 15 % discount imputed as 0). This affects 1,498 rows (1 %).
  No missing-indicator flags are used: they had no signal, and they waste Isolation Forest splits.

## 4. Features

`sub = qty·price·(1−disc/100)`, clipped to ≥ 0.01; `exp = sub + shipping + tax`; `pm` = product median unit price.
`pm` is learned on deduplicated training rows and stored in the bundle, with fallback to the category median and then the global median.
Clipping happens **only inside ratios** to avoid log(0) and division by zero.

| Feature | Formula | Behaviour targeted | Anomalous direction | Evidence |
|---|---|---|---|---|
| `log_quantity` | log1p(qty) | bulk orders | high | clean qty ≤ 10, cluster A 25–99, nothing in 11–20 |
| `log_unit_price` | log(price) | extreme pricing | high / low | skew 11.2 → 0.2 |
| `discount_pct` | raw (NaN → 0) | discount abuse | > 50 | clean grid {0,5,…,50}; B 70–95; skew only 1.75, so not logged |
| `log_shipping` | log1p(shipping) | shipping abuse | high | clean ≤ 23.75; C ≥ 153 |
| `log1p_tax_rate` | log1p(tax / sub) | tampered tax / tiny subtotal | > log1p(0.2) | tax rate is uniform 0–20 %, so only > 20 % is odd |
| `log_fee_rate` | log(clip(fee / sub, 1e-4)) | **fee inflation** (F), tampered subtotals | high (or very low) | clean 0.069–0.170, F 0.39–3.6; removing the fee features drops F recall to 0 |
| `price_vs_product` | log(price / pm) | price tampering vs the product's norm | \|x\| > 0.3 | clean ±0.24 (0.1–99.9 %); A ln 2–620, C ln 0.004–0.5 |
| `log_order_value` | log(order_value) | huge / tiny totals | high / low | A up to 200k, clean ≤ 5,902 |
| `log_value_ratio` | log(order_value / exp), 0 if imputed | tampered totals (A, B, C, E) | ≠ 0 | clean rows are exactly 0.00; 2,142 violators |
| `log_acct_age` | log1p(account_age_days) | new-account abuse (E) | low (0–7 d) | E = 536 rows aged 0–7 d; the log gives that pocket 26 % of the range |
| `log_fee_to_value` | log(clip(fee / order_value, 1e-6)) | fee vs the *charged* total | very low or high | complements `log_fee_rate` (ρ = 0.74) |

**Dropped features and why**
* `log_ship_ratio` = log1p(ship / sub). It is in the report's FULL-12 set but dropped after checking (§6.1): it is better on
  pseudo-labels without it, and it ties on held-out injections. The price is lower recall on shipping-only manipulation.
* `res_slog` / `abs_res_log`: ρ = 0.99 with `log_value_ratio`; adding them lowered AP.
* `formula_ok` flag: the binary version of the residual. Used for explanations only.
* `log_value_vs_cat`: ρ = 1.00 with `log_order_value`.
* Effective discount: the same information as the residual.
* `geo_mismatch`, `customer_age`, hour/dow, payment/channel/device/status/coupon/category/countries: no association
  (§2). Adding geo/age/hour *reduced* AP (0.871 → 0.842 / 0.852 / 0.847 in the report) by diluting random splits.
* Customer aggregates and time-window counts: no signal; cross-row, so they break a stateless one-order API; leakage risk.
* Missing-value flags; rare-combination frequency (smallest of 150 combos has 63 rows, so nothing is rare).
* IDs, `product_name` (1-to-1 with `product_id`), the raw timestamp. Raw `tax`, `fee`, `shipping`, `qty`, `price` and `value`
  are replaced by the logs and ratios above.

No scaling (Isolation Forest splits per feature) and no encoding (no categorical model inputs).

## 5. Model, hyperparameters, threshold

`Pipeline([("features", OrderFeatures(REDUCED_11)), ("iforest", IsolationForest(n_estimators=300, max_samples=4096, random_state=42))])`

* **n_estimators = 300.** 100 → 300 → 600 trees changed AP by ≤ 0.01.
* **max_samples = 4096** (the default is 256). Each cluster is ≈ 540 rows, so a 256-row subsample holds ≈ 1 anomaly per tree,
  and the fee cluster is under-detected. On held-out injections 4096 is better than 256 everywhere (fee ×8: 0.11 → 0.56;
  shipping: 0.04 → 0.29; validation §6).
* **Score sign:** `anomaly_score = −IsolationForest.score_samples(x)`. **Higher = more anomalous** (observed range 0.34–0.78).
* **Threshold:** `threshold = quantile(training anomaly_scores, 1 − 0.018) = 0.50306`, stored in the bundle and **never
  recomputed from inference data** (otherwise a batch with 30 % anomalies would silently shift it).
  `contamination = 0.018` is justified by 1.43 % formula violators + 0.36 % fee outliers. It was chosen from data
  structure, not tuned on labels; there are none.
* **Cross-check with the score-histogram valley (they disagree):**

  | Rule | Cut-off | Flags (of 150k) | Fee-cluster recall | Flags matching no rule |
  |---|---|---|---|---|
  | **training quantile (used)** | 0.503 | 2,702 (1.80 %) | 0.42 | 320 |
  | histogram valley | 0.564 | 2,170 (1.45 %) | 0.02 | 9 |
  | sklearn default (`offset_` −0.5) | 0.500 | 2,772 | – | – |

  The 532 rows between the two cut-offs are 213 fee-inflated rows + 311 rows that break no rule. The valley gives a
  high-confidence "suspicious" band, and the quantile adds a lower-confidence band that contains most of the detected F cluster.
  `threshold_valley` is stored in the bundle so the API can expose two bands later.

**Comparison model (optional, validation §9):** an Elliptic Envelope on robust-scaled, non-collinear features (9) gets AP 0.909 vs 0.960,
and its fee-cluster recall is 0.03. The 443 orders flagged by EE but not by IF all match no rule. Its robust covariance has
condition number ≈ 7 × 10⁸, because `log_value_ratio` has IQR 0, so it is ill-posed. It catches the price ×6 injection (0.97) but misses
fee ×8, discount, shipping and account-age injections (≤ 0.04). Isolation Forest stays the primary model.

## 6. Validation and limitations

> **Circularity caveat.** There is no ground truth. AP / precision / recall numbers below use **rule-derived pseudo-labels**,
> reverse-engineered from the data with rules (qty > 20, discount > 50 %, shipping > 100 or price < ½ product median, formula
> violation, fee rate > 0.3). These rules mirror several features (`log_quantity`, `discount_pct`, `log_shipping`, `price_vs_product`,
> `log_value_ratio`, `log_fee_rate`). **The feature selection and `max_samples` were partly judged against these same
> pseudo-labels**, so those numbers measure *consistency with the rules*, not fraud detection. The checks that do not depend
> on them are the NO_VALUE ablation, seed stability and synthetic injection into **held-out** rows.

### 6.1 Feature set: 11 vs 12, verified (5 seeds, `max_samples=4096`)
| | 11 (deployed) | 12 (FULL) |
|---|---|---|
| AP vs pseudo-labels | 0.973 ± 0.008 | 0.959 ± 0.007 |
| fee-cluster (F) recall at threshold | 0.54 ± 0.08 | 0.36 ± 0.06 |
| no-rule flags at threshold | 256 ± 43 | 350 ± 30 |
| held-out injection, mean recall S1–S8 | 0.463 | 0.457 |
| S5 shipping 150 (consistent) | **0.29** | **0.62** |
| control FPR | 0.25 % | 0.28 % |

11 wins on the circular lens and ties on the non-circular one, but loses on shipping-only manipulation in 5/5 seeds,
because `log_ship_ratio` is the only feature that sees shipping out of proportion to the subtotal. **Kept 11, with this documented cost.**

### 6.2 NO_VALUE ablation (no `order_value`, no residual)
With no access to the stated total, recall is A/B/C 1.00 / 1.00 / 1.00, E 0.54, F 0.18 (deployed: 1 / 1 / 1 / 1 / 0.42). Rank correlation with
the deployed model is 0.972. A, B and C are therefore visible from the order *inputs alone*; their detection is not just the residual echoing the
pseudo-label rule. E (inflated total on a normal-looking order) genuinely needs the total.

### 6.3 Seed stability (seeds 42–46)
Spearman vs seed 42 is 0.985–0.986, and flagged-set Jaccard is 0.90–0.93. All 391 seed-sensitive rows are no-rule or F rows near the
threshold; A/B/C/E are flagged by every seed. F recall varies from 0.42 to 0.63. The deployed seed 42 is the weakest of the five; I did not choose
the seed by its pseudo-label score.

### 6.4 Synthetic injection into held-out clean rows (the honest test)
5,600 clean, formula-consistent rows were **removed before training**. Then 600 rows per manipulation were altered and 800 left as
controls. Each model used its own training-quantile threshold. Recall is the mean of 5 seeds (min–max in brackets):

| Injection | Residual sees it? | IF 11 / 4096 (deployed config) | IF max_samples 256 | max\|robust z\| > 6 rule |
|---|---|---|---|---|
| control (untouched) — FPR | | 0.002 (0.00–0.00) | 0.001 | 0.001 |
| S1 quantity ×10 (fee/tax scaled, consistent) | no | **0.12** (0.10–0.15) | 0.04 | 0.43 |
| S2 unit price ×6 (consistent) | no | **0.13** (0.11–0.14) | 0.06 | 1.00 |
| S3 platform fee ×8 only | no | 0.56 (0.47–0.64) | 0.11 | 0.81 |
| S4 discount 70 % (consistent) | no | 0.73 (0.63–0.79) | 0.56 | 1.00 |
| S5 shipping 150 (consistent) | no | **0.29** (0.18–0.38) | 0.04 | 0.25 |
| S6 account age 1–5 d only | no | **0.14** (0.13–0.15) | 0.11 | 1.00 |
| S7 order value ×15 only | yes | 0.98 (0.94–1.00) | 0.97 | 1.00 |
| S8 two moderate: fee ×3 + account age 1–5 d | no | 0.76 (0.70–0.83) | 0.52 | 1.00 |

**Known weakness: single-feature deviations are mostly missed.** Mean recall over S1–S6 is **0.33**: a 10× quantity, a 6× price
or a 1–5-day account on an otherwise normal, formula-consistent order is caught only 12–14 % of the time. The model finds
orders that deviate on **several** features at once (every real cluster breaks 3–5 features; S8 with two moderate deviations: 0.76)
and large total mismatches (S7: 0.98). It should **not** be described as "detects any unusual order". A `max|robust z|` rule layer would
catch most single-feature cases, but on real data it fires on ~4,400 rows at z > 6 (report §7.4). It is a candidate second layer, not
implemented.

### 6.5 Other limitations
* **False-positive pattern (confirmed):** the 320 flagged orders that break no rule are mostly **very small, heavily discounted
  orders**:
  * median subtotal 6.70 vs 40.06 overall; median discount 40 % vs 5 %;
  * 44 % have a subtotal < 5;
  * the flag rate for no-rule orders with subtotal < 5 is 3.7 % vs 0.12 % otherwise.

  The second group is **brand-new accounts placing ordinary orders**: 16.4 % of clean orders from accounts ≤ 7 days old are flagged, and `log_acct_age`
  is the top reason for 97 of the 320. Their fee rates are normal (0.069–0.160). Expect reviewers to clear most of them.
* Large legitimate orders are flagged only 2.9 % of the time (43 of 1,474 orders above the clean 99th percentile), so "big" alone does not mean
  "suspicious", but a few qty-8–10 orders worth 1–4k sit just above the threshold.
* Product median prices are fixed at training time; they go stale as prices drift. Retrain periodically.
* Fee-cluster (F) recall is only 0.42 for the deployed seed; this is the model's weakest real pattern.
* The pseudo-labels are reverse-engineered and the clusters look synthetic/planted; real fraud may look different.

### 6.6 What did not reproduce from `REPORT_preprocessing.md`
Everything structural reproduced exactly: label counts, the 350 duplicates, 2,142 violators, the 2,695 pseudo-labels, and every feature
quantile. Differences:
1. **Headline numbers depend on the seed.** The report's FULL−ship_ratio/4096 run (seed 0) gives AP 0.977, F recall 0.58 and 225 false positives.
   The deployed seed 42 gives 0.960, 0.42 and 320. The 5-seed means are 0.973, 0.54 and 256, so the report's figures lie inside the seed range
   but are not what the shipped model achieves.
2. **Threshold cross-check.** The report's valley (0.575–0.60, 66 rows) was for `max_samples=256`. With 4096 the valley (0.564) and the
   quantile threshold (0.503) do *not* agree (§5); the report implied they would roughly coincide.
3. **sklearn default threshold.** The report says the default flags 4.8 %; that was with 256. With 4096 it flags 1.85 %.
4. **Injection recall** on truly held-out rows is somewhat lower than the report's injections into training rows
   (e.g. FULL/4096 quantity ×10: 0.07 vs 0.17; discount 70 %: 0.70 vs 0.81). Only held-out numbers are used here.
5. **Elliptic Envelope.** The report found 0.00 recall for price ×6 using FULL features. On the reduced set it gets 0.97, but its covariance is
   ill-conditioned (≈ 7 × 10⁸), confirming the report's warning not to rely on it.
6. **NO_VALUE** with 4096 finds E 0.54 and F 0.18 (the report had 0.22 and 0.11 at 256). The conclusion is unchanged.

Changes I made to the design: none beyond the report's own recommendation (11 features, 4096). I also added `--on-invalid flag`
(keeps the row count when a row fails validation) and stored `threshold_valley` and `explain_stats` in the bundle.

## 7. Findings

**2,702 of 150,000 orders (1.80 %) are suspicious** (`scored_orders.csv`). That is 2,694 of the 149,650 training rows plus 8 duplicates of
flagged rows. Breakdown by rule-derived pattern:

| Pattern | Rows | Flagged |
|---|---|---|
| A bulk / extreme price | 539 | 539 (100 %) |
| B extreme discount with inflated total | 543 | 543 (100 %) |
| C extreme shipping / price far below product norm | 540 | 540 (100 %) |
| E new account (0–7 d) with inflated total | 536 | 536 (100 %) |
| F platform fee 0.39–3.6 × subtotal (formula-consistent) | 537 | 224 (42 %) |
| matches no rule | 147,305 | 320 (0.22 %) |

The top 50 scores are all rule-breaking: 33 C, 16 A, 1 B. Every flagged order has top-3 reasons in `flagged_explanations.csv`.

**How explanations work.** For each feature, robust z = (value − training median) / max(IQR/1.349, 0.01), using the deduplicated training rows.
The statistics are stored in the bundle, so the same explanation works for a single API order. The 3 largest |z| are reported. Caveats:
* `log_value_ratio` has IQR 0 (clean orders sit exactly on the formula), so its z is set by the 0.01 floor and reaches the hundreds.
  Read it as "the total is ×N the formula".
* `log_fee_rate` saturates at the 1e-4 clip (z ≈ −23.8) for bulk orders.

**Explained orders** (from validation §8; each cluster example is the *median-scoring* flagged member, not the most extreme one):

| Pattern | row_id / order_id | Order | Score | Why flagged (top robust z) |
|---|---|---|---|---|
| A bulk | 61596 / ORD0068411 | qty 77 @ 1,642.58 (product median 18.64), fee 1.12 on a 126k subtotal, total 124,546.92 | 0.713 | price 88× the product norm (`price_vs_product` +31.1); fee negligible vs total (`log_fee_to_value` −26.7, `log_fee_rate` −23.8) |
| B discount | 20465 / ORD0072849 | qty 2 @ 9.21, discount 74 %, total 4,466.15 vs formula 15.10 | 0.673 | charged ×296 the discounted formula (`log_value_ratio` +569); `discount_pct` +6.7 |
| C shipping / low price | 81546 / ORD0013345 | qty 2 @ 2.98 (product median 23.08), shipping 866.66, total 50.28 vs formula 875.70 | 0.670 | total ×0.06 the formula (−285.7); price 13 % of norm (−14.2); shipping +8.6 |
| E new account | 105493 / ORD0023433 | account 2 days, qty 1 @ 51.85 (normal price), total 2,688.22 vs formula 55.13 | 0.641 | total ×49 the formula (+388.7); `log_acct_age` −7.6 |
| F fee inflated | 95462 / ORD0037372 | qty 4 @ 126.25, fee 321.92 = 0.85 × subtotal (clean max 0.17), total matches formula exactly | 0.518 | `log_fee_rate` +6.8, `log_fee_to_value` +6.3. The residual says nothing; this is why `log_fee_rate` exists, and why F scores only just above the threshold |
| near threshold | 39885 / ORD0043857 | account 8 days, qty 1 @ 494.50 (product median 528.37), everything consistent | 0.503 (suspicious, just above 0.50306) | `log_acct_age` −6.2, high price +3.1. Example of the new-account false-positive pattern; a reviewer would likely clear it |
| extreme but normal | 31732 / ORD0030461 | the largest clean order: qty 10 @ 389.40, total 4,096.76, consistent | 0.497 (normal) | `log_quantity` +5.7, `log_order_value` +4.5. Large but internally consistent, so not flagged, though it sits close to the threshold |

**What I would investigate manually first:** the high-confidence band (score > 0.564, 2,170 orders, almost all rule-breaking),
then F-type fee anomalies (fee/subtotal > 0.17), then the 0.503–0.564 band, deprioritising tiny-subtotal and new-account-only flags.

## 8. How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt      # Python 3.14
source .venv/bin/activate

python train.py --data orders-sheet.csv --out model.joblib
python detect_anomalies.py --model model.joblib --input orders-sheet.csv --output scored_orders.csv
python -m pytest -q                                                       # unit tests
jupyter nbconvert --to notebook --execute --inplace validation.ipynb      # evidence notebook (~3 min)
```

`train.py` options: `--features {reduced11,full12,no_value}`, `--contamination 0.018`, `--n-estimators 300`, `--max-samples 4096`,
`--seed 42`, `--model-version`.
`detect_anomalies.py` options: `--on-invalid error` (default: abort and list the bad rows) or `--on-invalid flag` (keep the row with status `invalid` and an empty score).

Output (`scored_orders.csv`):
```
row_id,order_uid,order_id,anomaly_score,status
0,UCA5E4916EB5B,ORD0020474,0.379065,normal
```
* `row_id` = position in the input file; use it, not `order_id`, to join back (order IDs repeat).
* `order_uid` = unique id of the order (see §3); look it up with `python score_order.py --uid <uid>` (see `MANUAL.md`).
* `anomaly_score` = −`score_samples`, **higher = more anomalous**.
* `status` = `suspicious` if `anomaly_score > threshold`.

Verified run: 150,000 rows in → 150,000 out; 2,702 suspicious / 147,298 normal.

## 9. Reproducibility

* **Versions** (pinned in `requirements.txt`, recorded in the bundle): Python 3.14.4, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3,
  joblib 1.6.0. The pickle depends on the scikit-learn/pandas versions; `features.py` must be importable when loading.
* **Seeds:** IsolationForest `random_state=42`; validation seeds 42–46; hold-out sample `RandomState(7)`, injections `RandomState(11)`.
* **Bundle** (`model.joblib`, 4.1 MB): `pipeline` (fitted `OrderFeatures` with the 1,200-product median-price lookup, category and
  global fallbacks, shipping median + fitted IsolationForest), `feature_order`, `feature_set`, `threshold` (0.50306), `threshold_method`,
  `threshold_valley` (0.5642), `contamination`, `score_definition`, `hyperparameters`, `explain_stats` (training median/scale),
  `n_train` (149,650), `n_duplicates_dropped` (350), `train_data_sha256` (`6ea5f8c6…`), `trained_at`, `model_version`
  (`iforest-reduced11-20261005`), `versions`.
* **Checks:**
  * Retraining with the same seed reproduces `model.joblib` scores exactly (max diff 0.0).
  * Reloading the bundle gives identical scores on all 150k rows.
  * Scoring 200 real rows one at a time equals the batch scores (`np.array_equal`).
  * 11 unit tests pass: dirty labels, unseen product, NaN discount/shipping, one-row vs batch, zero fee / tiny subtotal,
    `order_uid` uniqueness (incl. duplicates) and stability (row order, number formatting).

---

# Task 2 — Prediction API (FastAPI)

A strict HTTP service that scores **one order at a time** with the Task 1 model (`model.joblib`), reusing the
Task 1 code unchanged. `features.py`, `train.py`, `detect_anomalies.py`, `score_order.py` and `model.joblib` are not
modified.

```
app/
  main.py           app factory, lifespan (loads the model once), request middleware, error handlers
  config.py         settings from environment variables
  schemas.py        Pydantic v2 request / response models, allowed categorical values
  model_service.py  load + validate the bundle, score via Task 1's validate_orders + score_orders
  routes.py         GET /health, GET /model-info, POST /analyze-order, the batch endpoints, and the web UI at GET /
  reasons.py        plain-language wording for Task 1's top-3 explanations (features.explain)
  batch.py          whole-file checking: file checks, header check, row validation, chunked scoring
  reports.py        report storage on disk (UUID4 folders, expiry, paging, CSV/JSON export, formula safety)
  report_html.py    the self-contained HTML summary report
  logging_config.py
  static/           web UI: index.html, style.css, app.js (single order), batch.js (whole file)
tests/test_api.py   API tests (Task 1 tests in tests/test_features.py are untouched)
tests/test_ui.py    web UI tests: page + static files, allowed_values, reasons, the example orders
tests/test_batch.py batch tests: uploads, rejections, reports, downloads, consistency with /analyze-order + Task 1
```

## Setup

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pins scikit-learn 1.9.1 / pandas 3.0.6 so the pickled model loads
cp .env.example .env                     # edit if needed
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | — (**required**) | Path to the Task 1 bundle, e.g. `model.joblib`. Unset → startup fails with `MODEL_PATH is not set`. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `APP_ENV` | `development` | `development` / `test` / `production`. If the model file is missing or corrupt: **production exits at startup**; other envs start in degraded mode (`/health` 200 with `model_loaded:false`, the other endpoints 503). |
| `MAX_BODY_BYTES` | `16384` | Larger request bodies → 413 (checked on `Content-Length` and on the bytes actually received). Not used for `/analyze-batch`, which uses `MAX_UPLOAD_BYTES`. |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Bind port |
| `MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Largest CSV for `/analyze-batch` → 413 above (1 KB – 1 GB) |
| `MAX_UPLOAD_ROWS` | `200000` | Most orders per CSV → 413 above (1 – 10,000,000) |
| `BATCH_CHUNK_SIZE` | `10000` | Rows scored per model call (1 – 1,000,000) |
| `REPORTS_DIR` | `reports` | Where batch reports are saved. Created at startup; startup fails if it isn't writable. |
| `REPORT_TTL_MINUTES` | `1440` (1 day) | Reports older than this are deleted (1 – 525,600) |
| `MAX_REPORTS` | `50` | At most this many reports are kept; the oldest go first (1 – 10,000) |

Invalid values (e.g. `PORT=abc`, `LOG_LEVEL=LOUD`) also stop startup with a clear message.

## Run

```bash
set -a; source .env; set +a
uvicorn app.main:app --host $HOST --port $PORT
```

Startup log: `INFO app model loaded path=model.joblib version=iforest-reduced11-20261005 feature_count=11 threshold=0.503062 env=development`.
Uvicorn's own access log duplicates the app's per-request log line; add `--no-access-log` to keep only the app's.

## Test (one command)

```bash
pytest
```

Runs the Task 1 tests (`tests/test_features.py`, 11 tests), the API tests (`tests/test_api.py`), the web UI tests
(`tests/test_ui.py`) and the batch tests (`tests/test_batch.py`), 172 tests in total (about 40 seconds; one test
checks the full 150,000-row file).

## Request schema

The request body is one row of `orders-sheet.csv` as JSON. Every key below must be present; omitting a key → 422.
Unknown keys → 422 (`extra="forbid"`).

| Group | Fields | Rules |
|---|---|---|
| Read by the model (`features.REQUIRED_COLUMNS`) | `order_id`, `product_id` | string, stripped, 1–64 chars, `[A-Za-z0-9_-]` |
| | `category` | one of the 10 training categories (case-insensitive) |
| | `quantity` | **integer** ≥ 1 (2.5, "2", true rejected) |
| | `unit_price` | > 0 |
| | `discount_pct` | 0–95 (range seen in the data) **or null** |
| | `shipping_cost` | ≥ 0 **or null** |
| | `tax_amount`, `platform_fee` | ≥ 0 |
| | `order_value` | > 0 (the transformer takes `log(order_value)`) |
| | `account_age_days` | integer ≥ 0 |
| Context (validated, not used by the model) | `order_timestamp` | ISO 8601 (`2025-09-12 02:16:00`, `2025-09-12T02:16:00Z`, `…+02:00`); naive = UTC; not in the future (5 min clock-skew tolerance) |
| | `customer_age` | integer 18–120 **or null** |
| | `customer_country`, `ip_country` (`ip_country` may be null) | country names seen in training: Australia, Canada, France, Germany, Ireland, Italy, Netherlands, Spain, UK, US |
| | `payment_method` | Apple Pay, Bank Transfer, Card, Google Pay, PayPal |
| | `sales_channel` | Marketplace, Mobile App, Web |
| | `device_type` | Android, Desktop, Tablet, iPhone **or null** |
| | `order_status` | Cancelled, Completed, Pending, Refunded, Shipped |
| | `coupon_used` | Yes, No |
| Optional | `customer_id`, `product_name` | validated if sent; never logged |

All numbers must be JSON numbers (no strings); `NaN`, `Infinity` and `-Infinity` are rejected. Money fields are
capped at 1e9 to avoid overflow.

**Categorical policy: normalise like training, then validate.** Labels are stripped and matched case-insensitively,
exactly as `features.clean_labels` would clean them (`"paypal"` → `PayPal`, `"web "` → `Web`, `" toys"` → `Toys`),
and then must belong to the set of values seen in the training data. Unseen labels are rejected with 422 instead of
being passed through. A test checks that the category list equals the categories stored in the model.

**Nulls:** `discount_pct` and `shipping_cost` are null in ~1,500 training rows, and `OrderFeatures` imputes them
(0 and the training shipping median) and zeroes the order-value residual. The API accepts an explicit `null` for those
fields and passes it to the same code. It never fills in a default for a missing key.

## Endpoints (real requests/responses from a live run)

### `GET /health`
```bash
curl -i localhost:8000/health
```
```
HTTP/1.1 200 OK
content-type: application/json
x-request-id: a34a7ca3d35f4d9f9b0a1fcfd9752d71

{"status":"ok","model_loaded":true}
```
Never touches the model. It returns 200 even when the model failed to load (`"model_loaded": false`).

### `GET /model-info`
```bash
curl localhost:8000/model-info
```
```json
{
  "model_name": "IsolationForest (reduced11)",
  "model_version": "iforest-reduced11-20261005",
  "feature_names": ["log_quantity", "log_unit_price", "discount_pct", "log_shipping", "log1p_tax_rate",
                    "log_fee_rate", "price_vs_product", "log_order_value", "log_value_ratio", "log_acct_age",
                    "log_fee_to_value"],
  "threshold": 0.5030621506497381,
  "threshold_method": "quantile(train anomaly_score, 1 - 0.018)",
  "score_definition": "anomaly_score = -IsolationForest.score_samples(x); higher = more anomalous",
  "status_labels": ["normal", "suspicious"],
  "trained_with": {"scikit-learn": "1.9.1", "pandas": "3.0.6", "numpy": "2.5.3", "python": "3.14.4"},
  "n_training_rows": 149650,
  "trained_at": "2026-10-05T07:40:42+00:00",
  "allowed_values": {
    "category": ["Automotive", "Beauty", "Electronics", "Fashion", "Garden", "Home", "Office", "Sports", "Tools", "Toys"],
    "customer_country": ["Australia", "Canada", "France", "Germany", "Ireland", "Italy", "Netherlands", "Spain", "UK", "US"],
    "ip_country": ["Australia", "Canada", "France", "Germany", "Ireland", "Italy", "Netherlands", "Spain", "UK", "US"],
    "payment_method": ["Apple Pay", "Bank Transfer", "Card", "Google Pay", "PayPal"],
    "sales_channel": ["Marketplace", "Mobile App", "Web"],
    "device_type": ["Android", "Desktop", "Tablet", "iPhone"],
    "order_status": ["Cancelled", "Completed", "Pending", "Refunded", "Shipped"],
    "coupon_used": ["No", "Yes"]
  }
}
```
Every value is read from the loaded bundle except `status_labels`, which the bundle does not store, and
`allowed_values`. `status_labels` mirrors `detect_anomalies.score_orders`, and a test checks that they agree.
`allowed_values` is the list the request schema validates against (`app/schemas.ALLOWED`, the labels seen in
training). A test checks that its categories equal the categories the fitted transformer knows. The web UI builds
its dropdowns from it.

### `POST /analyze-order`
Normal order (row 0 of `orders-sheet.csv`):
```bash
curl -X POST localhost:8000/analyze-order -H 'Content-Type: application/json' -d '{
  "order_id": "ORD0020474", "product_id": "P01005", "category": "Home", "quantity": 1,
  "unit_price": 22.74, "discount_pct": 20.0, "shipping_cost": 5.37, "tax_amount": 0.25,
  "platform_fee": 2.79, "order_value": 23.81, "account_age_days": 806,
  "order_timestamp": "2025-09-12 02:16:00", "customer_age": 52, "customer_country": "France",
  "ip_country": "Italy", "payment_method": "PayPal", "sales_channel": "Mobile App",
  "device_type": "Android", "order_status": "Shipped", "coupon_used": "No",
  "customer_id": "C009777", "product_name": "HOM Product 1005"}'
```
```json
{"order_id":"ORD0020474","anomaly_score":0.3790653804547756,"status":"normal","threshold_used":0.5030621506497381,"model_version":"iforest-reduced11-20261005","reasons":["The discount is slightly higher than usual.","Tax is slightly lower than usual for an order of this size.","The item price is slightly lower than this product normally sells for."]}
```
Suspicious order (row 113: order_value 1212.88 on a 46.16 item, account 2 days old):
```bash
curl -X POST localhost:8000/analyze-order -H 'Content-Type: application/json' -d '{
  "order_id": "ORD0018746", "product_id": "P00481", "category": "Automotive", "quantity": 1,
  "unit_price": 46.16, "discount_pct": 10.0, "shipping_cost": 10.65, "tax_amount": 4.0,
  "platform_fee": 4.61, "order_value": 1212.88, "account_age_days": 2,
  "order_timestamp": "2026-02-07 01:40:00", "customer_age": 29, "customer_country": "US",
  "ip_country": "Italy", "payment_method": "Card", "sales_channel": "Marketplace",
  "device_type": "iPhone", "order_status": "Completed", "coupon_used": "No"}'
```
```json
{"order_id":"ORD0018746","anomaly_score":0.6205570964779656,"status":"suspicious","threshold_used":0.5030621506497381,"model_version":"iforest-reduced11-20261005","reasons":["The order total is much higher than price, discount, shipping and tax add up to.","The platform fee is much lower than usual compared with the order total.","The customer account is much newer than usual."]}
```
`reasons` are the top 3 features from Task 1's `features.explain` (largest robust z against the training medians in
`bundle["explain_stats"]`, as in `score_order.py`). `app/reasons.py` only puts them into words: the sign of z picks
"higher" or "lower", and |z| picks "slightly" (< 2), "noticeably" (2–5) or "much" (≥ 5). If a bundle has no
`explain_stats`, `reasons` is `[]`.

## Errors

Every error uses the same shape, and `request_id` equals the `X-Request-ID` response header:
```json
{"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `malformed_json` / `empty_body` | body is not JSON / body is empty |
| 404 | `not_found` | unknown route |
| 405 | `method_not_allowed` | e.g. `GET /analyze-order`, `POST /health` (with an `Allow` header) |
| 413 | `payload_too_large` | body > `MAX_BODY_BYTES` |
| 415 | `unsupported_media_type` | `POST /analyze-order` without `Content-Type: application/json` |
| 422 | `validation_error` | field errors; `details` = `[{"field", "reason"}]` (input values are not echoed back) |
| 503 | `model_unavailable` | model missing / corrupt / failed to load |
| 500 | `internal_error` | unexpected failure: generic message only, full traceback logged server-side |

422 example (quantity 2.5, unit_price 0, unknown category, extra key):
```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "request_id": "demo-422",
    "details": [
      {"field": "category", "reason": "must be one of ['Automotive', 'Beauty', 'Electronics', 'Fashion', 'Garden', 'Home', 'Office', 'Sports', 'Tools', 'Toys']"},
      {"field": "quantity", "reason": "Input should be a valid integer"},
      {"field": "unit_price", "reason": "Input should be greater than 0"},
      {"field": "coupon", "reason": "Extra inputs are not permitted"}
    ]
  }
}
```
503 example (server started with `MODEL_PATH=missing.joblib`, `APP_ENV=development`):
```
HTTP/1.1 503 Service Unavailable
x-request-id: 941484ba30b04f7e9e851f98094b78c1

{"error":{"code":"model_unavailable","message":"Model is not available. Try again later.","request_id":"941484ba30b04f7e9e851f98094b78c1","details":[]}}
```

## Logging

* Every request gets a `request_id`. A client-sent `X-Request-ID` is kept if it matches `[A-Za-z0-9._-]{1,64}`;
  otherwise the server generates a uuid4. The id is returned in the `X-Request-ID` header.
* One line per request: `method=POST path=/analyze-order status=200 latency_ms=3.12 request_id=…`
* Scoring: `scored order_id=ORD0020474 anomaly_score=0.379065 status=normal request_id=…`. No payloads,
  customer ids or product names are logged.
* Startup: model path, version, feature count and threshold, or the load error. Unhandled exceptions are logged
  with their full traceback.

## Flow: request → validation → feature function → model → response

1. **Middleware** assigns the request_id and enforces `Content-Type` (415) and `MAX_BODY_BYTES` (413).
2. **Schema validation** (`app/schemas.py`, Pydantic v2 strict) checks required keys, types, bounds, NaN/Inf, the ISO
   timestamp and the allowed categorical values → 422/400 on failure.
3. **Task 1 validation**: `features.validate_orders` runs on the one-row DataFrame as a second check, with the same
   rules as the CLI.
4. **Scoring**: `detect_anomalies.score_orders(bundle, df)`, the same function the CLI uses, calls
   `features.anomaly_scores` → `Pipeline.score_samples` → **`OrderFeatures.transform`** (the transformer fitted in
   `train.py`) → `IsolationForest`. The API has no preprocessing of its own.
5. **Threshold**: `status = "suspicious" if anomaly_score > bundle["threshold"] else "normal"`.
6. **Reasons**: `features.explain` on the same transformed row, worded by `app/reasons.py`.
7. **Response**: `order_id`, `anomaly_score`, `status`, `threshold_used`, `model_version`, `reasons`.

**Consistency guarantee.** `test_api_matches_detect_anomalies_on_real_rows` sends 30 real rows from
`orders-sheet.csv` (the first 20 plus 10 suspicious ones, some with null `device_type` / `customer_age`) through the
API. It asserts that each `anomaly_score` is **exactly equal** (`==`, not approximately) to `score_orders` on the same
rows, that each status matches, and that both match `scored_orders.csv` to 1e-6 (the file's precision).

---

# Web UI

A plain-English page for checking one order or a whole file, built for people who don't work with the API. It is
plain HTML, CSS and JavaScript in `app/static/`, with no build step and no framework. The same FastAPI app serves
it, so there is no CORS setup. The **Single order** tab (described here) calls only `/health`, `/model-info` and
`/analyze-order`, and stores nothing. The **Whole file** tab is described under
[Batch checking](#web-ui-whole-file). It calls `/analyze-batch` and `/reports/…`, and its reports are saved on the
server for `REPORT_TTL_MINUTES`.

## Open it

```bash
set -a; source .env; set +a
uvicorn app.main:app --host $HOST --port $PORT
```
Then go to **http://localhost:8000/** (or whatever `PORT` is set to). `GET /` returns `app/static/index.html`, and
`/static/*` serves `style.css` and `app.js`. Unknown paths still get the JSON 404 envelope.

## Screens

* **Header.** The page title and a status pill that follows `GET /health`: a green dot with "Service ready", or a
  red dot with "Service unavailable" when the model isn't loaded or the server can't be reached. It checks again
  every 30 seconds. **How does this work?** opens a plain-language explanation plus the model name, version, alert
  line (threshold) and number of training orders from `GET /model-info`.
* **Order form.** 22 fields in four groups: *Customer*, *Product*, *Money*, *Payment & channel*. Each field has a
  plain label and a one-line hint underneath. Dropdowns and the two country pickers are filled from
  `/model-info → allowed_values`. Number fields carry the API's min, max and step. The order time is a date-time
  picker in UTC that can't be set in the future. Optional fields are marked *optional*; left empty, they are sent as
  `null`.
* **Expected total.** Under the Money group: *items × price × (1 − discount %) + shipping + tax*, worked out live,
  with "Matches the order total" or "The order total is X more/less than this". **Use this total** copies it into
  *Order total* only when clicked, because a mismatch between the two is one of the model's strongest signals. The
  platform fee is left out, as it is in the data (see Assumptions).
* **Buttons.** **Check this order** sends the order; while the request runs it is disabled and shows a spinner and
  "Checking…". **Clear** empties the form and the result. **Try an example** has *Ordinary order* (row 0,
  ORD0020474) and *Unusual order* (row 135, ORD0000170, where the total is 2,465.67 but the parts add up to 425.05).
  Both are real rows from `orders-sheet.csv`; `tests/test_ui.py` checks them against the CSV and the API.
* **Result panel.** A large verdict: *Looks normal* (✓ in a green circle) or *Needs a closer look* (! in an amber
  diamond). The icon, shape and words all differ, so the verdict doesn't rely on colour. Below it, a score ruler
  shows the score as a filled bar with a pin and the threshold as an "Alert line" marker on a 0.3–0.8 scale, captioned
  "Higher score = more unusual". Then *What stands out* lists the three reasons as sentences, followed by "This is a
  statistical warning, not proof of fraud." A **Details** expander shows the exact score, the alert line, the model
  version, the request reference (`x-request-id`) and the time of the check.
* **Recent checks.** Every check on this page, newest first, with order reference, score and verdict. Clicking one
  shows its result again. The list is held in memory only and disappears on reload.
* **Error states.** Field problems appear under the field in plain words, on leaving the field and on submit. The
  form also shows a count at the top and moves focus to the first problem. If the server still answers 422, each
  `details[].field` is mapped to its field and shown there; anything it can't map goes in the form summary as text,
  never raw JSON. 503 shows "The checking service is not ready. Please try again shortly." 500 (or another error)
  shows an apology and the request reference for support. A network failure shows "Cannot reach the service."
  When the model isn't loaded, the form stays disabled and the result panel says why.

Layout: one column on phones, and the form beside a sticky result panel from 960 px wide. Light and dark themes
follow the system setting (`prefers-color-scheme`); every text/background pair is at least 4.5:1 (lowest 5.6:1).
Labels are tied to inputs, hints and errors are linked with `aria-describedby`, invalid fields get `aria-invalid`,
and the result panel and errors are `aria-live`. The page is fully keyboard usable with a visible focus ring, and
motion is reduced when the system asks for it. Every server-provided string goes into the page with `textContent`;
`app.js` never builds HTML from strings (a test checks this).

## Request flow

```
form (app.js)
  validate()         same bounds as app/schemas.py; plain-language errors; blocks submit
  buildPayload()     every key present; empty optional -> null; whole numbers as JSON ints; time + "Z" (UTC)
  callApi()          fetch POST /analyze-order (same origin, Content-Type: application/json)
    -> middleware (request_id, 415/413) -> OrderRequest schema (422) -> features.validate_orders (422)
    -> detect_anomalies.score_orders -> Pipeline[OrderFeatures, IsolationForest] -> threshold -> status
    -> features.explain(top 3) -> app/reasons.py wording
  renderResult()     verdict, score ruler, reasons, details; added to Recent checks
  renderErrors()     422 -> next to fields; 503 / 500 / network -> one plain message (+ request reference)
```

## Manual QA checklist

Start the server as above and open http://localhost:8000/.

- [ ] **Service ready**: the status pill is green and "How does this work?" shows the model name and version.
- [ ] **Valid normal order**: *Try an example → Ordinary order*. Expected total shows "Matches the order total".
      *Check this order* gives **Looks normal**, score 0.38, the pin left of the alert line, and 3 reasons. It
      appears in Recent checks.
- [ ] **Valid suspicious order**: *Unusual order*. Expected total shows 425.05 and "2,040.62 more". The result is
      **Needs a closer look**, score 0.63, and the first reason is about the order total.
- [ ] **Recent checks**: click the first entry and its result comes back.
- [ ] **Use this total**: change the unit price, check that *Order total* doesn't change on its own, then click
      *Use this total* and check that it does.
- [ ] **Validation errors**, each shown under its field when you leave it and blocking submit:
  - [ ] a required field left empty ("This is needed to check the order.")
  - [ ] a dropdown left on "Choose…"
  - [ ] Order reference `bad id!` (letters, numbers, - and _ only)
  - [ ] Customer age `17` or `121`; Account age `-1`
  - [ ] Quantity `0` or `1.5` (whole number from 1 to 1,000,000)
  - [ ] Discount `96` (0 to 95)
  - [ ] Price per item or Order total `0` (must be more than 0)
  - [ ] Shipping, Tax or Platform fee `-1` (can't be negative)
  - [ ] Order time in the future
  - [ ] Submitting with several errors: the summary shows the count and focus moves to the first error
- [ ] **Server-side 422**: in the browser console, run
      `renderErrors(await callApi('/analyze-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({...buildPayload(), quantity: 0})}))`
      and check the message appears under Quantity, with no JSON shown.
- [ ] **Service stopped**: load the page, stop uvicorn, then submit. You see "Cannot reach the service." and the
      pill turns red.
- [ ] **Bad model path**: start with `MODEL_PATH=missing.joblib APP_ENV=development`. The pill shows "Service
      unavailable", the form and examples are disabled, and the result panel says the service is not ready.
      (`/analyze-order` answers 503 with "The checking service is not ready…".) With `APP_ENV=production` the
      server refuses to start.
- [ ] **Keyboard only**: Tab through the whole page. Every control shows a focus ring, Enter submits, and
      Details / How does this work open with Enter or Space.
- [ ] **Dark mode**: switch the OS theme and check the page follows it and everything stays readable.
- [ ] **Phone width** (about 375 px): one column, no sideways scrolling, and the result appears below the form.

## Assumptions

* **Expected total excludes the platform fee.** The brief listed the fee as an input to the expected total, but
  the model's expected value (`OrderFeatures._numeric_inputs`) is `qty × price × (1 − discount/100) + shipping +
  tax`, and 97.6 % of rows in `orders-sheet.csv` match that to within 0.02. Adding the fee would make nearly every
  normal order look mismatched.
* When discount is empty it counts as 0 in the expected total (the model imputes 0 as well). When shipping is
  empty, no expected total is shown, because the model imputes the median and then ignores the total mismatch.
* The order time picker is in **UTC**. The API treats a time without a zone as UTC, and the UI sends it with `Z`.
* The score ruler runs from 0.3 to 0.8, the documented score range, and clamps scores outside it. The exact score
  is in Details.
* The wording of the reasons and their strength bands (< 2, 2–5, ≥ 5 robust z) are presentation choices in
  `app/reasons.py`. The ranking is Task 1's.
* Money amounts are shown with 2 decimals and no currency symbol, because the data has no currency column.
* `customer_id` and `product_name` are optional in the API and sent as `null` when empty. `customer_age`,
  `discount_pct`, `shipping_cost`, `device_type` and `ip_country` must be present but accept `null` ("Not known" /
  empty).
* The page loads the Atkinson Hyperlegible font from Google Fonts. Offline, it falls back to system fonts.

---

# Batch checking

Check a **whole CSV file** in one request and get a report you can page through, filter and download. It uses the
same schema, the same feature function and the same model as `/analyze-order`, so every row gets exactly the score
`/analyze-order` would give it (tested on 60 real rows, and against Task 1 on all 150,000 rows).

```
upload -> file checks -> header check -> row validation -> feature function -> model -> report
          (.csv, type,    (names, dups,    (OrderRequest,    (validate_orders,   (score_orders,  (REPORTS_DIR/<uuid4>/)
           text, size,     required         per row; a bad    OrderFeatures,      threshold,
           UTF-8, rows)    columns)         row never stops   in chunks of        explain)
                                            the others)       BATCH_CHUNK_SIZE)
```

## Required columns

Defined once as `REQUIRED_FIELDS` / `REQUIRED_COLUMNS` in `app/schemas.py`. The web UI shows the same list, which
`GET /` reads from the server.

| | Columns | Rule |
|---|---|---|
| **Required, never blank** (`REQUIRED_FIELDS`) | `order_id`, `quantity`, `unit_price`, `tax_amount`, `platform_fee`, `order_value`, `account_age_days`, `product_id`, `category` | What `features.py` needs: its `REQUIRED_COLUMNS` minus the two inputs it imputes |
| | `order_timestamp`, `customer_country`, `payment_method`, `sales_channel`, `order_status`, `coupon_used` | Not used by the model, but the shared `OrderRequest` schema (also used by `/analyze-order`) does not accept them blank |
| **Required column, cells may be blank** | `discount_pct`, `shipping_cost` | `features.py` needs the columns (`REQUIRED_COLUMNS`) but imputes blank cells (`NULLABLE_INPUTS`) |
| **Optional** (column may be absent) | `customer_id`, `product_name`, `customer_age`, `ip_country`, `device_type` | Blank or absent = null |

* A required **column** absent from the header → the whole file is rejected (422 `missing_columns`).
* A blank **cell** in a required field → that row is `missing_data`.
* Header names are stripped and lower-cased, so `" Order_ID "` matches `order_id`. Column order doesn't matter.
* Other columns are ignored, listed in `unknown_columns`, and kept in the CSV report.
* The columns a report adds (`row_number`, `anomaly_score`, `status`, `reasons`, `missing_fields`,
  `validation_errors`, `duplicate_order_id`) are ignored on upload and recalculated, so a downloaded report can
  be uploaded again (`ignored_columns`).
* Allowed values are the same as `/analyze-order` (see `GET /model-info → allowed_values`). They are matched
  case-insensitively, so `toys` and ` web ` are fine.

## Status of each row (exactly one)

| status | meaning | scored? |
|---|---|---|
| `normal` | score ≤ threshold (same threshold as `/analyze-order`) | yes |
| `suspicious` | score > threshold | yes |
| `missing_data` | one or more required fields are blank; `missing_fields` lists them | no (`anomaly_score` empty) |
| `invalid_data` | every required field is present, but a value fails validation: wrong type, out of bounds, unknown category, bad or future timestamp, NaN/Infinity, or more values than columns. `validation_errors` lists field + plain reason | no |

A `missing_data` row also lists any other validation problems in `validation_errors`, so everything can be fixed
at once. Rows that share an `order_id` with another row are all scored (as in Task 1) and get
`duplicate_order_id = yes`. Blank lines are skipped. `row_number` counts data rows from 1 (header and blank lines
excluded), so for `orders-sheet.csv`, `row_number = Task 1 row_id + 1`.

## Limits

| Limit | Default | Answer |
|---|---|---|
| File size | `MAX_UPLOAD_BYTES` = 50 MB | 413 `payload_too_large` "The file is larger than the 50 MB limit." |
| Rows | `MAX_UPLOAD_ROWS` = 200,000 | 413 `too_many_rows` "The file has more than 200,000 orders. Split it…" |
| Report page size | 200 | 422 above it |

**Speed (measured).** The full `orders-sheet.csv` (150,000 rows, 23.9 MB) takes **17.5 s** on the server
(17.7 s wall time with curl) and is processed synchronously. That is under the 30 s budget, so there is no
background-job mode. Paging a 150k report takes about 20 ms (the report is cached in memory after the first page),
the CSV download 0.3 s, and the HTML download 0.02 s. The JSON download is built on first request (5 s, 138 MB)
and then reused. On disk, a 150k report takes about 55 MB, plus 138 MB once its JSON has been downloaded.

## Template

`GET /template.csv` returns every expected column (in the order of `orders-sheet.csv`) plus one real example row.
It is utf-8-sig, so Excel opens it cleanly.

```
order_id,order_timestamp,customer_id,customer_country,customer_age,account_age_days,product_id,product_name,category,quantity,unit_price,discount_pct,shipping_cost,tax_amount,platform_fee,order_value,payment_method,sales_channel,device_type,ip_country,order_status,coupon_used
ORD0020474,2025-09-12 02:16:00,C009777,France,52,806,P01005,HOM Product 1005,Home,1,22.74,20.0,5.37,0.25,2.79,23.81,PayPal,Mobile App,Android,Italy,Shipped,No
```

## Endpoints (real requests/responses from a live run)

### `POST /analyze-batch`

`multipart/form-data` with the CSV in a field named `file`. The example file has 7 rows from `orders-sheet.csv`:
two normal (the same order twice), two suspicious, one with a blank `quantity`, one with category `Furniture`, one
with discount 120, plus an extra `notes` column.

```bash
curl -F "file=@mixed-orders.csv;type=text/csv" localhost:8000/analyze-batch
```
```json
{
  "report_id": "4c2297c3-f2fa-43b9-b973-5ac80d684125",
  "filename": "mixed-orders.csv",
  "uploaded_at": "2026-10-05T10:53:41+00:00",
  "model_version": "iforest-reduced11-20261005",
  "threshold": 0.5030621506497381,
  "summary": {"total_rows": 7, "normal": 2, "suspicious": 2, "missing_data": 1, "invalid_data": 2,
              "scored_rows": 4, "suspicious_pct": 50.0, "duplicate_order_id_rows": 2},
  "unknown_columns": ["notes"],
  "ignored_columns": [],
  "duration_ms": 128.2,
  "rows": {"report_id": "4c2297c3-…", "status": null, "sort": "score", "page": 1, "page_size": 50,
           "total_rows": 7, "total_pages": 1,
           "rows": [
             {"row_number": 2, "order_id": "ORD0000170", "anomaly_score": 0.6289455449115257, "status": "suspicious",
              "reasons": ["The order total is much higher than price, discount, shipping and tax add up to.",
                          "The discount is much higher than usual.",
                          "Tax is noticeably higher than usual for an order of this size."],
              "missing_fields": [], "validation_errors": [], "duplicate_order_id": false,
              "values": {"order_id": "ORD0000170", "product_id": "P00590", "category": "Office",
                         "order_value": "2465.67", "customer_country": "UK", "sales_channel": "Mobile App"}},
             "… 5 more …",
             {"row_number": 4, "order_id": "ORD0051725", "anomaly_score": null, "status": "missing_data",
              "reasons": [], "missing_fields": ["quantity"], "validation_errors": [], "duplicate_order_id": false,
              "values": {"…": "…"}}]},
  "links": {"rows": "/reports/4c2297c3-…/rows", "csv": "/reports/4c2297c3-…/download?format=csv",
            "html": "/reports/4c2297c3-…/download?format=html", "json": "/reports/4c2297c3-…/download?format=json"}
}
```

### `GET /reports/{report_id}/rows?status=&page=&page_size=&sort=`

`status` is one of the four statuses, or empty for all rows. `page` starts at 1. `page_size` is 1–200 (default 50).
`sort` is `score` (default: highest `anomaly_score` first, unscored rows last) or `row` (file order).

```bash
curl "localhost:8000/reports/4c2297c3-f2fa-43b9-b973-5ac80d684125/rows?status=invalid_data&page=1&page_size=2"
```
```json
{"report_id":"4c2297c3-f2fa-43b9-b973-5ac80d684125","status":"invalid_data","sort":"score","page":1,"page_size":2,
 "total_rows":2,"total_pages":1,"rows":[
  {"row_number":5,"order_id":"ORD0052945","anomaly_score":null,"status":"invalid_data","reasons":[],"missing_fields":[],
   "validation_errors":[{"field":"category","reason":"is not one of the accepted values: Automotive, Beauty, Electronics, Fashion, Garden, Home, Office, Sports, Tools, Toys"}],
   "duplicate_order_id":false,"values":{"order_id":"ORD0052945","product_id":"P00692","category":"Furniture","order_value":"116.87","customer_country":"Germany","sales_channel":"Marketplace"}},
  {"row_number":6,"order_id":"ORD0087395","anomaly_score":null,"status":"invalid_data","reasons":[],"missing_fields":[],
   "validation_errors":[{"field":"discount_pct","reason":"must be at most 95"}],
   "duplicate_order_id":false,"values":{"order_id":"ORD0087395","product_id":"P00372","category":"Tools","order_value":"48.75","customer_country":"US","sales_channel":"Mobile App"}}]}
```

### `GET /reports/{report_id}/download?format=csv|html|json`

```bash
curl -OJ "localhost:8000/reports/4c2297c3-f2fa-43b9-b973-5ac80d684125/download?format=csv"
# content-type: text/csv; charset=utf-8
# content-disposition: attachment; filename="mixed-orders-report.csv"
```
```
order_id,order_timestamp,…,coupon_used,notes,row_number,anomaly_score,status,reasons,missing_fields,validation_errors,duplicate_order_id
ORD0020474,2025-09-12 02:16:00,…,No,,1,0.3790653804547756,normal,The discount is slightly higher than usual.; Tax is slightly lower than usual for an order of this size.; The item price is slightly lower than this product normally sells for.,,,yes
ORD0051725,2026-01-12 19:59:00,…,No,,4,,missing_data,,quantity,,no
ORD0052945,2025-03-18 18:35:00,…,No,,5,,invalid_data,,,category: is not one of the accepted values: Automotive, …,no
```

* **csv**: every original column (unknown ones included) in file order, plus the 7 report columns. Lists inside a
  cell are joined with `; `. It is utf-8-sig so Excel shows accents correctly. The full-precision `anomaly_score`
  matches `/analyze-order` exactly.
* **html**: one self-contained file with inline CSS and an inline SVG chart, no scripts and no external requests.
  It contains the summary cards, the file name, upload time, file size, SHA-256 of the uploaded bytes, model
  version, threshold and processing time, a score histogram with the alert line, the top 20 suspicious orders with
  their reasons, flagged orders broken down by category, customer country and sales channel, and the first 500
  rows that couldn't be checked, with their reasons. Light and dark mode follow the viewer's system.
* **json**: `{report_id, filename, uploaded_at, sha256, model_version, threshold, summary, unknown_columns,
  ignored_columns, rows: [{row_number, anomaly_score, status, reasons, missing_fields, validation_errors,
  duplicate_order_id, values: {every original column}}]}`.

**Formula injection.** In every export, a text cell starting with `=`, `+`, `-`, `@`, tab or carriage return gets a
leading apostrophe (`=cmd()` → `'=cmd()`), so Excel and Sheets show it as text instead of running it. Plain
numbers such as `-5` are left alone because a spreadsheet reads them as numbers.

### Errors

Every failure uses the standard error envelope with `request_id`, and no stack traces are sent.

```bash
curl -F "file=@short.csv" localhost:8000/analyze-batch          # a file with only order_id,quantity
```
```json
{"error":{"code":"missing_columns","message":"The file is missing required columns: unit_price, tax_amount, platform_fee, order_value, account_age_days, product_id, category, order_timestamp, customer_country, payment_method, sales_channel, order_status, coupon_used, discount_pct, shipping_cost. Download the template to see every expected column.","request_id":"583745a118474db89fe9bcb8f535611e","details":[{"field":"unit_price","reason":"required column is missing"},"…"]}}
```

| Status | code | When |
|---|---|---|
| 400 | `bad_upload` | malformed multipart body, or more than one file |
| 404 | `not_found` | unknown, malformed or expired `report_id` (also path-traversal attempts) |
| 413 | `payload_too_large` | file over `MAX_UPLOAD_BYTES` |
| 413 | `too_many_rows` | more than `MAX_UPLOAD_ROWS` orders |
| 415 | `unsupported_media_type` | request `Content-Type` is not `multipart/form-data` |
| 415 | `unsupported_file_type` | file name doesn't end in `.csv`, or the file part is labelled e.g. `image/png` |
| 415 | `not_text` | binary content (NUL or control bytes), e.g. an `.xlsx` renamed to `.csv` |
| 422 | `no_file` | no form field named `file` |
| 422 | `empty_file` / `header_only` | nothing in the file / column names but no rows |
| 422 | `bad_encoding` | not UTF-8 (tried utf-8-sig, then utf-8) |
| 422 | `duplicate_columns` | two headers that are the same after strip + lower-case |
| 422 | `missing_columns` | a required column is absent |
| 422 | `bad_csv` | the CSV parser failed (the message gives the line number) |
| 422 | `validation_error` | bad `status`, `page`, `page_size`, `sort` or `format` query value |
| 503 | `model_unavailable` | the model isn't loaded (checked before reading the upload) |
| 500 | `internal_error` | anything unexpected (logged with traceback; the client only gets the request_id) |

Accepted file-part types: `text/csv`, `application/csv`, `text/x-csv`, `application/x-csv`, `text/plain`,
`text/comma-separated-values`, `application/vnd.ms-excel` (Excel on Windows), `application/octet-stream` and none
(curl and some browsers). Whatever the label, the content must be UTF-8 text.

## Report storage and retention

* Each report is a folder `REPORTS_DIR/<report_id>/` holding `meta.json` (summary, file details, chart data, top
  20, breakdowns, first 500 problem rows), `report.csv` (every row) and, after the first JSON download,
  `report.json`. There is no database.
* `report_id` is a random UUID4. Every request checks it against a strict pattern before touching the disk, so an
  id can never be a path. Unknown, malformed and expired ids all answer 404.
* A report is built in a hidden `.tmp-<uuid>` folder and renamed into place only when complete. A rejected or
  failed upload leaves nothing in `REPORTS_DIR` (tested for every rejection). Leftover temp folders are removed at
  startup. The multipart spool file is closed and deleted after each upload.
* Reports older than `REPORT_TTL_MINUTES` are deleted at startup, by a background check every
  min(TTL, 5 minutes), and when someone opens an expired one. Beyond `MAX_REPORTS`, the oldest are deleted.
* Logs contain only report_id, file name, row and status counts, duration and request_id (and the error code for
  rejections). They never contain row contents; a test checks this.

## Web UI: whole file

The page has two tabs at the top: **Single order** (the form described above) and **Whole file**. Use arrow keys,
Home or End to move between them. The address shows `#file` while on the Whole file tab, so a reload keeps it.

* **Single order → Upload order details.** Accepts one `.json` (an object, or an array of one object) or a
  one-row `.csv`. It fills the form: labels are matched case-insensitively, times with a zone are converted to UTC,
  and `true`/`false` become Yes/No. Then it marks missing or unusable values next to their fields, lists columns it
  didn't use, and moves focus to the first problem. It never submits; you review and press **Check this order**.
* **Whole file.** Drag a CSV onto the box or choose one; the file name and size are shown. Before uploading, the
  page checks the `.csv` name, that the file isn't empty, and the size limit. **Download template** gets
  `/template.csv`. *Columns the file needs* lists the required columns, from the server, with plain names.
  **Check this file** is disabled while a file is being checked and shows a spinner and progress text.
* **Results.** Cards for total rows, *Looks normal* (✓), *Needs a closer look* (! in a square, with the percentage
  of checked orders), *Missing data* (? dashed) and *Invalid data* (✕ dashed). Icons, shapes and words differ, so
  colour isn't the only signal. Notes cover rows that couldn't be checked, duplicate order references and unused
  columns. Three download buttons: full report (CSV), summary report (HTML) and data (JSON).
* **Table.** Filter by status, 50 rows per page with Previous/Next, highest score first. Each row has a score bar
  with the alert line. Click a row (or its order button, which works with the keyboard) to see its three reasons,
  or which values are missing or invalid, in plain sentences.
* **Errors.** Rejected files show the server's plain message plus a list of fields (for example each missing
  column with its plain name), never raw JSON. 503, 500 (with the request reference) and "Cannot reach the service"
  messages work as on the Single order tab.

## Manual QA: batch

Start the server and open http://localhost:8000/#file. Make the test files from `orders-sheet.csv` (for example in
Excel or with `head`).

- [ ] **Template**: *Download template* saves `order-template.csv`. Uploading it gives 1 row, *Looks normal*.
- [ ] **Valid file**: the first 100 rows of `orders-sheet.csv` give 100 checked rows. The table shows the highest
      score first, and the downloads work.
- [ ] **Mixed file**: blank one `quantity`, set one `category` to `Furniture` and one `discount_pct` to `120`. All
      four cards are non-zero. Filter *Missing data* and click the row: "Quantity (quantity)". Filter *Invalid data*:
      "Product category (category) is not one of the accepted values…" and "Discount (%) (discount_pct) must be at
      most 95."
- [ ] **Duplicates**: copy one row twice. Both rows are scored with the same score, and the note says 2 rows share
      their order reference.
- [ ] **Unknown column**: add a `notes` column. It appears under "Columns not used" and in the CSV report.
- [ ] **Re-upload**: upload the downloaded CSV report. You get the same counts, and the note says result columns
      were recalculated.
- [ ] **Full file**: upload `orders-sheet.csv`. It takes about 20 s with a spinner and the button disabled, then
      shows 147,298 normal, 2,702 need a closer look (1.80 %), 0 missing, 0 invalid.
- [ ] **HTML report**: open the downloaded `.html` with the network off. Everything shows: cards, chart, top 20,
      breakdowns and problem rows.
- [ ] **Formula safety**: set a `product_name` to `=1+1` and upload. In the CSV report opened in Excel, the cell
      shows the text `'=1+1`, not 2.
- [ ] **Rejections**, each showing a plain message with nothing added to `REPORTS_DIR`:
  - [ ] a `.xlsx` or `.txt` file: refused in the browser ("Only .csv files can be checked…")
  - [ ] an empty `.csv`: refused in the browser ("This file is empty.")
  - [ ] a file larger than `MAX_UPLOAD_BYTES` (try `MAX_UPLOAD_BYTES=1024`): refused in the browser. With curl:
        413 "The file is larger than the … limit."
  - [ ] more rows than `MAX_UPLOAD_ROWS` (try `MAX_UPLOAD_ROWS=5`): "The file has more than 5 orders…"
  - [ ] header only: "The file has column names but no orders."
  - [ ] a required column deleted: "The file is missing required columns: …" with each column listed
  - [ ] two columns with the same name (e.g. `quantity` and `Quantity`): "Some column names appear more than once…"
  - [ ] a picture renamed to `.csv`: "This file doesn't look like a CSV text file…"
  - [ ] a file saved as ANSI / Latin-1 with an accent: "The file isn't UTF-8 text…"
  - [ ] with curl, `-F "file=@x.csv;type=image/png"` gives 415, and `-H "Content-Type: text/csv" --data-binary @x.csv`
        gives 415 "Content-Type must be multipart/form-data."
  - [ ] with curl, `-F "other=@x.csv"` gives 422 "No file was uploaded…"
- [ ] **Expired report**: start with `REPORT_TTL_MINUTES=1`, check a file, wait 2 minutes, then change the filter.
      You see "This report has expired. Check the file again…"
- [ ] **Service stopped / model missing**: same as the Single order checks. *Check this file* stays disabled while
      the model isn't loaded.
- [ ] **Single order upload**: *Upload order details* with a `.json` of one order (one value wrong, one missing).
      The form is filled, both fields are marked, and nothing is sent until *Check this order*. A two-row CSV gives
      "This file has 2 orders. Use the Whole file tab…"
- [ ] **Keyboard and screen reader**: Tab to the tabs and use the arrow keys. Tab to the file box (focus ring
      visible) and press Enter to browse. In the table, Tab through the order buttons and press Enter to open the
      details. Results and errors are announced.

## Assumptions (batch)

* **Required = what the shared schema requires.** `features.py` truly needs 9 non-blank fields, plus the
  `discount_pct` and `shipping_cost` columns. Because batch must use the same `OrderRequest` as `/analyze-order`,
  which doesn't accept blank `order_timestamp`, `customer_country`, `payment_method`, `sales_channel`,
  `order_status` or `coupon_used`, those 6 context fields are required too. None of them is blank in
  `orders-sheet.csv`, so this doesn't change the Task 1 reproduction.
* Whole-number fields accept `52.0` (how pandas writes an int column that has blanks) but not `1.5`. Numbers must
  use ASCII digits, so `1_000` and `١٢` are `invalid_data` ("must be a number" / "must be a whole number").
* `-5`-style plain numbers are not prefixed in exports; only non-numeric text starting with `= + - @` (or tab/CR).
* A row with more values than the header is `invalid_data`; a row with fewer is padded with blanks.
* `suspicious_pct` is the share of **checked** (scored) rows, not of all rows.
* Accepted file-part content types are listed above; `application/octet-stream` is accepted because curl and some
  browsers send it for CSV.
* The report keeps cell text stripped of surrounding whitespace. Header names are written lower-cased.
* Tests that create the app (e.g. `tests/test_api.py`) use the default `REPORTS_DIR=reports`, which is created in
  the project folder if it doesn't exist.
