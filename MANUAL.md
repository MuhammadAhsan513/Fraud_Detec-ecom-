# Manual — how to find the anomaly score of any order or product

The model scores **orders**. A product has no score of its own: "the score of a product" means the scores of the
orders that contain it. This manual shows every way to look a score up.

---

## 0. One-time setup

```bash
cd "/home/ahsan/AI/Fraud_Detec(ecom)"
source .venv/bin/activate           # Python env with the pinned libraries
ls model.joblib                     # the trained model must exist (else: python train.py --data orders-sheet.csv --out model.joblib)
```

## 1. The three identifiers

| Identifier | Example | Unique? | Use it for |
|---|---|---|---|
| `order_uid` | `UCA5E4916EB5B` | **yes, always** | pointing at exactly one order |
| `row_id` | `0` | yes, within one file | position in the file (0 = first data row) |
| `order_id` | `ORD0020474` | **no**: only 74,002 distinct values in 150,000 rows | searching; it may return several orders |
| `product_id` | `P01005` | – (1,200 products) | all orders of a product |

**How `order_uid` is made:** `"U"` + the first 12 hex characters of a SHA-1 hash of the order's 22 original fields.
* The same order content always gets the same uid, in any file, at any position, and also when scored alone.
* 350 rows in the data are exact copies of an earlier row. The first copy keeps the plain uid; the repeat gets `-2`
  (e.g. `U1A2B3C4D5E6F-2`).
* If any field of an order changes, its uid changes: the uid identifies *this exact order record*.

## 2. Reading a result

```
anomaly_score 0.672511 -> SUSPICIOUS   reasons: log_value_ratio (z=+569.0), log_fee_to_value (z=-14.9), discount_pct (z=+6.7)
```
* **anomaly_score** runs from about 0.34 to 0.78. **Higher = more unusual.** About 0.5 is average; normal orders are mostly 0.35–0.45.
* **status** is `SUSPICIOUS` if score > **0.50306** (the threshold stored in the model; 1.8 % of orders), otherwise `NORMAL`.
  Scores above **0.564** are the high-confidence band, which is almost always a clear rule break.
* **reasons** are the 3 features furthest from a typical order. z is the number of "typical spreads" away from the median;
  |z| > 4 is unusual, and the sign gives the direction.

| Reason feature | Positive z means | Negative z means |
|---|---|---|
| `log_value_ratio` | charged **more** than qty×price×(1−disc)+ship+tax | charged **less** than the formula |
| `price_vs_product` | price above this product's usual price | price below it |
| `log_fee_rate` / `log_fee_to_value` | platform fee too large | fee tiny compared with the order |
| `log_acct_age` | – | very new account |
| `discount_pct` | unusually high discount | – |
| `log_quantity`, `log_unit_price`, `log_shipping`, `log_order_value`, `log1p_tax_rate` | unusually high | unusually low |

For `log_value_ratio`, z values in the hundreds are normal (clean orders sit *exactly* on the formula). Read it as
"the total does not match the formula", and compare `order_value` with the formula yourself.

---

## 3. Method A: the lookup tool (`score_order.py`)

It scores live with the saved model and prints the order, its score, its status and the top-3 reasons.

### 3.1 One order by its unique id
```bash
python score_order.py --uid UCA5E4916EB5B
```
```
  row_id 0 | order_uid UCA5E4916EB5B | order_id ORD0020474
    product P01005 (Home) | qty 1 @ 22.74 (product median 26.32) | discount 20.0 | shipping 5.37 | tax 0.25 | fee 2.79 | order_value 23.81 | account 806 d
    anomaly_score 0.379065 -> NORMAL   reasons: discount_pct (z=+1.8), log1p_tax_rate (z=-1.2), price_vs_product (z=-1.0)
```

### 3.2 One order by row position
```bash
python score_order.py --row-id 20465
```

### 3.3 By order_id (may return several different orders)
```bash
python score_order.py --order-id ORD0072849
```
This returns **3 different orders** (Garden, Office, Tools) that happen to share this id: one suspicious, two normal.
Use the `order_uid` printed next to each one to refer to a single order.

### 3.4 Every order of a product
```bash
python score_order.py --product P00594 --top 3
```
```
product P00594: 126 orders | median unit price (model lookup) 10.73
  suspicious 5 | normal 121 | score min 0.3540 / median 0.3826 / max 0.6797

  top 3 most anomalous orders of this product:
  row_id 88995 | order_uid UC70D0F8E144B | ...  order_value 3374.11 | account 0 d
    anomaly_score 0.679749 -> SUSPICIOUS   reasons: log_value_ratio (z=+471.2), log_fee_to_value (z=-14.4), log_acct_age (z=-8.9)
  ...
```
`--top N` controls how many orders are listed. The summary line always covers all orders of the product.

### 3.5 A new order that is not in the file
Pass the fields as JSON. Required: `product_id, category, quantity, unit_price, discount_pct, shipping_cost,
tax_amount, platform_fee, order_value, account_age_days`. Optional: `order_id` and the other columns
(they only change the uid, not the score). `discount_pct` / `shipping_cost` may be `null`.
```bash
python score_order.py --json '{"product_id":"P01005","category":"Home","quantity":1,"unit_price":22.74,
  "discount_pct":20,"shipping_cost":5.37,"tax_amount":0.25,"platform_fee":2.79,"order_value":4466.15,"account_age_days":3}'
```
```
    anomaly_score 0.648631 -> SUSPICIOUS   reasons: log_value_ratio (z=+523.4), log_fee_to_value (z=-14.3), log_acct_age (z=-7.2)
```
This is the same order as 3.1 but charged 4,466.15 instead of 23.81 from a 3-day-old account; the score jumps from 0.379 to 0.649.
A product not seen in training is allowed: its price is compared with the category median, then the global median.
Invalid input is rejected with a message, e.g. `ERROR: row 0: quantity must be >= 1`.

To search another file: `python score_order.py --data other_orders.csv --product P01005`.

---

## 4. Method B: the scored file (no model needed)

```bash
python detect_anomalies.py --model model.joblib --input orders-sheet.csv --output scored_orders.csv
```
Columns: `row_id, order_uid, order_id, anomaly_score, status`.

* **A spreadsheet** (LibreOffice/Excel): open `scored_orders.csv`, use the filter on `order_uid` or `order_id`, and sort by
  `anomaly_score` descending to see the most unusual orders first.
* **Command line:**
  ```bash
  grep UCA5E4916EB5B scored_orders.csv                 # one order
  grep ORD0072849 scored_orders.csv                    # all rows with this order_id
  ```
* **A product** (the scored file has no product column, so join on `row_id`):
  ```python
  import pandas as pd
  s = pd.read_csv("scored_orders.csv")
  o = pd.read_csv("orders-sheet.csv")
  s["product_id"] = o["product_id"]                    # same row order -> row_id aligns
  p = s[s.product_id == "P01005"].sort_values("anomaly_score", ascending=False)
  print(p.status.value_counts(), p.head(10), sep="\n")
  ```
  Never join on `order_id`: it repeats, so a join would multiply rows.

`flagged_explanations.csv` lists the 2,702 suspicious orders with their 3 reasons (written by `validation.ipynb`).

---

## 5. Method C: from Python (for your own code / the future API)

```python
import joblib, pandas as pd
from detect_anomalies import validate_orders, make_order_uid
from features import explain

bundle = joblib.load("model.joblib")
order = pd.DataFrame([{"order_id": "ORD0020474", "product_id": "P01005", "category": "Home", "quantity": 1,
                       "unit_price": 22.74, "discount_pct": 20.0, "shipping_cost": 5.37, "tax_amount": 0.25,
                       "platform_fee": 2.79, "order_value": 23.81, "account_age_days": 806}])

df, problems = validate_orders(order)                  # problems must be empty
score = -bundle["pipeline"].score_samples(df)[0]       # 0.379065
status = "suspicious" if score > bundle["threshold"] else "normal"
X = bundle["pipeline"].named_steps["features"].transform(df)
print(make_order_uid(order).iloc[0], score, status, explain(X, bundle["explain_stats"]))
```
(Here the uid differs from the one in the file, because only 11 of the 22 fields were given. The score is identical.)

---

## 6. Doing it by hand (to understand, not for daily use)

1. **Inputs:** sub = qty × price × (1 − disc/100); expected = sub + shipping + tax; pm = the product's median price
   (`bundle["pipeline"].named_steps["features"].prod_med_price_["P01005"]` = 26.32).
2. **11 features** (natural log `ln`):

   | Feature | Value for row 0 |
   |---|---|
   | `log_quantity` = ln(1+qty) | 0.6931 |
   | `log_unit_price` = ln(price) | 3.1241 |
   | `discount_pct` = disc | 20 |
   | `log_shipping` = ln(1+ship) | 1.8516 |
   | `log1p_tax_rate` = ln(1+tax/sub) | 0.0136 |
   | `log_fee_rate` = ln(fee/sub) | −1.8749 |
   | `price_vs_product` = ln(price/pm) | −0.1462 |
   | `log_order_value` = ln(value) | 3.1701 |
   | `log_value_ratio` = ln(value/expected) | −0.0001 |
   | `log_acct_age` = ln(1+age) | 6.6933 |
   | `log_fee_to_value` = ln(fee/value) | −2.1441 |

3. **Trees:** each of the 300 trees asks "feature ≤ value?" questions until the order lands in a leaf. Path length
   h = number of questions + c(points left in the leaf). Unusual orders land in a leaf quickly (short h).
4. **Score:** E[h] = average h over the 300 trees; c(4096) = 2(ln 4095 + 0.5772) − 2·4095/4096 = 15.79;
   **score = 2^(−E[h] / 15.79)**.
   * Row 0: E[h] = 22.10, so 2^(−22.10/15.79) = **0.379**: normal.
   * Row 20465: E[h] = 9.04, so **0.673**: suspicious.

Steps 1, 2 and 4 can be done with a calculator. Step 3 needs the trees, and the model does that walk exactly as described.

---

## 7. Troubleshooting

| Message | Cause / fix |
|---|---|
| `no matching order found` | wrong uid / id (check upper case; uids start with `U`), or the wrong `--data` file |
| `missing required column(s): [...]` | the CSV or JSON lacks a field the model needs (list in 3.5) |
| `row N: quantity must be >= 1` (or similar) | impossible value; extreme but valid values are allowed |
| `Can't get attribute 'OrderFeatures'` | run from the project folder so `features.py` is importable |
| scores differ from `scored_orders.csv` | the model was retrained; re-run `detect_anomalies.py` |
