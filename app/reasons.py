"""Plain-language wording for the Task 1 explanations (features.explain).

No new logic: features.explain picks the top-3 features by |robust z| exactly as score_order.py does;
this module only turns each (feature, z) pair into a sentence a non-technical reader understands.
"""

# feature -> (sentence when the value is above a typical order, sentence when it is below)
PLAIN = {
    "log_quantity": ("More items were bought than usual", "Fewer items were bought than usual"),
    "log_unit_price": ("The item price is {} higher than usual", "The item price is {} lower than usual"),
    "discount_pct": ("The discount is {} higher than usual", "The discount is {} lower than usual"),
    "log_shipping": ("Shipping cost is {} higher than usual", "Shipping cost is {} lower than usual"),
    "log1p_tax_rate": ("Tax is {} higher than usual for an order of this size",
                       "Tax is {} lower than usual for an order of this size"),
    "log_fee_rate": ("The platform fee is {} higher than usual for an order of this size",
                     "The platform fee is {} lower than usual for an order of this size"),
    "price_vs_product": ("The item price is {} higher than this product normally sells for",
                         "The item price is {} lower than this product normally sells for"),
    "log_order_value": ("The order total is {} larger than a typical order",
                        "The order total is {} smaller than a typical order"),
    "log_value_ratio": ("The order total is {} higher than price, discount, shipping and tax add up to",
                        "The order total is {} lower than price, discount, shipping and tax add up to"),
    "log_acct_age": ("The customer account is older than usual", "The customer account is {} newer than usual"),
    "log_ship_ratio": ("Shipping is {} higher than usual compared with the order size",
                       "Shipping is {} lower than usual compared with the order size"),
    "log_fee_to_value": ("The platform fee is {} higher than usual compared with the order total",
                         "The platform fee is {} lower than usual compared with the order total"),
}


def _strength(z: float) -> str:
    """How far from typical, in words (z = typical spreads away from the training median)."""
    size = abs(z)
    if size >= 5:
        return "much"
    if size >= 2:
        return "noticeably"
    return "slightly"


def to_sentence(feature: str, z: float) -> str:
    """One (feature, z) pair from features.explain -> one plain sentence."""
    high, low = PLAIN.get(feature, (f"{feature} is higher than usual", f"{feature} is lower than usual"))
    template = high if z >= 0 else low
    return " ".join(template.format(_strength(z)).split()) + "."
