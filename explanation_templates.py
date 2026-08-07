FEATURE_NAMES = {
    0: {"name": "Recent Demand Trend", "description": "How customer demand has been changing over the last 3 periods"},
    1: {"name": "Profit Trajectory", "description": "Whether the firm's profitability is rising or falling"},
    2: {"name": "Price Level", "description": "The firm's current pricing relative to cost"},
    3: {"name": "Market Price Index", "description": "Overall price level in the economy (inflation signal)"},
    4: {"name": "Interest Rate Environment", "description": "Central bank rate affecting borrowing and investment costs"},
}

OUTPUT_NAMES = {
    0: {"name": "Price Adjustment", "unit": "%", "positive": "increase price", "negative": "decrease price"},
    1: {"name": "Wage Adjustment", "unit": "%", "positive": "raise wages", "negative": "cut wages"},
    2: {"name": "Production Target", "unit": "units", "positive": "increase production", "negative": "reduce production"},
}

SUPPLY_CHAIN_TEMPLATES = {
    "recommendation": "Based on current market conditions, the model recommends {action} by {magnitude:.1f}{unit}.",
    "primary_driver": "The primary driver is {feature_name}: {description}. This factor has a {direction} influence of {sensitivity:.1f}% on the recommendation.",
    "risk_if_ignored": "If this adjustment is not made, {risk_scenario}.",
    "historical_parallel": "A similar pattern was observed during {episode}, where {outcome}.",
}

INSURANCE_TEMPLATES = {
    "recommendation": "Based on current market conditions, the model recommends {action} by {magnitude:.1f}{unit}.",
    "primary_driver": "The primary driver is {feature_name}: {description}. This factor has a {direction} influence of {sensitivity:.1f}% on the recommendation.",
    "risk_if_ignored": "If this adjustment is not made, {risk_scenario}.",
    "historical_parallel": "A similar pattern was observed during {episode}, where {outcome}.",
}

POLICY_TEMPLATES = {
    "recommendation": "Based on current market conditions, the model recommends {action} by {magnitude:.1f}{unit}.",
    "primary_driver": "The primary driver is {feature_name}: {description}. This factor has a {direction} influence of {sensitivity:.1f}% on the recommendation.",
    "risk_if_ignored": "If this adjustment is not made, {risk_scenario}.",
    "historical_parallel": "A similar pattern was observed during {episode}, where {outcome}.",
}
