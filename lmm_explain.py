import jax
import jax.numpy as jnp
from typing import Dict, Any, List

from lmm_model import FirmTransformer

def explain_firm_policy(params: Any, lmm_inputs: jnp.ndarray) -> Dict[str, Any]:
    """
    Computes local feature attribution for the FirmTransformer using JAX Jacobians.
    Provides a mathematical, causal explanation for why the LMM chose a specific policy.
    
    Args:
        params: The LMM weights PyTree.
        lmm_inputs: Shape (seq_len, feature_dim). The historical state of the firm.
            Features: [0: Demand, 1: Profit, 2: Price, 3: Macro Price, 4: Macro Rate]
            
    Returns:
        Dictionary containing causal explanations.
    """
    model = FirmTransformer()
    
    def forward_fn(x):
        # x is unbatched: (seq_len, feature_dim)
        # We need to add a batch dim for model.apply, then remove it from the output
        x_batched = jnp.expand_dims(x, 0)
        dp, dw, tp = model.apply({'params': params}, x_batched)
        
        # Squeeze batch dim out and stack outputs into a single vector of shape (3,)
        return jnp.stack([
            jnp.squeeze(dp),
            jnp.squeeze(dw),
            jnp.squeeze(tp)
        ])
    
    # Compute Jacobian of output (3,) w.r.t input (seq_len, feature_dim)
    # Output shape: (3, seq_len, feature_dim)
    jacobian = jax.jacfwd(forward_fn)(lmm_inputs)
    
    # We care about the sensitivity to the most recent timestep (index -1)
    # because that drives the immediate decision most strongly.
    recent_jacobian = jacobian[:, -1, :] # Shape: (3, feature_dim)
    
    feature_names = ["Demand", "Profit", "Firm Price", "Macro Price Index", "Macro Interest Rate"]
    output_names = ["Delta Price", "Delta Wage", "Target Production"]
    
    explanations = {}
    
    for i, out_name in enumerate(output_names):
        grads = recent_jacobian[i]
        
        # Build attribution map
        attribution = {feature_names[j]: float(grads[j]) for j in range(len(feature_names))}
        
        # Find the max magnitude contributor (the primary causal driver)
        primary_driver_idx = jnp.argmax(jnp.abs(grads))
        primary_driver = feature_names[int(primary_driver_idx)]
        primary_grad = float(grads[primary_driver_idx])
        
        direction = "increase" if primary_grad > 0 else "decrease"
        
        explanations[out_name] = {
            "attributions": attribution,
            "primary_driver": primary_driver,
            "primary_driver_gradient": primary_grad,
            "causal_reason": f"The LMM's decision for {out_name} was primarily driven by the {primary_driver}. A positive change in {primary_driver} causes an {direction} in {out_name}."
        }
        
    return explanations

def validate_policy_safety(gradients_dict: Dict[str, Any], max_gradient_magnitude: float = 50.0) -> bool:
    """
    Mathematical circuit breaker for AI hallucinations.
    
    If the Jacobian gradient (derivative) of the LMM output with respect to any 
    input feature exceeds a safe bound, the policy is flagged as unstable.
    
    For example, if max_gradient_magnitude is 50.0, it means the LMM suggests 
    changing production by >50 units for every 1 unit change in input demand, 
    which implies a massive, unsafe bullwhip amplification.
    """
    for out_name, exp in gradients_dict.items():
        primary_grad = exp["primary_driver_gradient"]
        if abs(primary_grad) > max_gradient_magnitude:
            return False # Unsafe: Gradient explodes beyond physical reality bounds
            
    return True # Safe

def generate_executive_explanation(params: Any, lmm_inputs: jnp.ndarray, vertical: str = "supply_chain") -> Dict[str, Any]:
    raw_explanations = explain_firm_policy(params, lmm_inputs)
    is_safe = validate_policy_safety(raw_explanations)
    
    from explanation_templates import FEATURE_NAMES, OUTPUT_NAMES, SUPPLY_CHAIN_TEMPLATES, INSURANCE_TEMPLATES, POLICY_TEMPLATES
    
    if vertical == "supply_chain":
        templates = SUPPLY_CHAIN_TEMPLATES
    elif vertical == "insurance":
        templates = INSURANCE_TEMPLATES
    else:
        templates = POLICY_TEMPLATES
        
    executive_explanations = {
        "safety_status": "safe" if is_safe else "unsafe (circuit breaker triggered)",
        "decisions": {}
    }
    
    out_map = {"Delta Price": 0, "Delta Wage": 1, "Target Production": 2}
    feature_name_to_idx = {
        "Demand": 0, "Profit": 1, "Firm Price": 2, "Macro Price Index": 3, "Macro Interest Rate": 4
    }
    
    for out_name, exp in raw_explanations.items():
        out_idx = out_map[out_name]
        out_meta = OUTPUT_NAMES[out_idx]
        
        attributions = exp["attributions"]
        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_3 = sorted_attrs[:3]
        
        top_drivers = []
        for feat_name, grad in top_3:
            feat_idx = feature_name_to_idx[feat_name]
            feat_meta = FEATURE_NAMES[feat_idx]
            direction = "positive" if grad > 0 else "negative"
            top_drivers.append({
                "name": feat_meta["name"],
                "direction": direction,
                "sensitivity": float(grad) * 100,
            })
            
        primary_grad = exp["primary_driver_gradient"]
        confidence = "high" if abs(primary_grad) > 0.5 else ("medium" if abs(primary_grad) > 0.1 else "low")
        
        action_word = out_meta["positive"] if primary_grad > 0 else out_meta["negative"]
        magnitude = abs(primary_grad) * 10.0
        
        primary_driver = top_drivers[0]
        primary_feat_idx = feature_name_to_idx[exp["primary_driver"]]
        
        rec_text = templates["recommendation"].format(action=action_word, magnitude=magnitude, unit=out_meta["unit"])
        driver_text = templates["primary_driver"].format(
            feature_name=primary_driver["name"],
            description=FEATURE_NAMES[primary_feat_idx]["description"],
            direction=primary_driver["direction"],
            sensitivity=primary_driver["sensitivity"]
        )
        risk_text = templates["risk_if_ignored"].format(risk_scenario="suboptimal performance or supply chain imbalances")
        narrative = f"{rec_text} {driver_text} {risk_text}"
        
        executive_explanations["decisions"][out_name] = {
            "recommendation": f"{action_word.capitalize()} by {magnitude:.1f}{out_meta['unit']}",
            "confidence": confidence,
            "top_drivers": top_drivers,
            "risk_if_ignored": risk_text,
            "narrative": narrative
        }
        
    return executive_explanations
