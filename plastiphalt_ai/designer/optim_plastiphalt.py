from typing import Dict, Any
import math
import json

# Import the LLM integration
try:
    from plastiphalt_ai.llm_integration import mix_with_llm
except ImportError:
    mix_with_llm = None  # fallback if not available


MIX_TEMP_C = (160, 170)
PLASTIC_PERCENT_RANGE = (0.06, 0.10)

SEVERITY_WEIGHTS = {
    "low": {"LDPE": 0.5, "HDPE": 0.7, "PP": 0.5, "PET": 0.4, "PS": 0.3},
    "medium": {"LDPE": 0.6, "HDPE": 0.9, "PP": 0.85, "PET": 0.5, "PS": 0.4},
    "high": {"LDPE": 0.4, "HDPE": 0.95, "PP": 0.95, "PET": 0.6, "PS": 0.5},
}


def suggest_mix(
    severity: str,
    traffic_adt: int,
    climate_temp_c: float,
    available_kg: Dict[str, float],
    batch_bitumen_kg: float = 1000.0,
    aggregate_to_bitumen_ratio: float = 10.0
) -> Dict[str, Any]:
    severity = severity.lower()
    weights = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["medium"]).copy()

    # Normalize available plastics (uppercase) and remove unsuitable ones
    available_kg = {k.upper(): float(v) for k, v in available_kg.items()}
    for k in list(available_kg.keys()):
        if k in ["PVC", "OTHER"]:
            available_kg.pop(k)

    # Plastic requirement range
    p_min = PLASTIC_PERCENT_RANGE[0] * batch_bitumen_kg
    p_max = PLASTIC_PERCENT_RANGE[1] * batch_bitumen_kg
    target_p = min(max((p_min + p_max) / 2, 0.0), sum(available_kg.values()))

    # Weight normalization
    w = {}
    tot = 0.0
    for k, v in weights.items():
        if k in available_kg and available_kg[k] > 0:
            w[k] = v
            tot += v
    if tot == 0:
        return {"error": "No suitable plastics available"}
    for k in list(w.keys()):
        w[k] = w[k] / tot

    # Allocation
    alloc = {k: min(available_kg.get(k, 0.0), w[k] * target_p) for k in w}
    used_p = sum(alloc.values())

    # Scale up if below minimum requirement
    if used_p < p_min and p_min > 0:
        scale = min(1.0, sum(available_kg.values()) / p_min)
        for k in alloc:
            alloc[k] *= scale
        used_p = sum(alloc.values())

    # Calculate aggregates
    aggregate_kg = aggregate_to_bitumen_ratio * batch_bitumen_kg

    # Durability score
    temp_penalty = max(0.0, abs((climate_temp_c - 30) / 30))
    traffic_factor = min(1.0, math.log1p(traffic_adt) / 10.0)
    severity_factor = {"low": 0.7, "medium": 0.85, "high": 1.0}[severity]
    comp_index = 0.0
    if target_p > 0:
        for k, kg in alloc.items():
            comp_index += kg / target_p * weights.get(k, 0.5)
    durability_score = max(
        0.0,
        min(1.0, 0.6 * comp_index + 0.3 * traffic_factor + 0.1 * severity_factor - 0.15 * temp_penalty)
    )

    return {
        "batch_bitumen_kg": batch_bitumen_kg,
        "aggregate_kg": aggregate_kg,
        "plastic_allocations_kg": {k: round(v, 2) for k, v in alloc.items()},
        "total_plastic_kg": round(used_p, 2),
        "plastic_pct_of_bitumen": round(used_p / max(batch_bitumen_kg, 1e-9), 4),
        "mix_temperature_c": MIX_TEMP_C,
        "expected_durability_score": round(durability_score, 3),
        "notes": [
            "Operate mixer between 160–170°C; plastics coat aggregate before bitumen addition.",
            "Target 6–10% plastic of bitumen weight; avoid PVC."
        ]
    }


if __name__ == "__main__":
    # Example run
    result = suggest_mix(
        severity="medium",
        traffic_adt=15000,
        climate_temp_c=32,
        available_kg={"HDPE": 300, "LDPE": 200, "PP": 100},
        batch_bitumen_kg=1000
    )

    print("=== Raw Mix Result ===")
    print(json.dumps(result, indent=2))

    # Safely try LLM explanation
    if mix_with_llm:
        try:
            print("\n=== LLM Explanation (Ollama) ===")
            explanation = mix_with_llm(result, model="llama3")
            print(explanation)
        except Exception as e:
            print(f"\nLLM integration failed: {e}")
            print("Skipping LLM explanation. Ensure Ollama CLI and model are installed if needed.")
    else:
        print("\nLLM integration not available. Skipping LLM explanation.")

