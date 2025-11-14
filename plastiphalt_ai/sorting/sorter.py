
from typing import List, Dict, Any

PLASTIC_DB = {
    "PET": {"melting_point": 250, "suitability": {"low": 0.6, "medium": 0.75, "high": 0.8}},
    "HDPE": {"melting_point": 130, "suitability": {"low": 0.8, "medium": 0.9, "high": 0.95}},
    "LDPE": {"melting_point": 110, "suitability": {"low": 0.9, "medium": 0.8, "high": 0.6}},
    "PP":  {"melting_point": 165, "suitability": {"low": 0.7, "medium": 0.9, "high": 0.95}},
    "PS":  {"melting_point": 240, "suitability": {"low": 0.5, "medium": 0.6, "high": 0.7}},
    "PVC": {"melting_point": 85,  "suitability": {"low": 0.2, "medium": 0.1, "high": 0.0}},
    "OTHER": {"melting_point": 0, "suitability": {"low": 0.2, "medium": 0.2, "high": 0.2}},
}

def sort_plastics(detections: List[Dict[str, Any]], severity: str = "medium") -> List[Dict[str, Any]]:
    """
    Rank detected plastic items by suitability × confidence for the given road severity.
    Each detection must be: {"id": str, "label": str, "confidence": float}
    """
    severity = severity.lower()
    def score(item):
        info = PLASTIC_DB.get(item.get("label","OTHER"), PLASTIC_DB["OTHER"])
        return info["suitability"].get(severity, 0.5) * float(item.get("confidence", 1.0))
    return sorted(detections, key=score, reverse=True)

