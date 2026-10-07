"""Core DGA data structures and feature engineering.

The implementation intentionally keeps the raw gas representation explicit so
engineering assumptions remain reviewable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

GASES = ("H2", "CH4", "C2H6", "C2H4", "C2H2")


@dataclass(frozen=True)
class DGASample:
    """Five-gas dissolved-gas-analysis sample in ppm."""

    h2: float
    ch4: float
    c2h6: float
    c2h4: float
    c2h2: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "DGASample":
        missing = [gas for gas in GASES if gas not in values]
        if missing:
            raise KeyError(f"Missing DGA gases: {', '.join(missing)}")

        numbers = {gas: float(values[gas]) for gas in GASES}
        for gas, value in numbers.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{gas} must be a finite non-negative ppm value")

        return cls(
            h2=numbers["H2"],
            ch4=numbers["CH4"],
            c2h6=numbers["C2H6"],
            c2h4=numbers["C2H4"],
            c2h2=numbers["C2H2"],
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "H2": self.h2,
            "CH4": self.ch4,
            "C2H6": self.c2h6,
            "C2H4": self.c2h4,
            "C2H2": self.c2h2,
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio while preserving zero-denominator information."""
    return numerator / denominator if denominator > 0 else 0.0


def engineer_features(sample: DGASample) -> dict[str, float]:
    """Create transparent ML features from a five-gas DGA sample.

    Ratios are chemically meaningful descriptors commonly used in DGA
    interpretation. They are features for the ML model here, not a claim that
    this function itself implements an IEC/IEEE diagnostic rule.
    """
    gases = sample.as_dict()
    total = sum(gases.values())

    features = dict(gases)
    features.update(
        {
            "TOTAL_GAS": total,
            "RATIO_CH4_H2": _safe_ratio(sample.ch4, sample.h2),
            "RATIO_C2H2_C2H4": _safe_ratio(sample.c2h2, sample.c2h4),
            "RATIO_C2H4_C2H6": _safe_ratio(sample.c2h4, sample.c2h6),
        }
    )

    for gas, value in gases.items():
        features[f"FRACTION_{gas}"] = value / total if total > 0 else 0.0

    return features
