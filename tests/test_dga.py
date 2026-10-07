"""Tests for independent DGA data validation and feature engineering."""

import pytest

from src.transformer_pm.dga import DGASample, engineer_features


def test_sample_requires_all_five_gases():
    with pytest.raises(KeyError):
        DGASample.from_mapping({"H2": 10})


def test_negative_gas_value_is_rejected():
    values = {"H2": 10, "CH4": 5, "C2H6": 2, "C2H4": 1, "C2H2": -1}
    with pytest.raises(ValueError):
        DGASample.from_mapping(values)


def test_engineered_features_include_ratios_total_and_fractions():
    sample = DGASample.from_mapping(
        {"H2": 20, "CH4": 10, "C2H6": 5, "C2H4": 10, "C2H2": 5}
    )
    features = engineer_features(sample)

    assert features["TOTAL_GAS"] == 50
    assert features["RATIO_CH4_H2"] == 0.5
    assert features["RATIO_C2H2_C2H4"] == 0.5
    assert features["RATIO_C2H4_C2H6"] == 2.0
    assert sum(features[f"FRACTION_{gas}"] for gas in ("H2", "CH4", "C2H6", "C2H4", "C2H2")) == pytest.approx(1.0)


def test_zero_sample_has_safe_zero_fractions():
    sample = DGASample(0, 0, 0, 0, 0)
    features = engineer_features(sample)
    assert features["TOTAL_GAS"] == 0
    assert features["RATIO_CH4_H2"] == 0
    assert features["FRACTION_C2H2"] == 0
