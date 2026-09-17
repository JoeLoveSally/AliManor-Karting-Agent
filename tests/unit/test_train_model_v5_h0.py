from __future__ import annotations

import pytest

from scripts.train_model_v5_h0 import EXPECTED_HORIZONS_MS, validate_v5_config


def make_config() -> dict[str, object]:
    return {
        "dataset": {
            "prediction_horizons_ms": list(EXPECTED_HORIZONS_MS),
            "control_horizon_ms": 0,
            "counterfactual_train_states": True,
        }
    }


def test_validate_v5_config_accepts_dense_h0_setup() -> None:
    assert validate_v5_config(make_config()) == EXPECTED_HORIZONS_MS


def test_validate_v5_config_rejects_nonzero_control_horizon() -> None:
    config = make_config()
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    dataset["control_horizon_ms"] = 50

    with pytest.raises(ValueError, match="control_horizon_ms=0"):
        validate_v5_config(config)


def test_validate_v5_config_rejects_missing_dense_horizon() -> None:
    config = make_config()
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    dataset["prediction_horizons_ms"] = [0, 100, 200, 300]

    with pytest.raises(ValueError, match="requires horizons"):
        validate_v5_config(config)


def test_validate_v5_config_requires_counterfactual_training() -> None:
    config = make_config()
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    dataset["counterfactual_train_states"] = False

    with pytest.raises(ValueError, match="counterfactual_train_states=true"):
        validate_v5_config(config)
