from __future__ import annotations

from rna_stability_elements.encode import _dataset_time_h


def test_dataset_time_defaults_to_zero_for_control():
    assert _dataset_time_h({"replicates": [{"library": {"biosample": {}}}]}) == 0.0


def test_dataset_time_from_embedded_replicate():
    dataset = {
        "replicates": [
            {"library": {"biosample": {"pulse_chase_time": 2, "pulse_chase_time_units": "hour"}}}
        ]
    }
    assert _dataset_time_h(dataset) == 2.0
