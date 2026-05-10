"""Shared helpers powering the viz inference API."""

from __future__ import annotations

from xshot.viz.inference_row import archetype_action_type, infer_shot_zone_basic, inference_X_from_payload


def test_archetype_action_mapping_fallback_jumper():
    assert "jump" in archetype_action_type("junk-name").lower()


def test_corner_three_classification():
    z = infer_shot_zone_basic(-22.0, 8.0, True)
    assert "Corner 3" in z


def test_inference_row_core_assembly_smoke():
    payload = {
        "loc_x_ft": -14.5,
        "loc_y_ft": 12.0,
        "is_three": False,
        "shot_archetype": "floater",
        "score_diff": 3,
        "period": 2,
        "minutes_remaining": 8.5,
        "seconds_remaining": 15.2,
        "player_profile": "league_average",
    }
    x = inference_X_from_payload(payload, features="core")
    assert len(x) == 1
    assert "prior_cum_fg_pct" in x.columns


def test_inference_row_advanced_tracking_columns():
    payload = {
        "loc_x_ft": -12.25,
        "loc_y_ft": 9.1,
        "is_three": False,
        "shot_archetype": "layup",
        "defender_distance_ft": 3.1,
        "defender_contest_azimuth_deg": -12.5,
        "player_profile": "combo_guard",
    }
    x = inference_X_from_payload(payload, features="core+advanced")
    assert len(x) == 1
    assert "defender_distance_ft" in x.columns
    assert float(x["defender_distance_ft"].iloc[0]) > 3.05


def test_defender_sweep_changes_advanced_feature_vector():
    base = {
        "loc_x_ft": -14.0,
        "loc_y_ft": 16.0,
        "is_three": True,
        "shot_archetype": "jumper",
        "player_profile": "league_average",
    }
    a = inference_X_from_payload({**base, "defender_distance_ft": 1.2}, features="core+advanced")
    b = inference_X_from_payload({**base, "defender_distance_ft": 10.5}, features="core+advanced")
    assert float(a["defender_distance_ft"].iloc[0]) < float(b["defender_distance_ft"].iloc[0])
    assert float(a["def_contest_open_bucket"].iloc[0]) != float(b["def_contest_open_bucket"].iloc[0])


def test_smoke_advanced_model_sensitivity():
    import tempfile
    from pathlib import Path

    import joblib
    import numpy as np

    from xshot.smoke_advanced_train import fit_smoke_advanced_classifier
    from xshot.viz.inference_row import inference_X_from_payload

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lr.joblib"
        fit_smoke_advanced_classifier(p, n_rows=320)
        clf = joblib.load(p)
        t = inference_X_from_payload(
            {
                "loc_x_ft": -16.0,
                "loc_y_ft": 18.0,
                "is_three": True,
                "defender_distance_ft": 1.1,
                "player_profile": "league_average",
            },
            features="core+advanced",
        )
        o = inference_X_from_payload(
            {
                "loc_x_ft": -16.0,
                "loc_y_ft": 18.0,
                "is_three": True,
                "defender_distance_ft": 12.0,
                "player_profile": "league_average",
            },
            features="core+advanced",
        )
        pre = clf.named_steps["pre"]
        xt = pre.transform(t)
        xo = pre.transform(o)
        # Preprocessed matrices must differ — proves advanced columns (not only zone OHE) move.
        assert np.max(np.abs(xt - xo)) > 0.05
