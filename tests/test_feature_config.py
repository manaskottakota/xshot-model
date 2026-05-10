"""Feature lists, omission, and predefined ablation group wiring."""

from __future__ import annotations

import pytest

from xshot.datasets import FEATURE_ABLATION_GROUPS, feature_columns, resolve_ablation_union


def test_omit_withholds_requested_columns_from_X():
    kept_num, kept_cat = feature_columns(
        "core",
        omit_features={"prior_cum_fg_pct", "shot_zone_basic"},
    )
    assert "prior_cum_fg_pct" not in kept_num
    assert "shot_zone_basic" not in kept_cat


def test_ablation_groups_are_nonempty_sets():
    for name, feats in FEATURE_ABLATION_GROUPS.items():
        assert name
        assert isinstance(feats, frozenset)
        assert len(feats) > 3
        assert all(isinstance(x, str) for x in feats)


def test_resolve_ablation_union_unknown_group_raises():
    with pytest.raises(KeyError):
        resolve_ablation_union(["__not_a_registered_group__"])


def test_resolve_ablation_union_known_merges():
    u = resolve_ablation_union(["spatial_context"])
    assert "loc_x_ft" in u
