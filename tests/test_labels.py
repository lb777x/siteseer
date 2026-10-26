import math

import pandas as pd
import pytest

from siteseer.data.labels import parse_site_labels, row_label_source
from siteseer.data.site_types import site_type_sets, type_targets


def test_goliath_range_is_inclusive():
    assert parse_site_labels("[[10, 17]]", "goliath_original") == set(range(10, 18))
    assert parse_site_labels("[[68], [71, 73], [128]]", "goliath_original") == {68, 71, 72, 73, 128}


def test_missing_source_uses_range_convention():
    assert parse_site_labels("[[10, 12]]", None) == {10, 11, 12}
    assert parse_site_labels("[[10, 12]]", float("nan")) == {10, 11, 12}


def test_range_errors():
    with pytest.raises(ValueError):
        parse_site_labels("[[10, 12, 15]]", "goliath_original")
    with pytest.raises(ValueError):
        parse_site_labels("[[17, 10]]", "goliath_original")


def test_other_sources_are_residue_lists():
    assert parse_site_labels("[[81, 119, 163, 165]]", "biolip_new") == {81, 119, 163, 165}
    for src in ("m3site", "m3site_new", "mcsa_exact", "mcsa_blast_homology"):
        assert parse_site_labels("[[7], [42]]", src) == {7, 42}


def test_malformed_and_default():
    assert parse_site_labels("not a list", "goliath_original") is None
    assert parse_site_labels("[[1, 3]]") == {1, 2, 3}


def test_row_label_source():
    assert row_label_source(pd.Series({"label_source": "biolip_new"})) == "biolip_new"
    assert row_label_source(pd.Series({"label_source": math.nan})) is None
    assert row_label_source(pd.Series({"site_labels": "[[1]]"})) is None


def test_site_types_goliath():
    sets = site_type_sets("[[2], [5, 6], [9]]", "[1, 0, 2]", "goliath_original")
    assert sets["catalytic"] == {2} and sets["binding"] == {5, 6} and sets["ignore"] == {9}
    targets, mask = type_targets(10, sets)
    assert targets[1, 0] == 1 and targets[4, 1] == 1 and mask[8].sum() == 0 and mask[0].sum() == 2


def test_site_types_mcsa_masks_binding():
    targets, mask = type_targets(5, site_type_sets("[[3]]", None, "mcsa"))
    assert mask[:, 0].sum() == 5 and mask[:, 1].sum() == 0 and targets[2, 0] == 1
