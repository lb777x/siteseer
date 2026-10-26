"""Parse the `site_labels` column: groups of 1-based residue numbers.

EasIFA's original rows (label_source 'goliath_original' or missing) follow EasIFA's range convention, [b, e] = residues
b..e; groups of other sources (M-CSA, M3Site, BioLiP) list the residues of one site.
"""
import ast
from typing import Optional

RANGE_LABEL_SOURCES = frozenset({"goliath_original", None})


def parse_site_labels(raw: str, label_source: Optional[str] = "goliath_original") -> Optional[set]:
    """'[[68], [71], [128]]' -> {68, 71, 128}; None if the string cannot be parsed."""
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return None
    if isinstance(label_source, float):  # NaN from a pandas row
        label_source = None
    as_range = label_source in RANGE_LABEL_SOURCES
    flat = set()
    for group in parsed:
        if not isinstance(group, (list, tuple)):
            flat.add(int(group))
        elif not as_range:
            flat.update(int(x) for x in group)
        elif len(group) == 1:
            flat.add(int(group[0]))
        elif len(group) == 2:
            b, e = int(group[0]), int(group[1])
            if e < b:
                raise ValueError(f"range label with end < start: {group!r} in {raw!r}")
            flat.update(range(b, e + 1))
        else:
            raise ValueError(f"range-format label group with {len(group)} elements: {group!r} in {raw!r}")
    return flat


def row_label_source(row) -> Optional[str]:
    """`label_source` of a CSV row, None when the column is absent or empty."""
    src = row.get("label_source") if hasattr(row, "get") else None
    return src if isinstance(src, str) and src else None
