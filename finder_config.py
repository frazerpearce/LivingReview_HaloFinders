"""Shared halo-finder names and plotting configuration.

Add new finders here after their converter reader is available. Dictionary
order controls plotting order. ``REFERENCE_FINDER`` selects the catalogue used
to seed cross-finder object matching.
"""

from __future__ import annotations


REFERENCE_FINDER = "ahf"

FINDERS = {
    "ahf": {
        "label": "AHF",
        "color": "C0",
        "linestyle": "-",
        "derived": "LRdata/cosmological_ahf_128_derived.txt",
        "catalogue": "LRdata/cosmological_ahf_128.hdf5",
        "king_catalogue": "LRdata/unit_test_king_ahf.hdf5",
    },
    "subfind": {
        "label": "SUBFIND",
        "color": "C1",
        "linestyle": "--",
        "derived": "LRdata/cosmological_subfind_128_derived.txt",
        "catalogue": "LRdata/cosmological_subfind_128.hdf5",
        "king_catalogue": "LRdata/unit_test_king_subfind.hdf5",
    },
    "rockstar": {
        "label": "ROCKSTAR",
        "color": "C2",
        "linestyle": ":",
        "derived": "LRdata/cosmological_rockstar_128_derived.txt",
        "catalogue": "LRdata/cosmological_rockstar_128.hdf5",
        "king_catalogue": "LRdata/unit_test_king_rockstar.hdf5",
    },
}

if REFERENCE_FINDER not in FINDERS:
    raise ValueError(
        f"REFERENCE_FINDER {REFERENCE_FINDER!r} is absent from FINDERS"
    )


def finder_keys() -> tuple[str, ...]:
    return tuple(FINDERS)


def finder_style(key: str) -> tuple[str, str, str]:
    try:
        config = FINDERS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown finder {key!r}") from exc
    return config["label"], config["color"], config["linestyle"]
