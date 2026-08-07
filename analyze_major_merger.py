#!/usr/bin/env python3
"""Plot recovered halo centres and radii through the major-merger sequence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.optimize import linear_sum_assignment

from finder_config import finder_style
from plot_config import apply_plot_style


apply_plot_style(plt)

FINDER_ORDER = ("ahf", "rockstar", "subfind")
HALO_LINESTYLES = ("-", "--")


@dataclass
class RecoveredHalo:
    centre: np.ndarray
    mass: float
    radius: float
    members: np.ndarray


def parse_args(default_test_name: str = "major_merger") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-name",
        choices=("major_merger", "minor_merger"),
        default=default_test_name,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--catalogue-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--snapshot-count", type=int, default=11)
    return parser.parse_args()


def truth_memberships(
    snapshot: Path,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    with h5py.File(snapshot, "r") as handle:
        config = handle["Config"].attrs
        particle_ids = np.asarray(
            handle["PartType1/ParticleIDs"], dtype=np.uint64
        )
        starts = np.asarray(config["KingStartIndices"], dtype=np.int64)
        if "KingParticleCounts" in config:
            counts = np.asarray(
                config["KingParticleCounts"], dtype=np.int64
            )
        else:
            counts = np.full(
                starts.size,
                int(config["KingParticleCountEach"]),
                dtype=np.int64,
            )
        box_size = float(handle["Header"].attrs["BoxSize"])
    memberships = [
        np.sort(particle_ids[int(start) : int(start) + int(count)])
        for start, count in zip(starts[:2], counts[:2])
    ]
    box_centre = np.full(3, 0.5 * box_size)
    return memberships, box_centre, box_size


def load_catalogue(path: Path) -> list[RecoveredHalo]:
    with h5py.File(path, "r") as handle:
        halo = handle["Haloes"]
        centres = np.asarray(halo["centre"], dtype=np.float64)
        masses = np.asarray(
            halo["catalogue_mass"], dtype=np.float64
        )
        radii = np.asarray(
            halo["catalogue_radius"], dtype=np.float64
        )
        offsets = np.asarray(halo["offset"], dtype=np.int64)
        particle_ids = np.asarray(
            halo["particle_id"], dtype=np.uint64
        )
    return [
        RecoveredHalo(
            centre=centres[index],
            mass=float(masses[index]),
            radius=float(radii[index]),
            members=np.unique(
                particle_ids[offsets[index] : offsets[index + 1]]
            ),
        )
        for index in range(centres.shape[0])
    ]


def match_progenitors(
    truth: list[np.ndarray],
    recovered: list[RecoveredHalo],
) -> dict[int, RecoveredHalo]:
    """Assign distinct recovered haloes by maximum particle-ID overlap."""
    if not recovered:
        return {}
    overlap = np.zeros((len(truth), len(recovered)), dtype=np.int64)
    for truth_index, truth_ids in enumerate(truth):
        for halo_index, halo in enumerate(recovered):
            overlap[truth_index, halo_index] = np.intersect1d(
                truth_ids,
                halo.members,
                assume_unique=True,
            ).size
    truth_indices, halo_indices = linear_sum_assignment(-overlap)
    return {
        int(truth_index): recovered[int(halo_index)]
        for truth_index, halo_index in zip(
            truth_indices, halo_indices
        )
        if overlap[truth_index, halo_index] > 0
    }


def periodic_delta(
    point: np.ndarray,
    origin: np.ndarray,
    box_size: float,
) -> np.ndarray:
    return (point - origin + 0.5 * box_size) % box_size - 0.5 * box_size


def two_significant_figures(value: float) -> str:
    """Format a finite value with two visible significant figures."""
    text = f"{value:.2g}"
    mantissa, separator, exponent = text.partition("e")
    significant = mantissa.lstrip("-").replace(".", "").lstrip("0")
    missing = 2 - len(significant)
    if missing > 0:
        if "." not in mantissa:
            mantissa += "."
        mantissa += "0" * missing
    return mantissa + (separator + exponent if separator else "")


def plot_sequence(
    output: Path,
    catalogue_dir: Path,
    truth: list[np.ndarray],
    box_centre: np.ndarray,
    box_size: float,
    snapshot_count: int,
    halo_labels: tuple[str, str],
    test_name: str,
) -> None:
    fig, axes = plt.subplots(
        3,
        snapshot_count,
        figsize=(16.5, 8.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, finder in enumerate(FINDER_ORDER):
        label, colour, _ = finder_style(finder)
        matches_by_snapshot: list[dict[int, RecoveredHalo]] = []
        for snapshot_index in range(snapshot_count):
            catalogue_path = catalogue_dir / (
                f"{test_name}_{snapshot_index:03d}_{finder}.hdf5"
            )
            if not catalogue_path.exists():
                raise FileNotFoundError(catalogue_path)
            matches_by_snapshot.append(
                match_progenitors(
                    truth,
                    load_catalogue(catalogue_path),
                )
            )
        reference_masses = {
            progenitor_index: halo.mass
            for progenitor_index, halo
            in matches_by_snapshot[0].items()
        }

        for snapshot_index, matches in enumerate(matches_by_snapshot):
            axis = axes[row, snapshot_index]
            for progenitor_index, halo in matches.items():
                centre = periodic_delta(
                    halo.centre,
                    box_centre,
                    box_size,
                )
                axis.add_patch(
                    Circle(
                        centre[[1, 0]] / 1000.0,
                        halo.radius / 1000.0,
                        fill=False,
                        edgecolor=colour,
                        linestyle=HALO_LINESTYLES[progenitor_index],
                        linewidth=1.25,
                    )
                )
                reference_mass = reference_masses.get(progenitor_index)
                if reference_mass is not None and reference_mass > 0.0:
                    axis.text(
                        0.5,
                        0.96 if progenitor_index == 1 else 0.04,
                        two_significant_figures(
                            halo.mass / reference_mass
                        ),
                        transform=axis.transAxes,
                        ha="center",
                        va="top" if progenitor_index == 1 else "bottom",
                        color=colour,
                    )
            axis.plot(
                0.0,
                0.0,
                marker="+",
                color="0.15",
                markersize=5,
                markeredgewidth=1.0,
                zorder=5,
            )
            axis.set_xlim(-0.45, 0.45)
            axis.set_ylim(-1.0, 1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(axis="x", labelbottom=False)
            if row == 0:
                axis.text(
                    0.5,
                    1.03,
                    f"{snapshot_index:02d}",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                )
            if snapshot_index == 0:
                axis.set_ylabel(
                    label + "\n" + r"$\Delta x\ [h^{-1}{\rm Mpc}]$",
                    color=colour,
                )
    axes[0, 0].legend(
        handles=[
            Line2D(
                [], [], color="0.2", linestyle="-",
                label=halo_labels[0],
            ),
            Line2D(
                [], [], color="0.2", linestyle="--",
                label=halo_labels[1],
            ),
        ],
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0.0, 1.36),
        ncol=2,
    )
    fig.supxlabel(r"$\Delta y\ [h^{-1}{\rm Mpc}]$", y=0.02)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(default_test_name: str = "major_merger") -> None:
    args = parse_args(default_test_name)
    if args.snapshot_count < 1:
        raise ValueError("--snapshot-count must be positive")
    directory_name = (
        "Major_Merger"
        if args.test_name == "major_merger"
        else "Minor_Merger"
    )
    snapshot_dir = args.snapshot_dir or Path(
        f"Simulation/{directory_name}_ICs"
    )
    catalogue_dir = args.catalogue_dir or Path(
        f"LRdata/{directory_name}"
    )
    output_dir = args.output_dir or Path(f"output_{args.test_name}")
    truth, box_centre, box_size = truth_memberships(
        snapshot_dir / f"{args.test_name}_000.hdf5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{args.test_name}_recovered_halo_sequence.png"
    )
    plot_sequence(
        output,
        catalogue_dir,
        truth,
        box_centre,
        box_size,
        args.snapshot_count,
        (
            ("King halo 1", "King halo 2")
            if args.test_name == "major_merger"
            else ("Primary King halo", "1:100 King halo")
        ),
        args.test_name,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
