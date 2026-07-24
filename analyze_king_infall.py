#!/usr/bin/env python3
"""Analyze radial recovery in the common-format King-infall catalogues."""

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

from finder_config import FINDERS, finder_style
from plot_config import apply_plot_style


apply_plot_style(plt)


@dataclass
class TruthObject:
    index: int
    label: str
    ratio: float
    radial_fraction: float
    centre: np.ndarray
    radius: float
    particle_ids: np.ndarray
    particle_radii: np.ndarray


@dataclass
class Catalogue:
    key: str
    label: str
    color: str
    halo_ids: np.ndarray
    centres: np.ndarray
    radii: np.ndarray
    memberships: list[np.ndarray]
    path: Path


@dataclass
class Match:
    truth_index: int
    halo_index: int
    overlap: int
    precision: float
    recall: float
    jaccard: float
    centre_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("Simulation/data/king_infall.hdf5"),
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        action="append",
        help="Common HDF5 catalogue; repeat for multiple finders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_king_infall"),
    )
    parser.add_argument(
        "--region-half-width-kpc",
        type=float,
        default=2200.0,
    )
    parser.add_argument(
        "--profile-points",
        type=int,
        default=80,
    )
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def periodic_delta(
    points: np.ndarray, centre: np.ndarray, box_size: float
) -> np.ndarray:
    return (points - centre + 0.5 * box_size) % box_size - 0.5 * box_size


def load_truth(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, list[TruthObject]]:
    with h5py.File(path, "r") as handle:
        positions = np.asarray(
            handle["PartType1/Coordinates"], dtype=np.float64
        )
        particle_ids = np.asarray(
            handle["PartType1/ParticleIDs"], dtype=np.uint64
        )
        box_size = float(handle["Header"].attrs["BoxSize"])
        config = handle["Config"]
        labels = decode_strings(np.asarray(config["ObjectLabel"]))
        ratios = np.asarray(config["MassRatio"], dtype=np.float64)
        fractions = np.asarray(
            config["HostRadialFraction"], dtype=np.float64
        )
        centres = np.asarray(config["CentreKpc"], dtype=np.float64)
        starts = np.asarray(config["ParticleStart"], dtype=np.int64)
        counts = np.asarray(config["ParticleCount"], dtype=np.int64)
        radii = np.asarray(config["R200Kpc"], dtype=np.float64)

    truth: list[TruthObject] = []
    for index, (
        label, ratio, fraction, centre, start, count, radius
    ) in enumerate(
        zip(labels, ratios, fractions, centres, starts, counts, radii)
    ):
        sl = slice(int(start), int(start + count))
        truth.append(
            TruthObject(
                index=index,
                label=label,
                ratio=float(ratio),
                radial_fraction=float(fraction),
                centre=centre,
                radius=float(radius),
                particle_ids=particle_ids[sl].copy(),
                particle_radii=np.linalg.norm(
                    periodic_delta(positions[sl], centre, box_size), axis=1
                ),
            )
        )
    return positions, particle_ids, box_size, truth


def default_catalogues() -> list[Path]:
    paths = [
        Path(f"LRdata/king_infall_{key}.hdf5")
        for key in FINDERS
    ]
    return [path for path in paths if path.exists()]


def load_catalogue(path: Path) -> Catalogue:
    with h5py.File(path, "r") as handle:
        key = handle["Header"].attrs["finder"]
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        key = str(key).lower()
        halo = handle["Haloes"]
        halo_ids = np.asarray(halo["haloid"], dtype=np.int64)
        centres = np.asarray(halo["centre"], dtype=np.float64)
        radii = np.asarray(halo["catalogue_radius"], dtype=np.float64)
        offsets = np.asarray(halo["offset"], dtype=np.int64)
        particle_ids = np.asarray(halo["particle_id"], dtype=np.uint64)
    label, color, _ = finder_style(key)
    memberships = [
        np.unique(particle_ids[offsets[i] : offsets[i + 1]])
        for i in range(halo_ids.size)
    ]
    return Catalogue(
        key, label, color, halo_ids, centres, radii, memberships, path
    )


def match_catalogue(
    truth: list[TruthObject],
    catalogue: Catalogue,
    box_size: float,
) -> dict[int, Match]:
    ntruth = len(truth)
    nhalo = catalogue.halo_ids.size
    overlap = np.zeros((ntruth, nhalo), dtype=np.int64)
    score = np.zeros((ntruth, nhalo), dtype=np.float64)
    for i, obj in enumerate(truth):
        truth_ids = np.unique(obj.particle_ids)
        for j, member_ids in enumerate(catalogue.memberships):
            shared = np.intersect1d(
                truth_ids, member_ids, assume_unique=True
            ).size
            overlap[i, j] = shared
            union = truth_ids.size + member_ids.size - shared
            if union:
                score[i, j] = shared / union

    rows, columns = linear_sum_assignment(-score)
    matches: dict[int, Match] = {}
    for i, j in zip(rows, columns):
        shared = int(overlap[i, j])
        if shared == 0:
            continue
        ntruth_ids = truth[i].particle_ids.size
        nfinder_ids = catalogue.memberships[j].size
        union = ntruth_ids + nfinder_ids - shared
        centre_error = float(
            np.linalg.norm(
                periodic_delta(
                    catalogue.centres[[j]], truth[i].centre, box_size
                )[0]
            )
        )
        matches[i] = Match(
            truth_index=i,
            halo_index=int(j),
            overlap=shared,
            precision=shared / nfinder_ids if nfinder_ids else np.nan,
            recall=shared / ntruth_ids if ntruth_ids else np.nan,
            jaccard=shared / union if union else np.nan,
            centre_error=centre_error,
        )
    return matches


def plot_apertures(
    path: Path,
    positions: np.ndarray,
    box_size: float,
    truth: list[TruthObject],
    catalogues: list[Catalogue],
    matches_by_finder: dict[str, dict[int, Match]],
    half_width: float,
) -> None:
    host = truth[0]
    delta = periodic_delta(positions, host.centre, box_size)
    selected = (
        (np.abs(delta[:, 0]) <= half_width)
        & (np.abs(delta[:, 1]) <= half_width)
        & (np.abs(delta[:, 2]) <= half_width)
    )
    sample = np.flatnonzero(selected)
    if sample.size > 120_000:
        rng = np.random.default_rng(8927)
        sample = rng.choice(sample, 120_000, replace=False)

    fig, axes = plt.subplots(
        1, len(catalogues),
        figsize=(5.2 * len(catalogues), 5.2),
        sharex=True, sharey=True,
        squeeze=False,
    )
    for column, (axis, catalogue) in enumerate(zip(axes[0], catalogues)):
        axis.scatter(
            delta[sample, 0] / 1000.0,
            delta[sample, 1] / 1000.0,
            s=0.2, color="0.65", alpha=0.25, rasterized=True,
        )
        matches = matches_by_finder[catalogue.key]
        for obj in truth:
            truth_xy = periodic_delta(
                obj.centre[None, :], host.centre, box_size
            )[0, :2]
            axis.add_patch(
                Circle(
                    truth_xy / 1000.0,
                    obj.radius / 1000.0,
                    fill=False, color="0.15", linestyle="--",
                    linewidth=1.0,
                )
            )
            axis.plot(
                truth_xy[0] / 1000.0,
                truth_xy[1] / 1000.0,
                marker="+", color="0.15", markersize=5,
            )
            match = matches.get(obj.index)
            if match is None:
                axis.plot(
                    truth_xy[0] / 1000.0,
                    truth_xy[1] / 1000.0,
                    marker="x", color="C3", markersize=7,
                    markeredgewidth=1.5,
                )
                continue
            recovered_xy = periodic_delta(
                catalogue.centres[[match.halo_index]],
                host.centre,
                box_size,
            )[0, :2]
            axis.add_patch(
                Circle(
                    recovered_xy / 1000.0,
                    catalogue.radii[match.halo_index] / 1000.0,
                    fill=False, color=catalogue.color,
                    linewidth=1.3,
                )
            )
        axis.text(
            0.97, 0.96, catalogue.label,
            transform=axis.transAxes, ha="right", va="top",
        )
        axis.set_aspect("equal")
        axis.set(
            xlim=(-2.0, 1.0),
            ylim=(-1.0, 2.0),
            xlabel=r"$\Delta x\ [h^{-1}{\rm Mpc}]$",
        )
        if column < len(catalogues) - 1:
            axis.get_xticklabels()[-1].set_visible(False)
    axes[0, 0].set_ylabel(r"$\Delta y\ [h^{-1}{\rm Mpc}]$")
    handles = [
        Line2D([], [], color="0.15", linestyle="--", label="Inserted $R_{200}$"),
        Line2D([], [], color="0.3", linestyle="-", label="Recovered radius"),
        Line2D([], [], marker="x", linestyle="none", color="C3", label="Unmatched"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", frameon=False)
    fig.subplots_adjust(wspace=0.0)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def recovery_curve(
    embedded: TruthObject,
    isolated: TruthObject,
    embedded_members: np.ndarray | None,
    isolated_members: np.ndarray | None,
    radial_grid: np.ndarray,
) -> np.ndarray:
    if isolated_members is None:
        return np.full(radial_grid.size, np.nan)
    isolated_kept = np.isin(
        isolated.particle_ids, isolated_members, assume_unique=False
    )
    if embedded_members is None:
        embedded_kept = np.zeros(embedded.particle_ids.size, dtype=bool)
    else:
        embedded_kept = np.isin(
            embedded.particle_ids, embedded_members, assume_unique=False
        )
    if isolated_kept.size != embedded_kept.size:
        raise ValueError(
            f"Template size mismatch for {embedded.label} and {isolated.label}"
        )
    scaled_radius = embedded.particle_radii / embedded.radius
    curve = np.full(radial_grid.size, np.nan)
    paired_kept = isolated_kept & embedded_kept
    for i, radius in enumerate(radial_grid):
        inside = scaled_radius <= radius
        denominator = np.count_nonzero(inside & isolated_kept)
        if denominator:
            curve[i] = np.count_nonzero(inside & paired_kept) / denominator
    return curve


def plot_recovery_profiles(
    path: Path,
    truth: list[TruthObject],
    catalogues: list[Catalogue],
    matches_by_finder: dict[str, dict[int, Match]],
    npoints: int,
) -> None:
    ratios = (0.1, 0.01)
    radial_grid = np.geomspace(0.03, 4.0, npoints)
    fractions = sorted(
        {
            obj.radial_fraction
            for obj in truth
            if np.isfinite(obj.radial_fraction)
            and obj.radial_fraction > 0.0
        }
    )
    colours = plt.cm.viridis(np.linspace(0.12, 0.9, len(fractions)))
    colour_by_fraction = dict(zip(fractions, colours))

    fig, axes = plt.subplots(
        len(ratios), len(catalogues),
        figsize=(5.0 * len(catalogues), 4.0 * len(ratios)),
        sharex=True, sharey=True,
        squeeze=False,
    )
    for row, ratio in enumerate(ratios):
        isolated = next(
            obj for obj in truth
            if obj.label == f"control_{ratio:g}"
        )
        embedded_objects = sorted(
            (
                obj for obj in truth
                if obj.ratio == ratio
                and np.isfinite(obj.radial_fraction)
                and obj.radial_fraction > 0.0
            ),
            key=lambda obj: obj.radial_fraction,
        )
        for column, catalogue in enumerate(catalogues):
            axis = axes[row, column]
            matches = matches_by_finder[catalogue.key]
            isolated_match = matches.get(isolated.index)
            isolated_members = (
                None if isolated_match is None
                else catalogue.memberships[isolated_match.halo_index]
            )
            for obj in embedded_objects:
                match = matches.get(obj.index)
                embedded_members = (
                    None if match is None
                    else catalogue.memberships[match.halo_index]
                )
                curve = recovery_curve(
                    obj,
                    isolated,
                    embedded_members,
                    isolated_members,
                    radial_grid,
                )
                axis.plot(
                    radial_grid,
                    curve,
                    color=colour_by_fraction[obj.radial_fraction],
                    linewidth=1.8,
                    label=rf"${obj.radial_fraction:g}R_{{200,\rm host}}$",
                )
            axis.axhline(1.0, color="0.5", linewidth=0.8, linestyle=":")
            axis.set_xscale("log")
            axis.set_ylim(-0.03, 1.05)
            axis.text(
                0.97, 0.08,
                f"{catalogue.label}\n1:{int(round(1.0 / ratio))}",
                transform=axis.transAxes,
                ha="right", va="bottom",
            )
            if row == len(ratios) - 1:
                axis.set_xlabel(r"$r/R_{200,\rm satellite}$")
            if column == 0:
                axis.set_ylabel(
                    "Paired recovery relative\nto isolated control"
                )
    axes[0, -1].legend(loc="lower left", frameon=False)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    truth: list[TruthObject],
    catalogues: list[Catalogue],
    matches_by_finder: dict[str, dict[int, Match]],
) -> None:
    lines = [
        "King-infall particle-ID matching summary",
        (
            "finder truth_label truth_ratio host_radial_fraction recovered_halo "
            "overlap precision recall jaccard centre_error_kpc_h"
        ),
    ]
    for catalogue in catalogues:
        matches = matches_by_finder[catalogue.key]
        for obj in truth:
            match = matches.get(obj.index)
            if match is None:
                lines.append(
                    f"{catalogue.key} {obj.label} {obj.ratio:.8g} "
                    f"{obj.radial_fraction:.8g} -1 0 nan 0 nan nan"
                )
                continue
            lines.append(
                f"{catalogue.key} {obj.label} {obj.ratio:.8g} "
                f"{obj.radial_fraction:.8g} "
                f"{int(catalogue.halo_ids[match.halo_index])} "
                f"{match.overlap} {match.precision:.8g} "
                f"{match.recall:.8g} {match.jaccard:.8g} "
                f"{match.centre_error:.8g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    catalogue_paths = (
        args.catalogue if args.catalogue is not None
        else default_catalogues()
    )
    if not catalogue_paths:
        raise ValueError("No King-infall common catalogues found")
    if args.profile_points < 2:
        raise ValueError("--profile-points must be at least 2")

    positions, _, box_size, truth = load_truth(args.snapshot)
    catalogues = [load_catalogue(path) for path in catalogue_paths]
    matches_by_finder = {
        catalogue.key: match_catalogue(truth, catalogue, box_size)
        for catalogue in catalogues
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aperture_path = args.output_dir / "king_infall_xy_recovery.png"
    profile_path = (
        args.output_dir / "king_infall_particle_recovery_profiles.png"
    )
    summary_path = args.output_dir / "king_infall_analysis_summary.txt"

    plot_apertures(
        aperture_path,
        positions,
        box_size,
        truth,
        catalogues,
        matches_by_finder,
        args.region_half_width_kpc,
    )
    plot_recovery_profiles(
        profile_path,
        truth,
        catalogues,
        matches_by_finder,
        args.profile_points,
    )
    write_summary(
        summary_path, truth, catalogues, matches_by_finder
    )

    for catalogue in catalogues:
        print(
            f"{catalogue.label}: matched "
            f"{len(matches_by_finder[catalogue.key])}/{len(truth)} objects"
        )
    print(f"Wrote {aperture_path}")
    print(f"Wrote {profile_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
