#!/usr/bin/env python3
"""Analyze King-sphere unit tests using common halo-membership catalogues.

Each ``--catalogue`` must be an HDF5 file produced by
``convert_halo_membership.py``. Multiple catalogues may be supplied to compare
any finders declared in ``finder_config.py`` in one run.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from finder_config import FINDERS
from plot_config import apply_plot_style


apply_plot_style(plt)


PARTICLE_MASS_UNIT_MSUN = 1.0e10
G_KPC_KMS2_PER_MSUN = 4.30091e-6
RHO_CRIT_MSUN_PER_KPC3 = 2.77536627e11 / 1.0e9
INNER_RADIUS_KPC = 10.0
FINDER_STYLES = {
    key: (config["label"], config["color"])
    for key, config in FINDERS.items()
}


@dataclass
class Snapshot:
    path: Path
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    ids: np.ndarray
    box_size: float
    id_order: np.ndarray
    sorted_ids: np.ndarray


@dataclass
class TruthHalo:
    index: int
    centre: np.ndarray
    translation: np.ndarray
    particle_ids: np.ndarray
    total_mass: float
    core_radius: float
    trunc_radius: float
    r200: float
    rvmax: float
    vmax: float


@dataclass
class FinderCatalogue:
    key: str
    label: str
    color: str
    ids: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    membership: dict[int, np.ndarray]
    source: str


@dataclass
class FinderMatch:
    finder: FinderCatalogue
    truth: TruthHalo
    obj_index: int
    centre_error: float
    ids_in_r200: np.ndarray
    truth_ids_in_r200: np.ndarray
    n_overlap: int
    precision: float
    recall: float
    jaccard: float
    r_profile: np.ndarray
    density_profile: np.ndarray
    vcirc_r: np.ndarray
    vcirc_profile: np.ndarray
    rvmax: float
    vmax: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare King-sphere truth with common halo catalogues."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("Simulation/data/unit_test_king_box.hdf5"),
        help="Input King unit-test HDF5 snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output_king"),
        help="Directory for PNG and summary outputs.",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        action="append",
        help=(
            "Common HDF5 catalogue; repeat once per finder. If omitted, use "
            "the existing king_catalogue paths in finder_config.py."
        ),
    )
    parser.add_argument(
        "--max-subfind-subhalos",
        type=int,
        default=10,
        help="Reject a SUBFIND catalogue containing more than this many subhalos.",
    )
    parser.add_argument(
        "--match-radius-factor",
        type=float,
        default=1.0,
        help="Match finder centres within this many true R200 radii.",
    )
    parser.add_argument("--n-density-bins", type=int, default=40, help="Number of radial density bins.")
    return parser.parse_args()


def load_snapshot(path: Path) -> Snapshot:
    with h5py.File(path, "r") as f:
        p1 = f["PartType1"]
        pos = p1["Coordinates"][:].astype(np.float64)
        vel = p1["Velocities"][:].astype(np.float64)
        mass = p1["Masses"][:].astype(np.float64) * PARTICLE_MASS_UNIT_MSUN
        ids = p1["ParticleIDs"][:].astype(np.int64)
        box_size = float(f["Header"].attrs["BoxSize"])

    order = np.argsort(ids)
    return Snapshot(
        path=path,
        pos=pos,
        vel=vel,
        mass=mass,
        ids=ids,
        box_size=box_size,
        id_order=order,
        sorted_ids=ids[order],
    )


def king_mass_shape(x: np.ndarray | float) -> np.ndarray | float:
    return np.arcsinh(x) - x / np.sqrt(1.0 + x * x)


def enclosed_king_mass(r: np.ndarray, total_mass: float, core_radius: float, trunc_radius: float) -> np.ndarray:
    r_clip = np.minimum(np.maximum(r, 0.0), trunc_radius)
    return total_mass * king_mass_shape(r_clip / core_radius) / king_mass_shape(trunc_radius / core_radius)


def king_density(r: np.ndarray, total_mass: float, core_radius: float, trunc_radius: float) -> np.ndarray:
    rho0 = total_mass / (4.0 * np.pi * core_radius**3 * king_mass_shape(trunc_radius / core_radius))
    rho = rho0 / (1.0 + (r / core_radius) ** 2) ** 1.5
    return np.where(r <= trunc_radius, rho, np.nan)


def r200_king(total_mass: float, core_radius: float, trunc_radius: float) -> float:
    grid = np.geomspace(max(core_radius * 1.0e-5, 1.0e-4), trunc_radius, 100_000)
    menc = enclosed_king_mass(grid, total_mass, core_radius, trunc_radius)
    scaled = menc / ((4.0 / 3.0) * np.pi * grid**3) / RHO_CRIT_MSUN_PER_KPC3
    above = scaled >= 200.0
    crossings = np.where(above[:-1] & (~above[1:]))[0]
    if crossings.size == 0:
        return float(grid[-1])
    i = int(crossings[0])
    x0, x1 = np.log10(grid[i]), np.log10(grid[i + 1])
    y0, y1 = np.log10(scaled[i]), np.log10(scaled[i + 1])
    xcross = x0 + (np.log10(200.0) - y0) * (x1 - x0) / (y1 - y0)
    return float(10.0**xcross)


def analytic_vcirc(r: np.ndarray, total_mass: float, core_radius: float, trunc_radius: float) -> np.ndarray:
    vcirc = np.zeros_like(r, dtype=np.float64)
    positive = r > 0.0
    menc = enclosed_king_mass(r[positive], total_mass, core_radius, trunc_radius)
    vcirc[positive] = np.sqrt(G_KPC_KMS2_PER_MSUN * menc / r[positive])
    return vcirc


def rvmax_from_vcirc_profile(
    r: np.ndarray,
    vcirc: np.ndarray,
    r_limit: Optional[float] = None,
) -> tuple[float, float, float, float]:
    if r.size == 0 or vcirc.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if r_limit is not None and np.isfinite(r_limit):
        use = np.flatnonzero(r < r_limit)
        if use.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        r_use = r[: int(use[-1]) + 1]
        vc_use = vcirc[: int(use[-1]) + 1]
    else:
        r_use = r
        vc_use = vcirc
    if r_use.size == 1:
        return float(r_use[0]), float(r_use[0]), float(r_use[0]), float(vc_use[0])

    start = min(5, r_use.size - 1)
    vc_work = vc_use[start:]
    peak_work_idx = int(np.argmax(vc_work))
    actual_vmax = float(vc_work[peak_work_idx])
    mask = vc_work > 0.97 * actual_vmax
    if not np.any(mask):
        idx = start + peak_work_idx
        return float(r_use[idx]), float(r_use[idx]), float(r_use[idx]), float(vc_use[idx])
    idx = np.flatnonzero(mask)
    r_lo = float(r_use[start + int(idx[0])])
    r_hi = float(r_use[start + int(idx[-1])])
    rvmax = 0.5 * (r_lo + r_hi)
    vmax = float(np.interp(rvmax, r_use, vc_use))
    return r_lo, r_hi, rvmax, vmax


def load_truth_halos(snapshot: Snapshot) -> list[TruthHalo]:
    with h5py.File(snapshot.path, "r") as f:
        cfg = f["Config"].attrs
        starts = np.atleast_1d(cfg["KingStartIndices"]).astype(np.int64)
        n_each = int(cfg["KingParticleCountEach"])
        centres = np.atleast_2d(cfg["KingCentresKpc"]).astype(np.float64)
        translations = np.atleast_2d(cfg["KingTranslationsKms"]).astype(np.float64)
        total_mass = float(cfg["KingMassMsun"])
        core_radius = float(cfg["KingCoreKpc"])
        trunc_radius = float(cfg["KingTruncKpc"])

    truth = []
    r200 = r200_king(total_mass, core_radius, trunc_radius)
    r_grid = np.geomspace(max(core_radius * 1.0e-4, 1.0e-3), trunc_radius, 100_000)
    vc = analytic_vcirc(r_grid, total_mass, core_radius, trunc_radius)
    _, _, rvmax, vmax = rvmax_from_vcirc_profile(r_grid, vc, r_limit=trunc_radius)
    for i, start in enumerate(starts):
        truth.append(
            TruthHalo(
                index=i,
                centre=centres[i] % snapshot.box_size,
                translation=translations[i],
                particle_ids=np.arange(start, start + n_each, dtype=np.int64),
                total_mass=total_mass,
                core_radius=core_radius,
                trunc_radius=trunc_radius,
                r200=r200,
                rvmax=rvmax,
                vmax=vmax,
            )
        )
    return truth


def periodic_delta(points: np.ndarray, centre: np.ndarray, box_size: float) -> np.ndarray:
    dr = points - centre[None, :]
    dr -= box_size * np.round(dr / box_size)
    return dr


def periodic_distance(points: np.ndarray, centre: np.ndarray, box_size: float) -> np.ndarray:
    dr = periodic_delta(points, centre, box_size)
    return np.sqrt(np.sum(dr * dr, axis=1))


def indices_for_ids(snapshot: Snapshot, particle_ids: np.ndarray) -> np.ndarray:
    particle_ids = np.asarray(particle_ids, dtype=np.int64)
    if particle_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    loc = np.searchsorted(snapshot.sorted_ids, particle_ids)
    valid = loc < snapshot.sorted_ids.size
    loc_valid = loc[valid]
    id_valid = particle_ids[valid]
    ok = snapshot.sorted_ids[loc_valid] == id_valid
    return snapshot.id_order[loc_valid[ok]]


class CatalogueError(RuntimeError):
    pass


def require_units(dataset: h5py.Dataset, expected: str) -> None:
    actual = dataset.attrs.get("units")
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual != expected:
        raise CatalogueError(
            f"{dataset.file.filename}:{dataset.name} has units {actual!r}; "
            f"expected {expected!r}"
        )


def read_common_catalogue(path: Path, box_size: float) -> FinderCatalogue:
    with h5py.File(path, "r") as handle:
        if "Header" not in handle or "Haloes" not in handle:
            raise CatalogueError(f"{path} must contain /Header and /Haloes")
        finder_value = handle["Header"].attrs.get("finder", "")
        if isinstance(finder_value, bytes):
            finder_value = finder_value.decode("utf-8")
        finder = str(finder_value).lower()
        if finder not in FINDER_STYLES:
            raise CatalogueError(f"{path} has unsupported finder value {finder!r}")

        haloes = handle["Haloes"]
        required = ("haloid", "centre", "catalogue_radius", "offset", "particle_id")
        missing = [name for name in required if name not in haloes]
        if missing:
            raise CatalogueError(f"{path} lacks /Haloes datasets: {', '.join(missing)}")
        require_units(haloes["centre"], "kpc/h")
        require_units(haloes["catalogue_radius"], "kpc/h")
        halo_ids = np.asarray(haloes["haloid"][...], dtype=np.int64)
        centres = np.asarray(haloes["centre"][...], dtype=np.float64)
        radii = np.asarray(haloes["catalogue_radius"][...], dtype=np.float64)
        offsets = np.asarray(haloes["offset"][...], dtype=np.int64)
        raw_particle_ids = np.asarray(haloes["particle_id"][...], dtype=np.uint64)

    nhalo = halo_ids.size
    if centres.shape != (nhalo, 3) or radii.shape != (nhalo,):
        raise CatalogueError(f"{path} halo-property shapes do not align with haloid")
    if offsets.shape != (nhalo + 1,) or offsets[0] != 0:
        raise CatalogueError(f"{path} offset must have length Nhalo + 1 and start at zero")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != raw_particle_ids.size:
        raise CatalogueError(f"{path} offset does not partition particle_id")
    if raw_particle_ids.size and raw_particle_ids.max() > np.iinfo(np.int64).max:
        raise CatalogueError(f"{path} has particle IDs outside the int64 range")

    particle_ids = raw_particle_ids.astype(np.int64, copy=False)
    membership = {
        int(haloid): particle_ids[offsets[index] : offsets[index + 1]]
        for index, haloid in enumerate(halo_ids)
    }
    label, color = FINDER_STYLES[finder]
    return FinderCatalogue(
        finder,
        label,
        color,
        halo_ids,
        centres % box_size,
        radii,
        membership,
        str(path),
    )


def load_catalogues(
    paths: list[Path], box_size: float, max_subfind_subhalos: int
) -> list[FinderCatalogue]:
    catalogues = [read_common_catalogue(path, box_size) for path in paths]
    keys = [catalogue.key for catalogue in catalogues]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise CatalogueError(f"Duplicate finder catalogues: {', '.join(duplicates)}")
    for catalogue in catalogues:
        if catalogue.key == "subfind" and catalogue.ids.size > max_subfind_subhalos:
            raise CatalogueError(
                f"SUBFIND catalogue contains {catalogue.ids.size} subhalos "
                f"(limit {max_subfind_subhalos})"
            )
    return catalogues


def radial_density_profile(
    radii: np.ndarray,
    masses: np.ndarray,
    r_min: float,
    r_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(radii) & (radii > 0.0) & (radii <= r_max)
    if not np.any(valid):
        return np.empty(0), np.empty(0)
    bins = np.geomspace(max(r_min, np.nanmin(radii[valid]) * 0.9), r_max, n_bins + 1)
    mass_shell, edges = np.histogram(radii[valid], bins=bins, weights=masses[valid])
    volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centres = np.sqrt(edges[:-1] * edges[1:])
    density = mass_shell / volume
    positive = density > 0.0
    return centres[positive], density[positive]


def particle_vcirc_profile(
    radii: np.ndarray,
    masses: np.ndarray,
    r_min: float = 0.0,
    r_limit: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(radii) & (radii > max(r_min, 0.0))
    if r_limit is not None and np.isfinite(r_limit):
        valid &= radii <= r_limit
    if not np.any(valid):
        return np.empty(0), np.empty(0)
    r = radii[valid]
    m = masses[valid]
    order = np.argsort(r)
    r = r[order]
    menc = np.cumsum(m[order])
    vcirc = np.sqrt(G_KPC_KMS2_PER_MSUN * menc / r)
    return r, vcirc


def truth_ids_within_r200(snapshot: Snapshot, truth: TruthHalo) -> np.ndarray:
    idx = indices_for_ids(snapshot, truth.particle_ids)
    radii = periodic_distance(snapshot.pos[idx], truth.centre, snapshot.box_size)
    return truth.particle_ids[radii <= truth.r200]


def match_finder_to_truth(
    snapshot: Snapshot,
    finder: FinderCatalogue,
    truth: TruthHalo,
    n_density_bins: int,
    match_radius_factor: float,
) -> Optional[FinderMatch]:
    if finder.ids.size == 0:
        return None
    dist = periodic_distance(finder.xyz, truth.centre, snapshot.box_size)
    candidates = np.flatnonzero(dist <= match_radius_factor * truth.r200)
    if candidates.size == 0:
        return None
    member_counts = np.array(
        [finder.membership.get(int(finder.ids[i]), np.empty(0, dtype=np.int64)).size for i in candidates],
        dtype=np.int64,
    )
    obj_index = int(candidates[np.lexsort((dist[candidates], -member_counts))[0]])
    hid = int(finder.ids[obj_index])
    member_ids = finder.membership.get(hid, np.empty(0, dtype=np.int64))
    member_idx = indices_for_ids(snapshot, member_ids)
    if member_idx.size == 0:
        return None

    centre = finder.xyz[obj_index] % snapshot.box_size
    radius = finder.radius[obj_index]
    if not (np.isfinite(radius) and radius > 0.0):
        radius = truth.r200
    member_r = periodic_distance(snapshot.pos[member_idx], centre, snapshot.box_size)
    inside = member_r <= radius
    ids_in_r200 = snapshot.ids[member_idx][inside]
    truth_ids_r200 = truth_ids_within_r200(snapshot, truth)
    overlap = np.intersect1d(ids_in_r200, truth_ids_r200, assume_unique=False)
    union_size = np.union1d(ids_in_r200, truth_ids_r200).size

    density_r, density = radial_density_profile(
        member_r,
        snapshot.mass[member_idx],
        r_min=INNER_RADIUS_KPC,
        r_max=max(radius, truth.r200),
        n_bins=n_density_bins,
    )
    vcirc_r, vcirc = particle_vcirc_profile(member_r, snapshot.mass[member_idx], r_min=INNER_RADIUS_KPC, r_limit=radius)
    _, _, rvmax, vmax = rvmax_from_vcirc_profile(vcirc_r, vcirc, r_limit=radius)

    return FinderMatch(
        finder=finder,
        truth=truth,
        obj_index=obj_index,
        centre_error=float(dist[obj_index]),
        ids_in_r200=ids_in_r200,
        truth_ids_in_r200=truth_ids_r200,
        n_overlap=int(overlap.size),
        precision=float(overlap.size / ids_in_r200.size) if ids_in_r200.size else np.nan,
        recall=float(overlap.size / truth_ids_r200.size) if truth_ids_r200.size else np.nan,
        jaccard=float(overlap.size / union_size) if union_size else np.nan,
        r_profile=density_r,
        density_profile=density,
        vcirc_r=vcirc_r,
        vcirc_profile=vcirc,
        rvmax=rvmax,
        vmax=vmax,
    )


def collect_matches(
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    finders: list[FinderCatalogue],
    args: argparse.Namespace,
) -> dict[int, list[FinderMatch]]:
    by_truth: dict[int, list[FinderMatch]] = {truth.index: [] for truth in truth_halos}
    for finder in finders:
        for truth in truth_halos:
            match = match_finder_to_truth(snapshot, finder, truth, args.n_density_bins, args.match_radius_factor)
            if match is not None:
                by_truth[truth.index].append(match)
    return by_truth


def filter_matches_for_finder(
    matches: dict[int, list[FinderMatch]],
    finder: FinderCatalogue,
) -> dict[int, list[FinderMatch]]:
    return {
        truth_index: [match for match in match_list if match.finder is finder]
        for truth_index, match_list in matches.items()
    }


def plot_density_profiles(
    path: Path,
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    matches: dict[int, list[FinderMatch]],
    n_bins: int,
) -> None:
    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.04},
    )
    halo_colors = ["0.50", "0.20", "0.65", "0.35"]
    for truth, halo_color in zip(truth_halos, halo_colors):
        truth_idx = indices_for_ids(snapshot, truth.particle_ids)
        truth_r = periodic_distance(snapshot.pos[truth_idx], truth.centre, snapshot.box_size)
        r_p, rho_p = radial_density_profile(truth_r, snapshot.mass[truth_idx], INNER_RADIUS_KPC, truth.trunc_radius, n_bins)
        if r_p.size:
            fit_at_particles = king_density(r_p, truth.total_mass, truth.core_radius, truth.trunc_radius)
            valid_resid = np.isfinite(fit_at_particles) & (fit_at_particles > 0.0)
            ax.loglog(
                r_p,
                rho_p,
                color=halo_color,
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=f"King {truth.index} particles",
            )
            ax_resid.semilogx(
                r_p[valid_resid],
                (rho_p[valid_resid] - fit_at_particles[valid_resid]) / fit_at_particles[valid_resid],
                color=halo_color,
                marker="o",
                markersize=3,
                linewidth=1.0,
            )

        r_grid = np.geomspace(INNER_RADIUS_KPC, truth.trunc_radius, 1000)
        ax.loglog(
            r_grid,
            king_density(r_grid, truth.total_mass, truth.core_radius, truth.trunc_radius),
            color="k",
            linestyle="-" if truth.index == 0 else "--",
            linewidth=1.9,
            label=f"King {truth.index} analytic fit",
        )
        for match in matches[truth.index]:
            finder_radius = float(match.finder.radius[match.obj_index])
            if match.r_profile.size:
                ax.loglog(
                    match.r_profile,
                    match.density_profile,
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=1.6,
                    label=f"{match.finder.label} King {truth.index}",
                )
                fit_at_finder = king_density(match.r_profile, truth.total_mass, truth.core_radius, truth.trunc_radius)
                valid_finder = np.isfinite(fit_at_finder) & (fit_at_finder > 0.0)
                ax_resid.semilogx(
                    match.r_profile[valid_finder],
                    (match.density_profile[valid_finder] - fit_at_finder[valid_finder]) / fit_at_finder[valid_finder],
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=1.2,
                    alpha=0.85,
                )
            if np.isfinite(finder_radius) and finder_radius > 0.0:
                ax.axvline(
                    finder_radius,
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=1.0,
                    alpha=0.75,
                    label=rf"{match.finder.label} King {truth.index} $R_{{200}}={finder_radius:.1f}$",
                )
                ax_resid.axvline(
                    finder_radius,
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=0.9,
                    alpha=0.75,
                )
    ax.axvline(truth_halos[0].r200, color="k", linestyle=":", linewidth=1.1, label=r"truth $R_{200}$")
    ax.axvline(truth_halos[0].trunc_radius, color="0.3", linestyle=":", linewidth=1.1, label="King truncation")
    ax_resid.axhline(0.0, color="k", linewidth=0.9, alpha=0.7)
    ax_resid.axvline(truth_halos[0].r200, color="k", linestyle=":", linewidth=1.0)
    ax_resid.axvline(truth_halos[0].trunc_radius, color="0.3", linestyle=":", linewidth=1.0)
    ax.set_ylabel(r"$\rho$ [$M_\odot/h\, (kpc/h)^{-3}$]")
    ax.legend(fontsize=8, ncol=2)
    ax_resid.set_xlabel(r"$r$ [kpc/h]")
    ax_resid.set_ylabel(r"$\Delta\rho/\rho_{\rm fit}$")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_vcirc_profiles(
    path: Path,
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    matches: dict[int, list[FinderMatch]],
) -> None:
    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.04},
    )
    halo_colors = ["0.50", "0.20", "0.65", "0.35"]
    for truth, halo_color in zip(truth_halos, halo_colors):
        truth_idx = indices_for_ids(snapshot, truth.particle_ids)
        truth_r = periodic_distance(snapshot.pos[truth_idx], truth.centre, snapshot.box_size)
        truth_vr, truth_vc = particle_vcirc_profile(
            truth_r,
            snapshot.mass[truth_idx],
            r_min=INNER_RADIUS_KPC,
            r_limit=truth.trunc_radius,
        )
        if truth_vr.size:
            fit_at_particles = analytic_vcirc(truth_vr, truth.total_mass, truth.core_radius, truth.trunc_radius)
            valid_resid = fit_at_particles > 0.0
            ax.plot(
                truth_vr,
                truth_vc,
                color=halo_color,
                linewidth=1.0,
                alpha=0.9,
                label=f"King {truth.index} particles",
            )
            ax_resid.semilogx(
                truth_vr[valid_resid],
                (truth_vc[valid_resid] - fit_at_particles[valid_resid]) / fit_at_particles[valid_resid],
                color=halo_color,
                linewidth=1.0,
            )

        r_grid = np.geomspace(INNER_RADIUS_KPC, truth.trunc_radius, 1000)
        vc_grid = analytic_vcirc(r_grid, truth.total_mass, truth.core_radius, truth.trunc_radius)
        ax.plot(
            r_grid,
            vc_grid,
            color="k",
            linestyle="-" if truth.index == 0 else "--",
            linewidth=1.9,
            label=f"King {truth.index} analytic fit",
        )
        ax.axvline(
            truth.rvmax,
            color="k",
            linestyle="-." if truth.index == 0 else ":",
            linewidth=1.2,
            label=rf"King {truth.index} fit $r_{{vmax}}={truth.rvmax:.1f}$",
        )
        ax.axvline(
            truth.r200,
            color="0.25",
            linestyle="--" if truth.index == 0 else (0, (5, 2, 1, 2)),
            linewidth=1.1,
            label=rf"King {truth.index} $R_{{200}}={truth.r200:.1f}$",
        )
        ax_resid.axvline(
            truth.r200,
            color="0.25",
            linestyle="--" if truth.index == 0 else (0, (5, 2, 1, 2)),
            linewidth=1.0,
        )
        ax.plot(truth.rvmax, truth.vmax, marker="o", color="k", markersize=4)

        for match in matches[truth.index]:
            finder_radius = float(match.finder.radius[match.obj_index])
            if match.vcirc_r.size:
                ax.plot(
                    match.vcirc_r,
                    match.vcirc_profile,
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=1.5,
                    label=rf"{match.finder.label} King {truth.index} $r_{{vmax}}={match.rvmax:.1f}$",
                )
                fit_at_finder = analytic_vcirc(
                    match.vcirc_r,
                    truth.total_mass,
                    truth.core_radius,
                    truth.trunc_radius,
                )
                valid_finder = fit_at_finder > 0.0
                ax_resid.semilogx(
                    match.vcirc_r[valid_finder],
                    (match.vcirc_profile[valid_finder] - fit_at_finder[valid_finder]) / fit_at_finder[valid_finder],
                    color=match.finder.color,
                    linestyle="-" if truth.index == 0 else "--",
                    linewidth=1.2,
                    alpha=0.85,
                )
                if np.isfinite(match.rvmax):
                    ax.axvline(match.rvmax, color=match.finder.color, linestyle=":", linewidth=1.1)
                    ax.plot(match.rvmax, match.vmax, marker="o", color=match.finder.color, markersize=4)
                if np.isfinite(finder_radius) and finder_radius > 0.0:
                    ax.axvline(
                        finder_radius,
                        color=match.finder.color,
                        linestyle="-" if truth.index == 0 else "--",
                        linewidth=1.0,
                        alpha=0.75,
                        label=rf"{match.finder.label} King {truth.index} $R_{{200}}={finder_radius:.1f}$",
                    )
                    ax_resid.axvline(
                        finder_radius,
                        color=match.finder.color,
                        linestyle="-" if truth.index == 0 else "--",
                        linewidth=0.9,
                        alpha=0.75,
                    )
    ax.set_xscale("log")
    ax_resid.set_xscale("log")
    ax_resid.axhline(0.0, color="k", linewidth=0.9, alpha=0.7)
    ax.set_ylabel(r"$V_c(<r)$ [km/s]")
    ax.legend(fontsize=8, ncol=2)
    ax_resid.set_xlabel(r"$r$ [kpc/h]")
    ax_resid.set_ylabel(r"$\Delta V_c/V_{c,\rm fit}$")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_membership_residuals(
    path: Path,
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    matches: dict[int, list[FinderMatch]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    halo_colors = ["C0", "C1", "C2", "C3"]
    legend_labels = set()
    all_xy = []

    for truth, halo_color in zip(truth_halos, halo_colors):
        for match in matches[truth.index]:
            hid = int(match.finder.ids[match.obj_index])
            finder_radius = float(match.finder.radius[match.obj_index])
            if not (np.isfinite(finder_radius) and finder_radius > 0.0):
                finder_radius = truth.r200

            background_ids = np.setdiff1d(match.ids_in_r200, truth.particle_ids, assume_unique=False)
            missing_ids = np.setdiff1d(match.truth_ids_in_r200, match.ids_in_r200, assume_unique=False)

            background_idx = indices_for_ids(snapshot, background_ids)
            if background_idx.size:
                background_xy = periodic_delta(snapshot.pos[background_idx], truth.centre, snapshot.box_size)[:, :2]
                all_xy.append(background_xy)
                label = f"King {truth.index} quiet background in {match.finder.label}"
                ax.scatter(
                    background_xy[:, 0],
                    background_xy[:, 1],
                    s=18,
                    marker="x",
                    color=halo_color,
                    alpha=0.85,
                    linewidths=0.8,
                    label=label if label not in legend_labels else None,
                )
                legend_labels.add(label)

            missing_idx = indices_for_ids(snapshot, missing_ids)
            if missing_idx.size:
                missing_xy = periodic_delta(snapshot.pos[missing_idx], truth.centre, snapshot.box_size)[:, :2]
                all_xy.append(missing_xy)
                label = f"King {truth.index} missing from {match.finder.label}"
                ax.scatter(
                    missing_xy[:, 0],
                    missing_xy[:, 1],
                    s=18,
                    marker="o",
                    facecolors="none",
                    edgecolors=halo_color,
                    alpha=0.85,
                    linewidths=0.8,
                    label=label if label not in legend_labels else None,
                )
                legend_labels.add(label)

            centre_xy = periodic_delta(match.finder.xyz[[match.obj_index]], truth.centre, snapshot.box_size)[0, :2]
            all_xy.append(centre_xy[None, :])
            label = f"{match.finder.label} King {truth.index} centre"
            ax.scatter(
                centre_xy[0],
                centre_xy[1],
                s=72,
                marker="*",
                color=halo_color,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
                label=label if label not in legend_labels else None,
            )
            legend_labels.add(label)

            circle_specs = [
                ((0.0, 0.0), truth.r200, "-", 1.6, f"King {truth.index} R200"),
                ((0.0, 0.0), truth.rvmax, ":", 1.8, f"King {truth.index} r(vmax)"),
                (centre_xy, finder_radius, "--", 1.5, f"{match.finder.label} {hid} R200"),
                (centre_xy, match.rvmax, "-.", 1.4, f"{match.finder.label} {hid} r(vmax)"),
            ]
            for circle_centre, radius, linestyle, linewidth, label in circle_specs:
                if not (np.isfinite(radius) and radius > 0.0):
                    continue
                ax.add_patch(
                    plt.Circle(
                        circle_centre,
                        radius,
                        fill=False,
                        color=halo_color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=0.95,
                        label=label if label not in legend_labels else None,
                    )
                )
                legend_labels.add(label)

    if all_xy:
        max_extent = max(float(np.nanmax(np.abs(xy))) for xy in all_xy if xy.size)
    else:
        max_extent = 0.0
    radius_extent = max(
        [truth.r200 for truth in truth_halos]
        + [truth.rvmax for truth in truth_halos]
        + [
            float(match.finder.radius[match.obj_index])
            for truth in truth_halos
            for match in matches[truth.index]
            if np.isfinite(match.finder.radius[match.obj_index])
        ]
        + [
            float(match.rvmax)
            for truth in truth_halos
            for match in matches[truth.index]
            if np.isfinite(match.rvmax)
        ]
    )
    lim = 1.08 * max(max_extent, radius_extent, 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlabel(r"$\Delta x$ from true King centre [kpc/h]")
    ax.set_ylabel(r"$\Delta y$ from true King centre [kpc/h]")
    ax.legend(fontsize=7.2, ncol=2, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_membership_residual_profiles(
    path: Path,
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    matches: dict[int, list[FinderMatch]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    location_colors = {0: "C0", 1: "C1"}
    location_labels = {0: "Centre", 1: "Corner"}
    finder_label = next(
        (match.finder.label for match_list in matches.values() for match in match_list),
        "Finder",
    )

    for truth in truth_halos:
        halo_color = location_colors[truth.index]
        for match in matches[truth.index]:
            king_r200_ids = match.truth_ids_in_r200
            finder_r200_ids = match.ids_in_r200

            king_idx = indices_for_ids(snapshot, king_r200_ids)
            finder_idx = indices_for_ids(snapshot, finder_r200_ids)
            king_r = periodic_distance(snapshot.pos[king_idx], truth.centre, snapshot.box_size)
            finder_r = periodic_distance(snapshot.pos[finder_idx], truth.centre, snapshot.box_size)
            r_pool = np.concatenate((king_r, finder_r))
            if r_pool.size == 0:
                continue
            r_outer = float(np.nanmax(r_pool))
            if not (np.isfinite(r_outer) and r_outer > 0.0):
                r_outer = truth.r200

            missing_ids = np.setdiff1d(king_r200_ids, finder_r200_ids, assume_unique=False)
            finder_extra_ids = np.setdiff1d(finder_r200_ids, king_r200_ids, assume_unique=False)
            missing_idx = indices_for_ids(snapshot, missing_ids)
            extra_idx = indices_for_ids(snapshot, finder_extra_ids)

            missing_r = periodic_distance(snapshot.pos[missing_idx], truth.centre, snapshot.box_size)
            extra_r = periodic_distance(snapshot.pos[extra_idx], truth.centre, snapshot.box_size)

            bins = np.linspace(0.0, r_outer, 81)
            ax.hist(
                missing_r,
                bins=bins,
                histtype="stepfilled",
                color=halo_color,
                alpha=0.2,
                linewidth=0,
            )
            ax.hist(
                missing_r,
                bins=bins,
                histtype="step",
                color=halo_color,
                linestyle="-",
                linewidth=1.6,
            )
            ax.hist(
                extra_r,
                bins=bins,
                histtype="step",
                color=halo_color,
                linestyle="--",
                linewidth=1.6,
            )

    ax.set_xlabel(r"$r$ from true King centre [kpc/h]", fontsize=15)
    ax.set_ylabel("Particles per radial bin", fontsize=15)
    ax.set_xlim(0.0, 275.0)
    ax.set_ylim(0.0, 170.0)
    ax.tick_params(axis="both", labelsize=15)
    ax.text(
        0.03,
        0.5,
        finder_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    location_handles = [
        Line2D([], [], color=location_colors[index], linewidth=2.0,
               label=location_labels[index])
        for index in (0, 1)
    ]
    residual_handles = [
        Line2D([], [], color="0.2", linestyle="-", linewidth=1.6,
               label="Missing from finder $R_{200}$"),
        Line2D([], [], color="0.2", linestyle="--", linewidth=1.6,
               label="Extra in finder $R_{200}$"),
    ]
    ax.legend(
        handles=location_handles + residual_handles,
        fontsize=10.0,
        ncol=2,
        loc="upper left",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def plot_additional_finder_halos(
    output_dir: Path,
    snapshot: Snapshot,
    truth_halos: list[TruthHalo],
    finders: list[FinderCatalogue],
    matches: dict[int, list[FinderMatch]],
    n_bins: int,
) -> list[Path]:
    written = []
    all_matches = [match for match_list in matches.values() for match in match_list]

    for finder in finders:
        matched_indices = {match.obj_index for match in all_matches if match.finder is finder}
        host_matches = [match for match in all_matches if match.finder is finder]

        for obj_index, hid in enumerate(finder.ids):
            if obj_index in matched_indices:
                continue
            member_ids = finder.membership.get(int(hid), np.empty(0, dtype=np.int64))
            member_idx = indices_for_ids(snapshot, member_ids)
            if member_idx.size == 0:
                continue

            centre = finder.xyz[obj_index] % snapshot.box_size
            radius = float(finder.radius[obj_index])
            if not (np.isfinite(radius) and radius > 0.0):
                continue
            local_delta = periodic_delta(snapshot.pos[member_idx], centre, snapshot.box_size)
            local_r = np.sqrt(np.sum(local_delta * local_delta, axis=1))
            local_use = local_r <= 2.0 * radius

            if host_matches:
                host = min(
                    host_matches,
                    key=lambda match: periodic_distance(finder.xyz[[obj_index]], finder.xyz[match.obj_index], snapshot.box_size)[0],
                )
            else:
                host = None

            fig, axs = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
            ax_local, ax_context, ax_density, ax_vcirc = axs.ravel()

            ax_local.scatter(
                local_delta[local_use, 0],
                local_delta[local_use, 1],
                s=9,
                color=finder.color,
                alpha=0.75,
                linewidths=0,
                label=f"{finder.label} {int(hid)} members",
            )
            ax_local.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, color="k", linewidth=1.4, label=r"$R_{200}$"))
            ax_local.scatter(0.0, 0.0, marker="*", s=80, color=finder.color, edgecolors="k", linewidths=0.5)
            ax_local.set_xlim(-2.0 * radius, 2.0 * radius)
            ax_local.set_ylim(-2.0 * radius, 2.0 * radius)
            ax_local.set_aspect("equal", adjustable="box")
            ax_local.set_xlabel(r"$\Delta x$ from finder centre [kpc/h]")
            ax_local.set_ylabel(r"$\Delta y$ from finder centre [kpc/h]")
            ax_local.legend(fontsize=7.5, loc="upper right")

            if host is not None:
                host_centre = finder.xyz[host.obj_index] % snapshot.box_size
                host_radius = float(finder.radius[host.obj_index])
                extra_xy = periodic_delta(centre[None, :], host_centre, snapshot.box_size)[0, :2]
                ax_context.add_patch(
                    plt.Circle((0.0, 0.0), host_radius, fill=False, color="0.25", linewidth=1.4, label="matched host R200")
                )
                ax_context.add_patch(
                    plt.Circle(extra_xy, radius, fill=False, color=finder.color, linestyle="--", linewidth=1.3, label="extra halo R200")
                )
                ax_context.scatter(0.0, 0.0, marker="*", s=80, color="0.25", edgecolors="k", linewidths=0.5, label="matched host")
                ax_context.scatter(extra_xy[0], extra_xy[1], marker="*", s=80, color=finder.color, edgecolors="k", linewidths=0.5, label="extra halo")
                for truth in truth_halos:
                    truth_xy = periodic_delta(truth.centre[None, :], host_centre, snapshot.box_size)[0, :2]
                    ax_context.scatter(truth_xy[0], truth_xy[1], marker="+", s=80, color="k", linewidths=1.2)
                context_lim = 1.08 * max(host_radius, np.max(np.abs(extra_xy)) + radius, 1.0)
                ax_context.set_xlim(-context_lim, context_lim)
                ax_context.set_ylim(-context_lim, context_lim)
            else:
                ax_context.scatter(0.0, 0.0, marker="*", s=80, color=finder.color, edgecolors="k", linewidths=0.5)
                ax_context.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, color=finder.color, linewidth=1.3))
                ax_context.set_xlim(-2.0 * radius, 2.0 * radius)
                ax_context.set_ylim(-2.0 * radius, 2.0 * radius)
            ax_context.set_aspect("equal", adjustable="box")
            ax_context.set_xlabel(r"$\Delta x$ from matched host centre [kpc/h]")
            ax_context.set_ylabel(r"$\Delta y$ from matched host centre [kpc/h]")
            ax_context.legend(fontsize=7.0, loc="upper right")

            density_r, density = radial_density_profile(
                local_r,
                snapshot.mass[member_idx],
                r_min=max(np.nanmin(local_r[local_r > 0.0]) * 0.9 if np.any(local_r > 0.0) else INNER_RADIUS_KPC, 1.0e-3),
                r_max=2.0 * radius,
                n_bins=n_bins,
            )
            if density_r.size:
                ax_density.loglog(density_r, density, color=finder.color, linewidth=1.5)
            ax_density.axvline(radius, color="k", linestyle="--", linewidth=1.0, label=r"$R_{200}$")
            ax_density.set_xlabel(r"$r$ from finder centre [kpc/h]")
            ax_density.set_ylabel(r"$\rho$ [$M_\odot/h\, (kpc/h)^{-3}$]")
            ax_density.legend(fontsize=7.5)

            vcirc_r, vcirc = particle_vcirc_profile(local_r, snapshot.mass[member_idx], r_min=0.0, r_limit=2.0 * radius)
            if vcirc_r.size:
                _, _, rvmax, vmax = rvmax_from_vcirc_profile(vcirc_r, vcirc, r_limit=2.0 * radius)
                ax_vcirc.plot(vcirc_r, vcirc, color=finder.color, linewidth=1.5)
                if np.isfinite(rvmax):
                    ax_vcirc.axvline(rvmax, color=finder.color, linestyle=":", linewidth=1.2, label=rf"$r(vmax)={rvmax:.2f}$")
                    ax_vcirc.plot(rvmax, vmax, marker="o", color=finder.color, markersize=4)
            ax_vcirc.axvline(radius, color="k", linestyle="--", linewidth=1.0, label=r"$R_{200}$")
            ax_vcirc.set_xlabel(r"$r$ from finder centre [kpc/h]")
            ax_vcirc.set_ylabel(r"$V_c(<r)$ [km/s]")
            ax_vcirc.legend(fontsize=7.5)

            path = output_dir / f"unit_test_king_extra_{safe_slug(finder.key)}_{int(hid)}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    return written


def summary_lines(
    truth_halos: list[TruthHalo],
    finders: list[FinderCatalogue],
    matches: dict[int, list[FinderMatch]],
) -> list[str]:
    lines = [f"Loaded {len(finders)} finder result(s): {', '.join(f.label for f in finders) if finders else 'none'}"]
    for finder in finders:
        lines.append(f"  {finder.label} source: {finder.source}")
    for truth in truth_halos:
        lines.append(
            f"Truth King {truth.index}: centre=({truth.centre[0]:.3f}, {truth.centre[1]:.3f}, {truth.centre[2]:.3f}) "
            f"R200={truth.r200:.6g} kpc/h rvmax={truth.rvmax:.6g} kpc/h vmax={truth.vmax:.6g} km/s"
        )
        if not matches[truth.index]:
            lines.append("  no matched finder halo within requested R200 aperture")
            continue
        for match in matches[truth.index]:
            hid = int(match.finder.ids[match.obj_index])
            lines.append(
                f"  {match.finder.label} halo {hid}: centre_error={match.centre_error:.6g} kpc/h "
                f"ids_in_r200={match.ids_in_r200.size} truth_ids_in_r200={match.truth_ids_in_r200.size} "
                f"overlap={match.n_overlap} precision={match.precision:.4f} recall={match.recall:.4f} "
                f"jaccard={match.jaccard:.4f} rvmax={match.rvmax:.6g} kpc/h vmax={match.vmax:.6g} km/s"
            )
    return lines


def main() -> None:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    truth_halos = load_truth_halos(snapshot)
    catalogue_paths = args.catalogue or [
        Path(config["king_catalogue"])
        for config in FINDERS.values()
        if config.get("king_catalogue")
        and Path(config["king_catalogue"]).is_file()
    ]
    if not catalogue_paths:
        raise CatalogueError(
            "No --catalogue inputs and no configured King catalogues were found"
        )
    finders = load_catalogues(
        catalogue_paths, snapshot.box_size, args.max_subfind_subhalos
    )
    matches = collect_matches(snapshot, truth_halos, finders, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in args.output_dir.glob("unit_test_king_extra_*.png"):
        stale_path.unlink()
    density_path = args.output_dir / "unit_test_king_density_profiles.png"
    vcirc_path = args.output_dir / "unit_test_king_vcirc_profiles.png"
    summary_path = args.output_dir / "unit_test_king_analysis_summary.txt"
    plot_density_profiles(density_path, snapshot, truth_halos, matches, args.n_density_bins)
    plot_vcirc_profiles(vcirc_path, snapshot, truth_halos, matches)
    membership_paths = []
    for finder in finders:
        finder_matches = filter_matches_for_finder(matches, finder)
        if not any(finder_matches.values()):
            continue
        finder_slug = safe_slug(finder.key)
        membership_path = args.output_dir / f"unit_test_king_membership_residuals_{finder_slug}.png"
        membership_profile_path = args.output_dir / f"unit_test_king_membership_residual_profiles_{finder_slug}.png"
        plot_membership_residuals(membership_path, snapshot, truth_halos, finder_matches)
        plot_membership_residual_profiles(membership_profile_path, snapshot, truth_halos, finder_matches)
        membership_paths.extend((membership_path, membership_profile_path))
    extra_halo_paths = plot_additional_finder_halos(args.output_dir, snapshot, truth_halos, finders, matches, args.n_density_bins)

    lines = summary_lines(truth_halos, finders, matches)
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {density_path}")
    print(f"Wrote {vcirc_path}")
    for path in membership_paths:
        print(f"Wrote {path}")
    for path in extra_halo_paths:
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
