#!/usr/bin/env python3
"""
Overlay analysis for configured particle-derived halo catalogues
===============================================================

Reads the derived text catalogues written by ``LR_base_analysis.py``:

Input paths, display names, colours, line styles, and the matching reference
finder are declared in ``finder_config.py``.

Each file must contain rows:
  # id x y z M200derived R200derived rvmax vmax_kms

All positions/radii are in kpc/h, masses in Msun/h, and vmax in km/s.

Outputs
-------
Figure 1: Cumulative mass functions for level 0, 1, 2 for all catalogues.
Figure 2: Local environment around the reference finder's largest object using R200derived circles.
Figure 3: Deviations from the mean cumulative mass function (three panels: level 0, level 1, level 2).
Figure 4: Cumulative vmax functions for level 0, 1, 2 for all catalogues.
Figure 5: Local environment around the reference finder's largest object using 2*rvmax circles.
Figure 6: Deviations from the mean cumulative vmax function (three panels: level 0, level 1, level 2).
Figure 7: Three-panel count-based radial summary: differential, cumulative, normalised cumulative.
Figure 8: Three-panel mass-based radial summary: differential, cumulative, normalised cumulative.
Figure 9: R200 vs rvmax with marginal histograms.
Matched diagnostics: finder member particles for one matched host/subhalo pair,
plus separate host and subhalo circular-velocity profiles. Memberships come
from the common HDF5 catalogues written by ``convert_halo_membership.py``.

Notes
-----
- Host/level assignment uses a periodic KD-tree and a geometric nesting rule:
    object i is inside object j if mass[j] > mass[i] and |r_i-r_j| < R_j.
  Immediate host chosen as smallest-mass enclosing larger object.
- Radial subhalo plots use the immediate host for each subhalo (level >= 1), and use
  r / R_host where r is the periodic distance to that host centre.
- Purely geometric; does not use particle membership.
- Python 3.8+.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from finder_config import FINDERS, REFERENCE_FINDER, finder_keys, finder_style
from plot_config import apply_plot_style


apply_plot_style(plt)


DERIVED_FILES = {key: Path(config["derived"]) for key, config in FINDERS.items()}
COMMON_FILES = {key: Path(config["catalogue"]) for key, config in FINDERS.items()}
SNAPSHOT_FILE = Path("./Simulation/data/snap_128.hdf5")
FIGURE_DIR = Path("./output_cosmo")

BOX_SIZE_KPC = 100000.0
BOX_SIZE_MPC = 100.0
VOLUME = BOX_SIZE_MPC ** 3

MIN_HALO_MASS = 3.0e10  # Msun/h
MIN_FIG4_VMAX = 100.0   # km/s
MIN_DEVIATION_BIN_COUNT = 10
G_KPC_KMS2_PER_MSUN = 4.30091e-6
VCIRC_PROFILE_SKIP = 5

R_LOCAL_OCTANT = 2000.0
RADIAL_BINS = np.linspace(0.0, 1., 50)


FINDER_KEYS = finder_keys()
CATALOGUES = tuple(finder_style(key) for key in FINDER_KEYS)
LEVEL_LINESTYLES = {
    0: "-",
    1: "--",
    2: ":",
}


def save_figure(fig, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def load_derived_catalogue(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError("Missing input file: {}".format(path))

    rows = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            try:
                object_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                m = float(parts[4])
                r = float(parts[5])
                rv = float(parts[6])
                vmax = float(parts[7])
            except ValueError:
                continue
            rows.append((object_id, x, y, z, m, r, rv, vmax))

    if not rows:
        raise ValueError("No data rows parsed from {}".format(path))

    object_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    arr = np.asarray([row[1:] for row in rows], dtype=np.float64)
    xyz = arr[:, 0:3]
    mass = arr[:, 3]
    radius = arr[:, 4]
    rvmax = arr[:, 5]
    vmax = arr[:, 6]

    valid = np.isfinite(mass) & (mass >= MIN_HALO_MASS)
    valid &= np.isfinite(radius) & (radius >= 0.0)
    valid &= np.all(np.isfinite(xyz), axis=1)

    object_id = object_id[valid]
    xyz = xyz[valid]
    mass = mass[valid]
    radius = radius[valid]
    rvmax = rvmax[valid]
    vmax = vmax[valid]

    if mass.size == 0:
        raise ValueError("No valid rows after filtering in {}".format(path))

    return object_id, mass, xyz, radius, rvmax, vmax


def cumulative_function(values: np.ndarray, volume: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    v_sorted = np.sort(values)[::-1]
    n_cum = np.arange(1, v_sorted.size + 1, dtype=np.float64)
    n_density = n_cum / volume
    return np.log10(v_sorted), np.log10(n_density)


def mass_within_rvmax(rvmax: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    mass_rvmax = np.full(rvmax.size, np.nan, dtype=np.float64)
    valid = np.isfinite(rvmax) & np.isfinite(vmax) & (rvmax > 0.0) & (vmax > 0.0)
    mass_rvmax[valid] = (vmax[valid] * vmax[valid] * rvmax[valid]) / G_KPC_KMS2_PER_MSUN
    return mass_rvmax


def subhalo_redefined_mass(m200: np.ndarray, rvmax: np.ndarray, vmax: np.ndarray, level: np.ndarray) -> np.ndarray:
    out = m200.copy()
    mrv = mass_within_rvmax(rvmax, vmax)
    is_subhalo = level >= 1
    out[is_subhalo] = mrv[is_subhalo]
    return out


def positive_log_bins(values: List[np.ndarray], nbins: int = 50) -> np.ndarray:
    combined = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    combined = combined[np.isfinite(combined) & (combined > 0.0)]
    if combined.size == 0:
        raise ValueError("No positive finite values available for histogram bins.")
    return np.logspace(np.log10(float(combined.min())), np.log10(float(combined.max())), nbins + 1)


def periodic_delta(xyz: np.ndarray, centre: np.ndarray, boxsize: float) -> np.ndarray:
    dr = xyz - centre[None, :]
    dr -= boxsize * np.round(dr / boxsize)
    return dr


def assign_hosts_and_levels_periodic(
    xyz: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    boxsize: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = mass.size
    tree = cKDTree(xyz, boxsize=boxsize)

    parent = np.full(n, -1, dtype=np.int64)
    parent_mass = np.full(n, np.inf, dtype=np.float64)

    order = np.argsort(mass)[::-1]
    for host_idx in order:
        host_r = radius[host_idx]
        if host_r <= 0.0:
            continue
        neigh = tree.query_ball_point(xyz[host_idx], r=host_r)
        if not neigh:
            continue
        for j in neigh:
            if j == host_idx:
                continue
            if mass[j] >= mass[host_idx]:
                continue
            if mass[host_idx] < parent_mass[j]:
                parent[j] = host_idx
                parent_mass[j] = mass[host_idx]

    level = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p = parent[i]
        lev = 0
        while p != -1:
            lev += 1
            p = parent[p]
        level[i] = lev

    return parent, level


def largest_index(mass: np.ndarray) -> int:
    return int(np.argmax(mass))


def adaptive_log_thresholds(
    log_values_list: List[np.ndarray],
    lower_limit: float,
    min_per_bin: int = MIN_DEVIATION_BIN_COUNT,
) -> np.ndarray:
    logs = []
    log_lower = np.log10(lower_limit)
    for values in log_values_list:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr >= log_lower)]
        if arr.size >= min_per_bin:
            logs.append(arr)

    if len(logs) < 2:
        return np.empty(0, dtype=np.float64)

    lo = max(log_lower, max(float(np.min(arr)) for arr in logs))
    hi = min(float(np.max(arr)) for arr in logs)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.empty(0, dtype=np.float64)

    thresholds = []
    current_hi = hi
    while current_hi > lo:
        candidates = []
        for arr in logs:
            in_bin_range = arr[(arr <= current_hi) & (arr >= lo)]
            if in_bin_range.size < min_per_bin:
                return np.asarray(thresholds, dtype=np.float64)
            desc = np.sort(in_bin_range)[::-1]
            candidates.append(float(desc[min_per_bin - 1]))

        next_lo = max(lo, min(candidates))
        if not np.isfinite(next_lo) or next_lo >= current_hi:
            break

        if all(np.count_nonzero((arr >= next_lo) & (arr <= current_hi)) >= min_per_bin for arr in logs):
            thresholds.append(next_lo)
        if next_lo <= lo:
            break
        current_hi = np.nextafter(next_lo, -np.inf)

    return np.asarray(thresholds, dtype=np.float64)


def binned_cumulative_deviation_curves(
    labelled_values: List[Tuple[str, np.ndarray]],
    lower_limit: float,
    volume: float,
    min_per_bin: int = MIN_DEVIATION_BIN_COUNT,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    filtered = []
    log_values = []
    for label, values in labelled_values:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr >= lower_limit)]
        if arr.size >= min_per_bin:
            logs = np.log10(arr)
            filtered.append((label, logs))
            log_values.append(logs)

    thresholds = adaptive_log_thresholds(log_values, lower_limit, min_per_bin=min_per_bin)
    if thresholds.size == 0:
        return []

    curves = []
    for label, logs in filtered:
        counts = np.array([np.count_nonzero(logs >= threshold) for threshold in thresholds], dtype=np.float64)
        good = counts > 0.0
        curves.append((label, thresholds[good], np.log10(counts[good] / volume)))
    return curves


def deviation_panel_binned_multi(ax, curves: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    usable = [(label, lx, ly) for (label, lx, ly) in curves if lx.size >= 1 and ly.size >= 1]
    if len(usable) < 2:
        ax.text(0.05, 0.5, "Insufficient data", transform=ax.transAxes)
        return

    min_len = min(lx.size for _, lx, _ in usable)
    x = usable[0][1][:min_len]
    y_stack = np.vstack([ly[:min_len] for _, _, ly in usable])
    yavg = np.mean(y_stack, axis=0)

    for label, _, ly in usable:
        ax.plot(x, ly[:min_len] - yavg, marker="o", markersize=3.0, label="{} - avg".format(label))

    ax.axhline(0.0, linestyle="--", linewidth=1.0)


def plot_cumulative_by_level_multi(ax, values_list, levels_list, x_label):
    for lev, lev_ls in LEVEL_LINESTYLES.items():
        for (label, color, _finder_ls), values, levels in zip(CATALOGUES, values_list, levels_list):
            mask = levels == lev
            if np.any(mask):
                x, y = cumulative_function(values[mask], VOLUME)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=lev_ls,
                    alpha=0.9,
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}(N(>x)/V)$")
    finder_handles = [
        Line2D([0], [0], color=color, linestyle="-", label=label)
        for label, color, _ in CATALOGUES
    ]
    level_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle, label=f"Level {level}")
        for level, linestyle in LEVEL_LINESTYLES.items()
    ]
    ax.legend(
        handles=finder_handles + level_handles,
        frameon=False,
        loc="upper right",
        ncol=1,
    )


def plot_local_circles(ax, xyz, radii, centre, indices, boxsize, linestyle="-", color="C0", linewidth=1.2, alpha=0.8):
    for idx in indices:
        dx, dy, _ = periodic_delta(xyz[idx:idx + 1], centre, boxsize)[0]
        ax.add_patch(
            plt.Circle(
                (dx, dy),
                float(radii[idx]),
                fill=False,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                edgecolor=color,
            )
        )


def plot_r200_rvmax_joint(catalogue_data):
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=(4.0, 1.25),
        height_ratios=(1.25, 4.0),
        hspace=0.06,
        wspace=0.06,
    )
    ax_hist_r200 = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_hist_r200)
    ax_hist_rvmax = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    filtered = []
    for label, color, r200, rvmax in catalogue_data:
        mask = np.isfinite(r200) & np.isfinite(rvmax) & (r200 > 0.0) & (rvmax > 0.0)
        if np.any(mask):
            filtered.append((label, color, r200[mask], rvmax[mask]))

    r200_bins = positive_log_bins([r200 for _, _, r200, _ in filtered])
    rvmax_bins = positive_log_bins([rvmax for _, _, _, rvmax in filtered])

    for label, color, r200, rvmax in filtered:
        ax_scatter.plot(
            r200,
            rvmax,
            linestyle="none",
            marker=".",
            markersize=2.5,
            alpha=0.55,
            color=color,
            label=label,
        )
        ax_hist_r200.hist(r200, bins=r200_bins, histtype="step", linewidth=1.4, color=color)
        ax_hist_rvmax.hist(
            rvmax,
            bins=rvmax_bins,
            histtype="step",
            linewidth=1.4,
            orientation="horizontal",
            color=color,
        )

    ax_scatter.set_xlabel(r"$R_{200}$")
    ax_scatter.set_ylabel(r"$r_{\rm vmax}$")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_hist_r200.set_xscale("log")
    ax_hist_rvmax.set_yscale("log")
    ax_scatter.legend(frameon=False)
    ax_hist_r200.set_ylabel("Count")
    ax_hist_rvmax.set_xlabel("Count")
    ax_hist_r200.tick_params(axis="x", labelbottom=False)
    ax_hist_rvmax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.97)
    return fig


def subhalo_radial_data(
    xyz: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    parent: np.ndarray,
    level: np.ndarray,
    boxsize: float,
    host_mass_min: Optional[float] = None,
    subhalo_mass: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = level >= 1
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    host = parent[idx]
    good = host >= 0
    idx = idx[good]
    host = host[good]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    host_r = radius[host]
    good = np.isfinite(host_r) & (host_r > 0.0)
    if host_mass_min is not None:
        host_m = mass[host]
        good &= np.isfinite(host_m) & (host_m > host_mass_min)
    idx = idx[good]
    host = host[good]
    host_r = host_r[good]
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    dr = xyz[idx] - xyz[host]
    dr -= boxsize * np.round(dr / boxsize)
    rr = np.sqrt(np.sum(dr * dr, axis=1)) / host_r

    mass_weight = mass if subhalo_mass is None else subhalo_mass
    valid = np.isfinite(rr) & np.isfinite(mass_weight[idx]) & (mass_weight[idx] > 0.0)
    rr = rr[valid]
    mm = mass_weight[idx][valid]
    return rr.astype(np.float64), mm.astype(np.float64)




def cumulative_from_sorted_radius(rr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rr.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    x = np.sort(rr.astype(np.float64))
    y = np.arange(1, x.size + 1, dtype=np.float64)
    return x, y


def cumulative_mass_from_sorted_radius(rr: np.ndarray, mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rr.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    order = np.argsort(rr.astype(np.float64))
    x = rr[order].astype(np.float64)
    y = np.cumsum(mm[order].astype(np.float64))
    return x, y

def subhalo_radial_profiles(
    xyz: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    parent: np.ndarray,
    level: np.ndarray,
    boxsize: float,
    bins: np.ndarray,
    host_mass_min: Optional[float] = None,
    subhalo_mass: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rr_count, _ = subhalo_radial_data(
        xyz,
        mass,
        radius,
        parent,
        level,
        boxsize,
        host_mass_min=host_mass_min,
    )
    rr_mass, mm = subhalo_radial_data(
        xyz,
        mass,
        radius,
        parent,
        level,
        boxsize,
        host_mass_min=host_mass_min,
        subhalo_mass=subhalo_mass,
    )

    counts, _ = np.histogram(rr_count, bins=bins)
    mass_sum, _ = np.histogram(rr_mass, bins=bins, weights=mm)
    return counts.astype(np.float64), mass_sum.astype(np.float64)



def load_snapshot_positions(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError("Missing snapshot file: {}".format(path))

    pos_list = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("PartType"):
                continue
            grp = f[key]
            if "Coordinates" not in grp:
                continue
            coords = np.asarray(grp["Coordinates"], dtype=np.float64)
            if coords.ndim == 2 and coords.shape[1] == 3 and coords.size > 0:
                pos_list.append(coords)

    if not pos_list:
        raise ValueError("No particle Coordinates datasets found in {}".format(path))

    xyz = np.vstack(pos_list)
    if not np.all(np.isfinite(xyz)):
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    return xyz


def local_octant_particle_density(xyz: np.ndarray, centre: np.ndarray, r_local: float, boxsize: float, nnei: int = 20):
    dr = periodic_delta(xyz, centre, boxsize)
    rr = np.sqrt(np.sum(dr * dr, axis=1))
    mask = (rr <= r_local) & (dr[:, 0] >= 0.0) & (dr[:, 1] >= 0.0) & (dr[:, 2] >= 0.0)
    dloc = dr[mask]
    if dloc.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    if dloc.shape[0] == 1:
        return dloc, np.ones(1, dtype=np.float64)

    k = min(nnei + 1, dloc.shape[0])
    tree = cKDTree(dloc)
    dist, _ = tree.query(dloc, k=k)
    r_n = dist[:, -1]
    tiny = np.finfo(np.float64).tiny
    rho = float(nnei) / ((4.0 / 3.0) * np.pi * np.maximum(r_n, tiny) ** 3)
    return dloc, rho


def octant_local_indices(xyz: np.ndarray, centre: np.ndarray, r_local: float, boxsize: float) -> np.ndarray:
    dr = periodic_delta(xyz, centre, boxsize)
    rr2 = np.sum(dr * dr, axis=1)
    mask = (rr2 <= r_local * r_local) & (dr[:, 0] >= 0.0) & (dr[:, 1] >= 0.0) & (dr[:, 2] >= 0.0)
    return np.where(mask)[0]


def include_indices(indices: np.ndarray, required: List[int]) -> np.ndarray:
    """Return sorted unique indices, forcing selected objects into a local plot."""
    if len(required) == 0:
        return np.unique(indices)
    return np.unique(np.concatenate([indices, np.asarray(required, dtype=np.int64)]))


def underplot_particles(ax, dxyz: np.ndarray, density: np.ndarray) -> None:
    if dxyz.shape[0] == 0:
        return
    cval = np.log10(np.maximum(density, np.finfo(np.float64).tiny))
    ax.scatter(dxyz[:, 0], dxyz[:, 1], c=cval, s=3, alpha=0.15, linewidths=0, cmap="viridis", zorder=0)


def format_local_quadrant_axes(ax) -> None:
    ax.set_xlabel("x[kpc/h]")
    ax.set_ylabel("y[kpc/h]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, R_LOCAL_OCTANT)
    ax.set_ylim(0.0, R_LOCAL_OCTANT)


def add_simulation_linestyle_legend(ax):
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=label)
        for label, color, linestyle in CATALOGUES
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)


@dataclass
class CosmoData:
    key: str
    label: str
    color: str
    linestyle: str
    ids: np.ndarray
    mass: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    rvmax: np.ndarray
    vmax: np.ndarray
    parent: np.ndarray
    level: np.ndarray
    subhalo_mass: np.ndarray
    common_path: Path


@dataclass
class MatchedFinder:
    key: str
    label: str
    color: str
    ids: np.ndarray
    xyz: np.ndarray
    catalogue_radius: np.ndarray
    m200: np.ndarray
    r200: np.ndarray
    rvmax: np.ndarray
    common_path: Path
    membership_ranges: dict[int, tuple[int, int]]
    parent: np.ndarray
    level: np.ndarray


def load_common_memberships(
    path: Path,
    expected_finder: str,
    derived_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
    """Align common catalogue properties and memberships to derived halo IDs."""
    with h5py.File(path, "r") as handle:
        if "Header" not in handle or "Haloes" not in handle:
            raise ValueError(f"{path} must contain /Header and /Haloes")
        finder = handle["Header"].attrs.get("finder", "")
        if isinstance(finder, bytes):
            finder = finder.decode("utf-8")
        if str(finder).lower() != expected_finder:
            raise ValueError(
                f"{path} contains finder {finder!r}; expected {expected_finder!r}"
            )
        haloes = handle["Haloes"]
        required = ("haloid", "centre", "catalogue_radius", "offset", "particle_id")
        missing = [name for name in required if name not in haloes]
        if missing:
            raise ValueError(f"{path} lacks /Haloes datasets: {', '.join(missing)}")
        common_ids = np.asarray(haloes["haloid"], dtype=np.int64)
        centres = np.asarray(haloes["centre"], dtype=np.float64)
        radii = np.asarray(haloes["catalogue_radius"], dtype=np.float64)
        offsets = np.asarray(haloes["offset"], dtype=np.int64)
        particle_count = int(haloes["particle_id"].size)

    if centres.shape != (common_ids.size, 3) or radii.shape != (common_ids.size,):
        raise ValueError(f"{path} halo properties do not align with haloid")
    if offsets.shape != (common_ids.size + 1,) or offsets[0] != 0:
        raise ValueError(f"{path} has invalid membership offsets")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != particle_count:
        raise ValueError(f"{path} offsets do not partition particle_id")

    index_by_id = {int(haloid): index for index, haloid in enumerate(common_ids)}
    missing_ids = [int(haloid) for haloid in derived_ids if int(haloid) not in index_by_id]
    if missing_ids:
        raise ValueError(f"{path} lacks derived halo IDs: {missing_ids[:10]}")
    indices = np.asarray([index_by_id[int(haloid)] for haloid in derived_ids])
    membership_ranges = {
        int(haloid): (int(offsets[index]), int(offsets[index + 1]))
        for haloid, index in zip(derived_ids, indices)
    }
    return centres[indices] % BOX_SIZE_KPC, radii[indices], membership_ranges


def build_matched_finder(
    key: str,
    common_path: Path,
    ids: np.ndarray,
    mass: np.ndarray,
    xyz: np.ndarray,
    r200: np.ndarray,
    rvmax: np.ndarray,
    parent: np.ndarray,
    level: np.ndarray,
) -> MatchedFinder:
    common_xyz, catalogue_radius, membership_ranges = load_common_memberships(
        common_path, key, ids
    )
    if not np.allclose(common_xyz, xyz % BOX_SIZE_KPC, rtol=0.0, atol=1.0e-3):
        raise ValueError(f"{common_path} centres do not match its derived catalogue")
    label, color, _ = finder_style(key)
    return MatchedFinder(
        key, label, color, ids, xyz, catalogue_radius, mass, r200, rvmax,
        common_path, membership_ranges, parent, level,
    )


def top_level_host(parent: np.ndarray, index: int) -> int:
    while parent[index] != -1:
        index = int(parent[index])
    return index


def nearest_counterpart(
    finder: MatchedFinder,
    centre: np.ndarray,
    max_radius: float,
    level_mode: str,
) -> int | None:
    eligible = np.all(np.isfinite(finder.xyz), axis=1)
    eligible &= finder.level == 0 if level_mode == "host" else finder.level >= 1
    eligible &= np.asarray(
        [
            finder.membership_ranges[int(haloid)][1]
            > finder.membership_ranges[int(haloid)][0]
            for haloid in finder.ids
        ]
    )
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return None
    distances = np.sqrt(
        np.sum(periodic_delta(finder.xyz[indices], centre, BOX_SIZE_KPC) ** 2, axis=1)
    )
    local = int(np.argmin(distances))
    return int(indices[local]) if distances[local] <= max_radius else None


def select_matched_pair(
    finders: dict[str, MatchedFinder],
    target_sub_log_mass: float,
    host_match_radius: float,
    sub_match_radius: float,
    reference_host_id: int | None,
    reference_sub_id: int | None,
) -> dict[str, tuple[int, int]]:
    reference = finders[REFERENCE_FINDER]
    id_to_index = {
        int(haloid): index for index, haloid in enumerate(reference.ids)
    }
    if reference_sub_id is not None:
        if reference_sub_id not in id_to_index:
            raise ValueError(
                f"{reference.label} subhalo ID {reference_sub_id} is absent"
            )
        candidates = np.asarray([id_to_index[reference_sub_id]], dtype=np.int64)
    else:
        candidates = np.flatnonzero(
            (reference.level >= 1)
            & np.isfinite(reference.m200)
            & (reference.m200 > 0.0)
        )
        score = np.abs(
            np.log10(reference.m200[candidates]) - target_sub_log_mass
        )
        candidates = candidates[np.argsort(score)]

    forced_host = None
    if reference_host_id is not None:
        if reference_host_id not in id_to_index:
            raise ValueError(
                f"{reference.label} host ID {reference_host_id} is absent"
            )
        forced_host = id_to_index[reference_host_id]

    for sub_index in candidates:
        host_index = top_level_host(reference.parent, int(sub_index))
        if forced_host is not None and host_index != forced_host:
            continue
        start, stop = reference.membership_ranges[int(reference.ids[sub_index])]
        if stop == start:
            continue
        matches = {REFERENCE_FINDER: (host_index, int(sub_index))}
        for key in FINDER_KEYS:
            if key == REFERENCE_FINDER:
                continue
            host = nearest_counterpart(
                finders[key], reference.xyz[host_index], host_match_radius, "host"
            )
            sub = nearest_counterpart(
                finders[key], reference.xyz[sub_index], sub_match_radius, "subhalo"
            )
            if host is None or sub is None or host == sub:
                break
            matches[key] = (host, sub)
        if len(matches) == len(FINDER_KEYS):
            return matches
    raise RuntimeError(
        f"Could not find a matched {reference.label} host/subhalo pair; increase "
        "the match radii or select explicit reference IDs"
    )


def load_membership_snapshot(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["PartType1"]
        positions = np.asarray(group["Coordinates"], dtype=np.float64)
        particle_ids = np.asarray(group["ParticleIDs"], dtype=np.uint64)
        if "Masses" in group:
            masses = np.asarray(group["Masses"], dtype=np.float64) * 1.0e10
        else:
            mass = float(np.asarray(handle["Header"].attrs["MassTable"])[1]) * 1.0e10
            masses = np.full(particle_ids.size, mass)
    order = np.argsort(particle_ids)
    return particle_ids[order], positions[order], masses[order]


def resolve_member_particles(
    finder: MatchedFinder,
    object_index: int,
    sorted_ids: np.ndarray,
    sorted_positions: np.ndarray,
    sorted_masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start, stop = finder.membership_ranges[int(finder.ids[object_index])]
    with h5py.File(finder.common_path, "r") as handle:
        ids = np.asarray(
            handle["Haloes"]["particle_id"][start:stop], dtype=np.uint64
        )
    locations = np.searchsorted(sorted_ids, ids)
    valid = locations < sorted_ids.size
    matched = np.zeros(ids.size, dtype=bool)
    matched[valid] = sorted_ids[locations[valid]] == ids[valid]
    locations = locations[matched]
    return sorted_positions[locations], sorted_masses[locations]


def matched_vcirc_profile(
    finder: MatchedFinder,
    object_index: int,
    positions: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if positions.size == 0:
        return np.empty(0), np.empty(0)
    radii = np.sqrt(
        np.sum(periodic_delta(positions, finder.xyz[object_index], BOX_SIZE_KPC) ** 2, axis=1)
    )
    order = np.argsort(radii)
    radii = radii[order]
    masses = masses[order]
    positive = radii > 0.0
    radii = radii[positive]
    masses = masses[positive]
    vcirc = np.sqrt(G_KPC_KMS2_PER_MSUN * np.cumsum(masses) / radii)
    if radii.size > VCIRC_PROFILE_SKIP + 1:
        radii = radii[VCIRC_PROFILE_SKIP:]
        vcirc = vcirc[VCIRC_PROFILE_SKIP:]
    return radii, vcirc


def matched_output_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def plot_matched_vcirc(
    finders: dict[str, MatchedFinder],
    matches: dict[str, tuple[int, int]],
    profiles: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    row: int,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in FINDER_KEYS:
        finder = finders[key]
        index = matches[key][row]
        radii, vcirc = profiles[(key, row)]
        if radii.size == 0:
            continue
        ax.plot(radii, vcirc, color=finder.color, label=f"{finder.label} {int(finder.ids[index])}")
        peak_radius = finder.rvmax[index]
        if radii[0] <= peak_radius <= radii[-1]:
            ax.axvline(peak_radius, color=finder.color, linestyle=":", linewidth=1.2)
            ax.plot(peak_radius, np.interp(peak_radius, radii, vcirc), "o", color=finder.color, markersize=4)
    ax.set_xlabel("r [kpc/h]")
    ax.set_ylabel(r"$V_{\rm circ}$ [km s$^{-1}$]")
    if row == 0:
        ax.set_ylim(bottom=350.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def plot_matched_members(
    finders: dict[str, MatchedFinder],
    matches: dict[str, tuple[int, int]],
    snapshot: Path,
    output: Path,
    max_points: int,
    seed: int,
) -> None:
    sorted_ids, sorted_positions, sorted_masses = load_membership_snapshot(snapshot)
    rng = np.random.default_rng(seed)
    nfinders = len(FINDER_KEYS)
    fig, axes = plt.subplots(
        2, nfinders, figsize=(4.7 * nfinders, 9), sharex="row", sharey="row",
        squeeze=False,
    )
    profiles: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    limits = []
    for row in range(2):
        values = [50.0]
        for key in FINDER_KEYS:
            finder = finders[key]
            index = matches[key][row]
            values.extend(
                value for value in (
                    finder.catalogue_radius[index], finder.r200[index], finder.rvmax[index]
                ) if np.isfinite(value) and value > 0.0
            )
        limits.append(1.25 * max(values))

    for column, key in enumerate(FINDER_KEYS):
        finder = finders[key]
        for row in range(2):
            index = matches[key][row]
            positions, masses = resolve_member_particles(
                finder, index, sorted_ids, sorted_positions, sorted_masses
            )
            delta = periodic_delta(positions, finder.xyz[index], BOX_SIZE_KPC)
            plotted = delta
            if delta.shape[0] > max_points:
                plotted = delta[rng.choice(delta.shape[0], max_points, replace=False)]
            ax = axes[row, column]
            ax.scatter(plotted[:, 0], plotted[:, 1], s=2, alpha=0.25, color=finder.color, rasterized=True)
            ax.plot(0.0, 0.0, "+", color="k", markersize=9)
            for radius, color, linestyle in (
                (finder.catalogue_radius[index], finder.color, "-"),
                (finder.r200[index], "k", "--"),
                (finder.rvmax[index], "k", ":"),
            ):
                if np.isfinite(radius) and radius > 0.0:
                    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=color, linestyle=linestyle, linewidth=1.5))
            ax.set(xlim=(-limits[row], limits[row]), ylim=(-limits[row], limits[row]))
            ax.set_aspect("equal", adjustable="box")
            if column == 0:
                ax.set_ylabel("y - y0 [kpc/h]")
            if row == 1:
                ax.set_xlabel("x - x0 [kpc/h]")
            ax.text(
                0.03, 0.97,
                f"{finder.label} {'host' if row == 0 else 'subhalo'} {int(finder.ids[index])}\n"
                f"N={positions.shape[0]}\nM200={finder.m200[index]:.2e}\n"
                f"Rcat={finder.catalogue_radius[index]:.1f}\nR200={finder.r200[index]:.1f}\n"
                f"rvmax={finder.rvmax[index]:.1f}",
                transform=ax.transAxes, va="top", fontsize=9,
            )
            profiles[(key, row)] = matched_vcirc_profile(finder, index, positions, masses)

    handles = [
        Line2D([0], [0], color="0.2", linestyle="-", label="catalogue radius"),
        Line2D([0], [0], color="0.2", linestyle="--", label="derived R200"),
        Line2D([0], [0], color="0.2", linestyle=":", label="derived rvmax"),
    ]
    axes[0, -1].legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")
    plot_matched_vcirc(finders, matches, profiles, 0, matched_output_path(output, "host_vcirc"))
    plot_matched_vcirc(finders, matches, profiles, 1, matched_output_path(output, "subhalo_vcirc"))


def main() -> None:
    """Run the comparison for every finder declared in ``finder_config.py``."""
    global SNAPSHOT_FILE, FIGURE_DIR, CATALOGUES

    parser = argparse.ArgumentParser(
        description="Make comparison plots for the configured halo finders."
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_FILE)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    for key in FINDER_KEYS:
        config = FINDERS[key]
        parser.add_argument(f"--{key}-file", type=Path, default=DERIVED_FILES[key])
        parser.add_argument(f"--{key}-catalogue", type=Path, default=COMMON_FILES[key])
        parser.add_argument(f"--{key}-label", default=config["label"])
    parser.add_argument("--target-sub-log-mass", type=float, default=11.1)
    parser.add_argument("--host-match-radius", type=float, default=300.0)
    parser.add_argument("--sub-match-radius", type=float, default=75.0)
    parser.add_argument(
        "--reference-host-id", "--ahf-host-id", dest="reference_host_id", type=int
    )
    parser.add_argument(
        "--reference-sub-id", "--ahf-sub-id", dest="reference_sub_id", type=int
    )
    parser.add_argument("--matched-max-points", type=int, default=20000)
    parser.add_argument("--matched-seed", type=int, default=42)
    parser.add_argument("--matched-output", type=Path)
    parser.add_argument("--skip-matched-members", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    SNAPSHOT_FILE = args.snapshot
    FIGURE_DIR = args.figure_dir
    data: dict[str, CosmoData] = {}
    for key in FINDER_KEYS:
        path = getattr(args, f"{key}_file")
        ids, mass, xyz, radius, rvmax, vmax = load_derived_catalogue(path)
        parent, level = assign_hosts_and_levels_periodic(xyz, mass, radius, BOX_SIZE_KPC)
        label = getattr(args, f"{key}_label")
        _, color, linestyle = finder_style(key)
        data[key] = CosmoData(
            key, label, color, linestyle, ids, mass, xyz, radius, rvmax, vmax,
            parent, level, subhalo_redefined_mass(mass, rvmax, vmax, level),
            getattr(args, f"{key}_catalogue"),
        )
    CATALOGUES = tuple((d.label, d.color, d.linestyle) for d in data.values())
    reference = data[REFERENCE_FINDER]
    host_mass_min = 1.0e13

    centre = reference.xyz[largest_index(reference.mass)]
    local = {
        key: include_indices(
            octant_local_indices(d.xyz, centre, R_LOCAL_OCTANT, BOX_SIZE_KPC),
            [largest_index(d.mass)],
        ) for key, d in data.items()
    }
    snap_xyz = load_snapshot_positions(SNAPSHOT_FILE)
    part_dxyz, part_rho = local_octant_particle_density(
        snap_xyz, centre, R_LOCAL_OCTANT, BOX_SIZE_KPC, nnei=20
    )
    for filename, radius_getter in (
        ("xy_mass_octant.png", lambda d: d.radius),
        ("xv_vmax_octant.png", lambda d: 2.0 * d.rvmax),
    ):
        fig, ax = plt.subplots(figsize=(7, 7)); underplot_particles(ax, part_dxyz, part_rho)
        for key, d in data.items():
            plot_local_circles(
                ax, d.xyz, radius_getter(d), centre, local[key], BOX_SIZE_KPC,
                linestyle=d.linestyle, color=d.color,
            )
        ax.scatter(0.0, 0.0, marker="x", s=80, linewidths=1.5, color="k")
        format_local_quadrant_axes(ax); add_simulation_linestyle_legend(ax)
        fig.tight_layout(); save_figure(fig, filename)

    vmax_masks = {key: d.vmax >= MIN_FIG4_VMAX for key, d in data.items()}
    comparison_figures = (
        (
            "Cumulative_Mass_Function.png",
            [d.mass for d in data.values()],
            [d.level for d in data.values()],
            "subhalo_mass",
            MIN_HALO_MASS,
            r"$\log_{10}(M_{200,\mathrm{derived}})$",
            r"$\Delta \log_{10}(N(>M)/V)$",
        ),
        (
            "Cumulative_Vmax_Function.png",
            [d.vmax[vmax_masks[key]] for key, d in data.items()],
            [d.level[vmax_masks[key]] for key, d in data.items()],
            "vmax",
            MIN_FIG4_VMAX,
            r"$\log_{10}(v_{\rm max}/{\rm km\,s^{-1}})$",
            r"$\Delta \log_{10}(N(>v_{\rm max})/V)$",
        ),
    )
    for (
        filename, cumulative_values, cumulative_levels, deviation_quantity,
        lower_limit, xlabel, deviation_ylabel,
    ) in comparison_figures:
        fig, axes = plt.subplots(
            nrows=4,
            figsize=(8, 12),
            sharex=True,
            gridspec_kw={"height_ratios": (2.2, 1.0, 1.0, 1.0), "hspace": 0.0},
        )
        plot_cumulative_by_level_multi(
            axes[0], cumulative_values, cumulative_levels, xlabel
        )
        axes[0].set_xlabel("")
        for ax, lev in zip(axes[1:], (0, 1, 2)):
            curves = binned_cumulative_deviation_curves(
                [
                    (d.label, getattr(d, deviation_quantity)[d.level == lev])
                    for d in data.values()
                ],
                lower_limit=lower_limit,
                volume=VOLUME,
            )
            deviation_panel_binned_multi(ax, curves)
            ax.text(
                0.98,
                0.85,
                f"Level {lev}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            ax.set_ylabel(deviation_ylabel)
        axes[-1].set_xlabel(xlabel)
        fig.align_ylabels(axes)
        fig.subplots_adjust(
            left=0.13, right=0.97, bottom=0.07, top=0.99, hspace=0.0
        )
        save_figure(fig, filename)

    for obsolete in ("Mass_deviation.png", "Vmax_deviation.png"):
        obsolete_path = FIGURE_DIR / obsolete
        if obsolete_path.exists():
            obsolete_path.unlink()

    radial = {}
    for key, d in data.items():
        counts, masssum = subhalo_radial_profiles(
            d.xyz, d.mass, d.radius, d.parent, d.level, BOX_SIZE_KPC,
            RADIAL_BINS, host_mass_min=host_mass_min, subhalo_mass=d.subhalo_mass,
        )
        rr_count, _ = subhalo_radial_data(
            d.xyz, d.mass, d.radius, d.parent, d.level, BOX_SIZE_KPC,
            host_mass_min=host_mass_min,
        )
        rr_mass, mm = subhalo_radial_data(
            d.xyz, d.mass, d.radius, d.parent, d.level, BOX_SIZE_KPC,
            host_mass_min=host_mass_min, subhalo_mass=d.subhalo_mass,
        )
        radial[key] = (
            counts, masssum, cumulative_from_sorted_radius(rr_count),
            cumulative_mass_from_sorted_radius(rr_mass, mm),
        )
    bin_centres = 0.5 * (RADIAL_BINS[:-1] + RADIAL_BINS[1:])
    for mass_plot, filename in (
        (False, "Radial_subhalo_number_counts.png"),
        (True, "Radial_subhalo_mass.png"),
    ):
        fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
        for key, d in data.items():
            differential = radial[key][1 if mass_plot else 0]
            x, y = radial[key][3 if mass_plot else 2]
            style = dict(label=d.label, color=d.color, linestyle=d.linestyle)
            axes[0].step(bin_centres, differential, where="mid", **style)
            if x.size:
                axes[1].plot(x, y, **style)
                if y[-1] > 0.0:
                    axes[2].plot(x, y / y[-1], **style)
        if mass_plot:
            axes[0].set_ylabel(r"$dM_{\rm sub}(<r_{\rm vmax})$ / bin [$M_\odot/h$]")
            axes[1].set_ylabel(r"$M_{\rm sub}(<r;\,r_{\rm vmax})$ [$M_\odot/h$]")
            axes[2].set_ylabel(r"$M_{\rm sub}(<r;\,r_{\rm vmax}) / M_{\rm sub,tot}$")
            axes[2].legend(frameon=False, loc="lower right")
        else:
            axes[0].set_ylabel(r"$dN_{\rm sub}$ / bin"); axes[0].legend(frameon=False)
            axes[1].set_ylabel(r"$N_{\rm sub}(<r)$")
            axes[2].set_ylabel(r"$N_{\rm sub}(<r) / N_{\rm sub,tot}$")
        axes[2].set_xlabel(r"$r / R_{200,{\rm host}}$"); axes[2].set_yscale("log")
        for ax in axes: ax.set_xlim(0.0, 1.0)
        fig.tight_layout(); save_figure(fig, filename)

    fig = plot_r200_rvmax_joint(
        [(d.label, d.color, d.radius, d.rvmax) for d in data.values()]
    )
    save_figure(fig, "R_vmax_vs_R200.png")

    if not args.skip_matched_members:
        finders = {
            key: build_matched_finder(
                key, d.common_path, d.ids, d.mass, d.xyz, d.radius, d.rvmax,
                d.parent, d.level,
            ) for key, d in data.items()
        }
        for key, d in data.items():
            finders[key].label, finders[key].color = d.label, d.color
        matches = select_matched_pair(
            finders, args.target_sub_log_mass, args.host_match_radius,
            args.sub_match_radius, args.reference_host_id, args.reference_sub_id,
        )
        print("Selected matched objects:")
        for key, finder in finders.items():
            host, subhalo = matches[key]
            print(f"  {finder.label}: host {int(finder.ids[host])}, subhalo {int(finder.ids[subhalo])}")
        output = args.matched_output or FIGURE_DIR / "matched_member_particles.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_matched_members(
            finders, matches, SNAPSHOT_FILE, output,
            args.matched_max_points, args.matched_seed,
        )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
