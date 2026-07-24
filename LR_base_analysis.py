#!/usr/bin/env python3
"""Derive M200, R200, rvmax, and vmax from a common halo catalogue.

The catalogue must be produced by ``convert_halo_membership.py`` and contain
the ``/Haloes`` datasets documented there. The snapshot is a single GADGET
HDF5 file containing particle coordinates, IDs, and masses (or MassTable).
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from plot_config import apply_plot_style


apply_plot_style(plt)


DEFAULT_PARTICLE_TYPE = "PartType1"
DEFAULT_HUBBLE_PARAM = 0.6777
DEFAULT_MASS_UNIT_MSUN_H = 1.0e10
DEFAULT_BOX_SIZE_KPC_H = 100000.0
DEFAULT_PLOT_RADIUS_KPC_H = 2000.0
RVMAX_EDGE_REJECT_FRACTION = 0.8
G_KPC_KMS2_PER_MSUN = 4.30091e-6


class AnalysisError(RuntimeError):
    pass


def read_common_catalogue(
    filename: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filename, "r") as handle:
        if "Haloes" not in handle:
            raise AnalysisError(f"{filename} has no /Haloes group")
        haloes = handle["Haloes"]
        required = (
            "haloid",
            "centre",
            "catalogue_mass",
            "catalogue_radius",
            "offset",
            "particle_id",
        )
        missing = [name for name in required if name not in haloes]
        if missing:
            raise AnalysisError(f"{filename} lacks /Haloes datasets: {', '.join(missing)}")

        halo_ids = np.asarray(haloes["haloid"][...], dtype=np.int64)
        centres = np.asarray(haloes["centre"][...], dtype=np.float64)
        masses = np.asarray(haloes["catalogue_mass"][...], dtype=np.float64)
        radii = np.asarray(haloes["catalogue_radius"][...], dtype=np.float64)
        offsets = np.asarray(haloes["offset"][...], dtype=np.int64)
        particle_ids = np.asarray(haloes["particle_id"][...], dtype=np.uint64)
        finder = (
            str(handle["Header"].attrs.get("finder", "catalogue"))
            if "Header" in handle
            else "catalogue"
        )

    nhalo = halo_ids.size
    if centres.shape != (nhalo, 3):
        raise AnalysisError(f"centre has shape {centres.shape}; expected ({nhalo}, 3)")
    if masses.shape != (nhalo,) or radii.shape != (nhalo,):
        raise AnalysisError("catalogue mass/radius shapes do not match haloid")
    if offsets.shape != (nhalo + 1,) or offsets[0] != 0:
        raise AnalysisError("offset must have shape [Nhalo + 1] and begin at zero")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != particle_ids.size:
        raise AnalysisError("offset does not exactly partition particle_id")
    return finder, halo_ids, centres, masses, radii, offsets, particle_ids


def load_snapshot_particles(
    filename: Path,
    particle_type: str,
    mass_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    with h5py.File(filename, "r") as handle:
        if particle_type not in handle:
            raise AnalysisError(f"{filename} has no /{particle_type} group")
        group = handle[particle_type]
        if "Coordinates" not in group or "ParticleIDs" not in group:
            raise AnalysisError(f"/{particle_type} lacks Coordinates or ParticleIDs")
        positions = np.asarray(group["Coordinates"][...], dtype=np.float64)
        particle_ids = np.asarray(group["ParticleIDs"][...], dtype=np.uint64)
        if "Masses" in group:
            masses = np.asarray(group["Masses"][...], dtype=np.float64) * mass_unit
        else:
            if "Header" not in handle or "MassTable" not in handle["Header"].attrs:
                raise AnalysisError("Snapshot has neither particle Masses nor Header/MassTable")
            index = int(particle_type.removeprefix("PartType"))
            table = np.asarray(handle["Header"].attrs["MassTable"], dtype=np.float64)
            if index >= table.size or table[index] <= 0.0:
                raise AnalysisError(f"No usable mass for {particle_type}")
            masses = np.full(particle_ids.size, table[index] * mass_unit)
        box_size = None
        if "Header" in handle and "BoxSize" in handle["Header"].attrs:
            box_size = float(handle["Header"].attrs["BoxSize"])

    if positions.shape != (particle_ids.size, 3) or masses.shape != particle_ids.shape:
        raise AnalysisError("Snapshot coordinate, mass, and ID shapes disagree")
    return positions, masses, particle_ids, box_size


def periodic_delta(xyz: np.ndarray, centre: np.ndarray, box_size: float) -> np.ndarray:
    delta = xyz - centre[None, :]
    delta -= box_size * np.round(delta / box_size)
    return delta


def enclosed_profile(
    centre: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    rho_crit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if positions.size == 0:
        return None
    radii = np.sqrt(np.sum(periodic_delta(positions, centre, box_size) ** 2, axis=1))
    order = np.argsort(radii)
    radii = radii[order]
    masses = masses[order]
    positive = radii > 0.0
    radii = radii[positive]
    masses = masses[positive]
    if radii.size == 0:
        return None
    enclosed_mass = np.cumsum(masses)
    volume = (4.0 / 3.0) * np.pi * radii**3
    overdensity = (enclosed_mass / volume) / rho_crit
    vcirc = np.sqrt(G_KPC_KMS2_PER_MSUN * enclosed_mass / radii)
    return radii, enclosed_mass, overdensity, vcirc


def halo_radius_from_profile(
    radii: np.ndarray,
    enclosed_mass: np.ndarray,
    overdensity: np.ndarray,
) -> tuple[float, float]:
    above = overdensity >= 200.0
    crossings = np.flatnonzero(above[:-1] & ~above[1:])
    if crossings.size == 0:
        return float(radii[-1]), float(enclosed_mass[-1])

    index = int(crossings[0])
    x0, x1 = np.log10(radii[index : index + 2])
    y0, y1 = np.log10(overdensity[index : index + 2])
    threshold = np.log10(200.0)
    xcross = x1 if y1 == y0 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    radius = 10.0**xcross
    m0, m1 = enclosed_mass[index : index + 2]
    mass = m1 if x1 == x0 else m0 + (xcross - x0) * (m1 - m0) / (x1 - x0)
    return float(radius), float(mass)


def rvmax_search_limit(r200: float, catalogue_radius: float) -> tuple[float, float | None]:
    limits = [value for value in (r200, catalogue_radius) if np.isfinite(value) and value > 0]
    if not limits:
        return np.nan, None
    limit = min(limits)
    catalogue_limited = (
        np.isfinite(r200)
        and r200 > 0
        and np.isfinite(catalogue_radius)
        and 0 < catalogue_radius < r200
    )
    return limit, RVMAX_EDGE_REJECT_FRACTION if catalogue_limited else None


def rvmax_from_profile(
    radii: np.ndarray,
    vcirc: np.ndarray,
    radius_limit: float,
    edge_reject_fraction: float | None,
) -> tuple[float, float, float, float]:
    use = np.flatnonzero(radii < radius_limit) if np.isfinite(radius_limit) else np.arange(radii.size)
    if use.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    radii = radii[: int(use[-1]) + 1]
    vcirc = vcirc[: radii.size]
    if radii.size == 1:
        return float(radii[0]), float(radii[0]), float(radii[0]), float(vcirc[0])

    start = min(5, radii.size - 1)
    peak = start + int(np.argmax(vcirc[start:]))
    actual_vmax = float(vcirc[peak])
    if edge_reject_fraction is not None:
        at_edge = peak == radii.size - 1 or radii[peak] >= edge_reject_fraction * radius_limit
        if at_edge:
            value = float(radii[peak])
            return value, value, value, actual_vmax

    near_peak = np.flatnonzero(vcirc[start:] > 0.97 * actual_vmax)
    if near_peak.size == 0:
        value = float(radii[peak])
        return value, value, value, actual_vmax
    low = start + int(near_peak[0])
    high = start + int(near_peak[-1])
    rlow, rhigh = float(radii[low]), float(radii[high])
    rvmax = 0.5 * (rlow + rhigh)
    return rlow, rhigh, rvmax, float(np.interp(rvmax, radii, vcirc))


def plot_vcirc(diagnostic: dict[str, object] | None, label: str) -> None:
    if diagnostic is None:
        return
    radii = diagnostic["radii"]
    vcirc = diagnostic["vcirc"]
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(radii, vcirc, linewidth=1.4)
    for key, style, name in (("r_lo", "--", "Min r with Vc > 0.97 vmax"), ("r_hi", ":", "Max r with Vc > 0.97 vmax"), ("rvmax", "-.", "rvmax")):
        value = float(diagnostic[key])
        if np.isfinite(value):
            axis.axvline(value, linestyle=style, linewidth=1.2, label=name)
    axis.set(xlabel="r [kpc/h]", ylabel=r"$V_{\rm circ}$ [km s$^{-1}$]")
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    plt.show(block=False)


def plot_local_halos(
    centres: np.ndarray,
    catalogue_radii: np.ndarray,
    calculated_radii: np.ndarray,
    centre: np.ndarray,
    box_size: float,
    plot_radius: float,
    label: str,
) -> None:
    delta = periodic_delta(centres, centre, box_size)
    selected = np.sqrt(np.sum(delta**2, axis=1)) <= plot_radius
    fig, axis = plt.subplots(figsize=(8, 8))
    for x, y, radius in zip(delta[selected, 0], delta[selected, 1], catalogue_radii[selected]):
        axis.add_patch(plt.Circle((x, y), radius, fill=False, color="C0", linewidth=1.2))
    for x, y, radius in zip(delta[selected, 0], delta[selected, 1], calculated_radii[selected]):
        if np.isfinite(radius) and radius > 0:
            axis.add_patch(plt.Circle((x, y), radius, fill=False, color="C3", linestyle="--"))
    axis.set(xlim=(-plot_radius, plot_radius), ylim=(-plot_radius, plot_radius), xlabel="x - x0 [kpc/h]", ylabel="y - y0 [kpc/h]")
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show(block=False)


def write_derived_catalogue(
    filename: Path,
    halo_ids: np.ndarray,
    centres: np.ndarray,
    masses: np.ndarray,
    radii: np.ndarray,
    rvmax: np.ndarray,
    vmax: np.ndarray,
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("wt", encoding="utf-8") as stream:
        stream.write("# id x y z M200derived R200derived rvmax vmax_kms\n")
        for haloid, centre, mass, radius, peak_radius, peak_velocity in zip(
            halo_ids, centres, masses, radii, rvmax, vmax
        ):
            stream.write(
                f"{int(haloid)} {centre[0]:.8e} {centre[1]:.8e} {centre[2]:.8e} "
                f"{mass:.8e} {radius:.8e} {peak_radius:.8e} {peak_velocity:.8e}\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute particle-derived halo properties from a common HDF5 catalogue."
    )
    parser.add_argument("catalogue", type=Path, help="Output from convert_halo_membership.py")
    parser.add_argument("snapshot", type=Path, help="GADGET HDF5 snapshot")
    parser.add_argument("-o", "--output", type=Path, help="Derived text catalogue")
    parser.add_argument("--particle-type", default=DEFAULT_PARTICLE_TYPE)
    parser.add_argument("--box-size", type=float, help="Periodic box size in kpc/h")
    parser.add_argument("--hubble-param", type=float, default=DEFAULT_HUBBLE_PARAM)
    parser.add_argument("--mass-unit", type=float, default=DEFAULT_MASS_UNIT_MSUN_H)
    parser.add_argument("--plot-radius", type=float, default=DEFAULT_PLOT_RADIUS_KPC_H)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        finder, halo_ids, centres, catalogue_masses, catalogue_radii, offsets, member_ids = read_common_catalogue(args.catalogue)
        positions, particle_masses, snapshot_ids, snapshot_box_size = load_snapshot_particles(
            args.snapshot, args.particle_type, args.mass_unit
        )
        box_size = args.box_size if args.box_size is not None else snapshot_box_size
        if box_size is None:
            box_size = DEFAULT_BOX_SIZE_KPC_H

        order = np.argsort(snapshot_ids)
        sorted_ids = snapshot_ids[order]
        sorted_positions = positions[order]
        sorted_masses = particle_masses[order]
        rho_crit = 2.77536627e11 * args.hubble_param**2 / 1.0e9 / args.hubble_param**2

        nhalo = halo_ids.size
        calculated_radii = np.full(nhalo, np.nan)
        calculated_masses = np.full(nhalo, np.nan)
        rvmax = np.full(nhalo, np.nan)
        vmax = np.full(nhalo, np.nan)
        mass_order = np.argsort(catalogue_masses)[::-1]
        diagnostic_index = int(
            mass_order[1] if mass_order.size > 1 else mass_order[0]
        )
        diagnostic: dict[str, object] | None = None

        for index, haloid in enumerate(halo_ids):
            ids = member_ids[offsets[index] : offsets[index + 1]]
            locations = np.searchsorted(sorted_ids, ids)
            valid = locations < sorted_ids.size
            matched = np.zeros(ids.size, dtype=bool)
            matched[valid] = sorted_ids[locations[valid]] == ids[valid]
            locations = locations[matched]
            profile = enclosed_profile(
                centres[index], sorted_positions[locations], sorted_masses[locations], box_size, rho_crit
            )
            if profile is None:
                continue
            radii, enclosed_mass, overdensity, vcirc = profile
            calculated_radii[index], calculated_masses[index] = halo_radius_from_profile(
                radii, enclosed_mass, overdensity
            )
            limit, edge_reject = rvmax_search_limit(
                calculated_radii[index], catalogue_radii[index]
            )
            rlow, rhigh, rvmax[index], vmax[index] = rvmax_from_profile(
                radii, vcirc, limit, edge_reject
            )
            if not args.no_plots and index == diagnostic_index:
                diagnostic = {
                    "id": int(haloid), "radii": radii, "vcirc": vcirc,
                    "r_lo": rlow, "r_hi": rhigh, "rvmax": rvmax[index],
                }

        output = args.output or Path(f"Simulation/{finder}_halos")
        write_derived_catalogue(
            output, halo_ids, centres, calculated_masses, calculated_radii, rvmax, vmax
        )
        print(f"Wrote derived halo file: {output}")

        largest = int(np.argmax(catalogue_masses))
        print(
            f"Largest {finder} halo: id={int(halo_ids[largest])} "
            f"catalogue_mass={catalogue_masses[largest]:.6e} "
            f"members={int(offsets[largest + 1] - offsets[largest])}"
        )
        if not args.no_plots:
            plot_vcirc(diagnostic, finder.upper())
            plot_local_halos(
                centres, catalogue_radii, calculated_radii, centres[largest],
                box_size, args.plot_radius, finder.upper(),
            )
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        build_parser().exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
