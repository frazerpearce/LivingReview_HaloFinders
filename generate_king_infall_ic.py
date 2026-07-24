#!/usr/bin/env python3
"""Generate a controlled King-halo radial-infall recovery test.

The box contains one 100,000-particle host, isolated 1:10 and 1:100 controls,
and copies of both satellites at several host-centric radii.  All particles
have the same mass.  Satellite length scales vary as M^(1/3), preserving the
host's characteristic density, while their Jeans velocities are calculated
from their own scaled mass profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import generate_unit_test_king_ic as king
from plot_config import apply_plot_style


apply_plot_style(plt)

HOST_CENTRE_KPC_H = np.array([5000.0, 5000.0, 5000.0])
DEFAULT_RADIAL_FRACTIONS = (0.5, 1.0, 2.0, 3.0)
RATIOS = (0.1, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Simulation/data/king_infall.hdf5"),
    )
    parser.add_argument("--binary-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--n-total", type=int, default=1_000_000)
    parser.add_argument("--n-host", type=int, default=100_000)
    parser.add_argument("--box-size-kpc", type=float, default=10_000.0)
    parser.add_argument("--host-core-kpc", type=float, default=60.0)
    parser.add_argument("--host-trunc-kpc", type=float, default=1000.0)
    parser.add_argument("--translation-kms", type=float, default=4000.0)
    parser.add_argument("--jitter-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument(
        "--radial-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_RADIAL_FRACTIONS,
        help="Embedded satellite centre radii in units of host R200.",
    )
    parser.add_argument(
        "--control-gap-kpc",
        type=float,
        default=100.0,
        help="Gap between the host and isolated-control truncation surfaces.",
    )
    parser.add_argument(
        "--small-control-extra-kpc",
        type=float,
        default=500.0,
        help="Additional host-centric distance for the isolated 1:100 control.",
    )
    parser.add_argument("--no-binary", action="store_true")
    return parser.parse_args()


def r200_for_king(
    mass: float, core: float, trunc: float
) -> float:
    grid = np.geomspace(max(core * 1.0e-5, 1.0e-6), trunc, 200_000)
    enclosed = mass * king.king_mass_shape(grid / core)
    enclosed /= king.king_mass_shape(np.array(trunc / core))
    mean_density = enclosed / ((4.0 * np.pi / 3.0) * grid**3)
    target = 200.0 * king.CRITICAL_DENSITY_MSUN_H_PER_MPC_H3 / 1.0e9
    crossing = np.flatnonzero(mean_density <= target)
    if crossing.size == 0:
        raise ValueError("King profile does not cross 200 times critical density")
    i = int(crossing[0])
    if i == 0:
        return float(grid[0])
    x = np.interp(
        np.log(target),
        np.log(mean_density[i - 1 : i + 1])[::-1],
        np.log(grid[i - 1 : i + 1])[::-1],
    )
    return float(np.exp(x))


def choose_nonoverlapping_centres(
    host_centre: np.ndarray,
    box_size: float,
    requested: list[tuple[str, float, float]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Choose directions so satellite R200 apertures do not overlap."""
    centres: list[np.ndarray] = []
    placed_radii: list[float] = []
    for label, distance, satellite_r200 in requested:
        for _ in range(100_000):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            candidate = (host_centre + distance * direction) % box_size
            valid = True
            for other, other_r200 in zip(centres, placed_radii):
                delta = (candidate - other + 0.5 * box_size) % box_size
                delta -= 0.5 * box_size
                if np.linalg.norm(delta) <= satellite_r200 + other_r200:
                    valid = False
                    break
            if valid:
                centres.append(candidate)
                placed_radii.append(satellite_r200)
                break
        else:
            raise RuntimeError(f"Could not place non-overlapping satellite {label}")
    return centres


def write_hdf5(
    path: Path,
    coordinates: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
    labels: list[str],
    ratios: np.ndarray,
    radial_fractions: np.ndarray,
    centres: np.ndarray,
    translations: np.ndarray,
    relative_radial_speeds: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    core_radii: np.ndarray,
    trunc_radii: np.ndarray,
    r200: np.ndarray,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = box_size
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        npart = np.zeros(6, dtype=np.uint32)
        npart[1] = coordinates.shape[0]
        header.attrs["NumPart_ThisFile"] = npart
        header.attrs["NumPart_Total"] = npart
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 1.0
        header.attrs["Redshift"] = 0.0
        header.attrs["Omega0"] = king.OMEGA_M
        header.attrs["OmegaLambda"] = king.OMEGA_LAMBDA
        header.attrs["HubbleParam"] = king.HUBBLE_PARAM
        for name in (
            "Flag_Cooling", "Flag_DoublePrecision", "Flag_Feedback",
            "Flag_Metals", "Flag_Sfr", "Flag_StellarAge",
        ):
            header.attrs[name] = 0

        params = handle.create_group("Parameters")
        params.attrs["BoxSize"] = box_size
        params.attrs["ComovingIntegrationOn"] = 1
        params.attrs["HubbleParam"] = king.HUBBLE_PARAM
        params.attrs["Omega0"] = king.OMEGA_M
        params.attrs["OmegaBaryon"] = king.OMEGA_BARYON
        params.attrs["OmegaLambda"] = king.OMEGA_LAMBDA
        params.attrs["TimeBegin"] = 1.0
        params.attrs["TimeMax"] = 1.0
        params.attrs["ICFormat"] = 3
        params.attrs["SnapFormat"] = 3
        params.attrs["UnitLength_in_cm"] = 3.085678e21
        params.attrs["UnitMass_in_g"] = 1.989e43
        params.attrs["UnitVelocity_in_cm_per_s"] = 1.0e5

        config = handle.create_group("Config")
        config.attrs["ICGenerator"] = b"generate_king_infall_ic.py"
        config.attrs["Seed"] = seed
        config.attrs["RNGStreamScheme"] = b"SeedSequence-spawn-component-v1"
        config.attrs["HostCentreKpc"] = HOST_CENTRE_KPC_H
        config.create_dataset("ObjectLabel", data=np.asarray(labels, dtype="S32"))
        config.create_dataset("MassRatio", data=ratios)
        config.create_dataset("HostRadialFraction", data=radial_fractions)
        config.create_dataset("CentreKpc", data=centres)
        config.create_dataset("TranslationKms", data=translations)
        config.create_dataset(
            "HostRelativeRadialSpeedKms", data=relative_radial_speeds
        )
        config.create_dataset("ParticleStart", data=starts)
        config.create_dataset("ParticleCount", data=counts)
        config.create_dataset("CoreRadiusKpc", data=core_radii)
        config.create_dataset("TruncRadiusKpc", data=trunc_radii)
        config.create_dataset("R200Kpc", data=r200)

        particles = handle.create_group("PartType1")
        particles.create_dataset(
            "Coordinates", data=coordinates.astype(np.float32),
            compression="gzip", shuffle=True,
        )
        particles.create_dataset(
            "Velocities", data=velocities.astype(np.float32),
            compression="gzip", shuffle=True,
        )
        particles.create_dataset(
            "Masses", data=masses.astype(np.float32),
            compression="gzip", shuffle=True,
        )
        particles.create_dataset(
            "ParticleIDs", data=particle_ids.astype(np.uint32),
            compression="gzip", shuffle=True,
        )


def quicklook(
    path: Path,
    coordinates: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    centres: np.ndarray,
    r200: np.ndarray,
    labels: list[str],
    box_size: float,
    rng: np.random.Generator,
) -> None:
    background_end = int(starts[0])
    n_background_plot = min(40_000, background_end)
    background = rng.choice(background_end, n_background_plot, replace=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        coordinates[background, 0] / 1000.0,
        coordinates[background, 1] / 1000.0,
        s=0.15, color="0.55", alpha=0.35, rasterized=True,
    )
    ratio_colours = {1.0: "C0", 0.1: "C1", 0.01: "C2"}
    for start, count, centre, radius, label in zip(
        starts, counts, centres, r200, labels
    ):
        ratio = (
            1.0 if label == "host"
            else 0.01 if "0.01" in label
            else 0.1
        )
        colour = ratio_colours[ratio]
        ids = np.arange(int(start), int(start + count))
        if ids.size > 20_000:
            ids = rng.choice(ids, 20_000, replace=False)
        ax.scatter(
            coordinates[ids, 0] / 1000.0,
            coordinates[ids, 1] / 1000.0,
            s=0.35, color=colour, alpha=0.55, rasterized=True,
        )
        ax.add_patch(
            plt.Circle(
                centre[:2] / 1000.0, radius / 1000.0,
                fill=False, color=colour, linewidth=1.0,
                linestyle="-" if label.startswith("control") or label == "host" else "--",
            )
        )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color="C0", label="Host"),
        Line2D([], [], marker="o", linestyle="none", color="C1", label="1:10"),
        Line2D([], [], marker="o", linestyle="none", color="C2", label="1:100"),
        Line2D([], [], color="0.25", linestyle="-", label="Isolated $R_{200}$"),
        Line2D([], [], color="0.25", linestyle="--", label="Embedded $R_{200}$"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.set(
        xlim=(0, box_size / 1000.0),
        ylim=(0, box_size / 1000.0),
        xlabel=r"$x\ [h^{-1}{\rm Mpc}]$",
        ylabel=r"$y\ [h^{-1}{\rm Mpc}]$",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_host <= 0 or args.n_total <= args.n_host:
        raise ValueError("Require 0 < --n-host < --n-total")
    if any(f <= 0.0 for f in args.radial_fractions):
        raise ValueError("--radial-fractions must be positive")
    if not 0.0 <= args.jitter_fraction <= 0.5:
        raise ValueError("--jitter-fraction must be between 0 and 0.5")
    if args.control_gap_kpc < 0.0 or args.small_control_extra_kpc < 0.0:
        raise ValueError("Control gaps must be non-negative")

    satellite_counts = {
        ratio: int(round(args.n_host * ratio)) for ratio in RATIOS
    }
    if any(count < 20 for count in satellite_counts.values()):
        raise ValueError("Satellite particle counts must be at least 20")

    particle_mass = king.cosmological_particle_mass(
        args.box_size_kpc, args.n_total
    )
    host_mass = particle_mass * args.n_host
    host_r200 = r200_for_king(
        host_mass, args.host_core_kpc, args.host_trunc_kpc
    )

    labels = ["host"]
    ratios = [1.0]
    radial_fractions = [0.0]
    counts = [args.n_host]
    core_radii = [args.host_core_kpc]
    trunc_radii = [args.host_trunc_kpc]
    object_r200 = [host_r200]
    placement_requests: list[tuple[str, float, float]] = []

    for ratio in RATIOS:
        scale = ratio ** (1.0 / 3.0)
        sat_r200 = host_r200 * scale
        label = f"control_{ratio:g}"
        distance = (
            args.host_trunc_kpc
            + args.host_trunc_kpc * scale
            + args.control_gap_kpc
        )
        if ratio == 0.01:
            distance += args.small_control_extra_kpc
        placement_requests.append((label, distance, sat_r200))
        labels.append(label)
        ratios.append(ratio)
        radial_fractions.append(np.nan)
        counts.append(satellite_counts[ratio])
        core_radii.append(args.host_core_kpc * scale)
        trunc_radii.append(args.host_trunc_kpc * scale)
        object_r200.append(sat_r200)

    for fraction in args.radial_fractions:
        for ratio in RATIOS:
            scale = ratio ** (1.0 / 3.0)
            label = f"sat_{ratio:g}_r{fraction:g}"
            sat_r200 = host_r200 * scale
            placement_requests.append(
                (label, fraction * host_r200, sat_r200)
            )
            labels.append(label)
            ratios.append(ratio)
            radial_fractions.append(fraction)
            counts.append(satellite_counts[ratio])
            core_radii.append(args.host_core_kpc * scale)
            trunc_radii.append(args.host_trunc_kpc * scale)
            object_r200.append(sat_r200)

    total_injected = int(np.sum(counts))
    if total_injected >= args.n_total:
        raise ValueError("Injected objects leave no background particles")

    n_objects = len(labels)
    # The host and each mass ratio receive their own phase-space template.
    # Reinitializing a ratio's template streams for every copy makes the
    # isolated and embedded realizations exactly identical apart from centre.
    sequences = np.random.SeedSequence(args.seed).spawn(13)
    background_position_rng = np.random.default_rng(sequences[0])
    background_selection_rng = np.random.default_rng(sequences[1])
    placement_rng = np.random.default_rng(sequences[2])
    plot_rng = np.random.default_rng(sequences[3])

    satellite_centres = choose_nonoverlapping_centres(
        HOST_CENTRE_KPC_H,
        args.box_size_kpc,
        placement_requests,
        placement_rng,
    )
    centres = np.vstack([HOST_CENTRE_KPC_H, *satellite_centres])

    background = king.quiet_grid_positions(
        args.n_total,
        args.box_size_kpc,
        args.jitter_fraction,
        background_position_rng,
    )
    remove = np.zeros(args.n_total, dtype=bool)
    remove[
        background_selection_rng.choice(
            args.n_total, total_injected, replace=False
        )
    ] = True
    background = background[~remove]

    host_streams = sequences[4:7]
    ratio_streams = {
        0.1: sequences[7:10],
        0.01: sequences[10:13],
    }
    object_positions: list[np.ndarray] = []
    object_velocities: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    relative_radial_speeds: list[float] = []
    for i, (centre, count, ratio, core, trunc) in enumerate(
        zip(centres, counts, ratios, core_radii, trunc_radii)
    ):
        template_streams = host_streams if ratio == 1.0 else ratio_streams[ratio]
        position_rng = np.random.default_rng(template_streams[0])
        velocity_rng = np.random.default_rng(template_streams[1])
        translation_rng = np.random.default_rng(template_streams[2])
        pos, vel, _, translation = king.build_king_sphere(
            count,
            host_mass * ratio,
            core,
            trunc,
            args.box_size_kpc,
            centre,
            args.translation_kms if ratio == 1.0 else 0.0,
            "jeans_isotropic",
            position_rng,
            velocity_rng,
            translation_rng,
        )
        if ratio == 1.0:
            relative_radial_speeds.append(0.0)
        else:
            delta = (
                centre - HOST_CENTRE_KPC_H + 0.5 * args.box_size_kpc
            ) % args.box_size_kpc - 0.5 * args.box_size_kpc
            distance = float(np.linalg.norm(delta))
            if distance <= 0.0:
                raise ValueError("Satellite centre coincides with host centre")
            infall_speed = float(
                king.king_escape_speed(
                    np.asarray([distance]),
                    host_mass,
                    args.host_core_kpc,
                    args.host_trunc_kpc,
                )[0]
            )
            translation = (
                translations[0] - infall_speed * delta / distance
            )
            vel += translation[None, :]
            relative_radial_speeds.append(infall_speed)
        object_positions.append(pos)
        object_velocities.append(vel)
        translations.append(translation)

    starts = background.shape[0] + np.concatenate(
        ([0], np.cumsum(counts[:-1], dtype=np.int64))
    )
    coordinates = np.vstack([background, *object_positions])
    velocities = np.vstack(
        [np.zeros_like(background), *object_velocities]
    )
    masses = np.full(
        args.n_total,
        particle_mass / king.PARTICLE_MASS_UNIT_MSUN,
        dtype=np.float64,
    )
    particle_ids = np.arange(args.n_total, dtype=np.uint32)

    write_hdf5(
        args.output,
        coordinates,
        velocities,
        masses,
        particle_ids,
        args.box_size_kpc,
        labels,
        np.asarray(ratios),
        np.asarray(radial_fractions),
        centres,
        np.asarray(translations),
        np.asarray(relative_radial_speeds),
        starts,
        np.asarray(counts),
        np.asarray(core_radii),
        np.asarray(trunc_radii),
        np.asarray(object_r200),
        args.seed,
    )

    binary_output = (
        args.binary_output
        if args.binary_output is not None
        else args.output.with_suffix("")
    )
    if not args.no_binary:
        king.write_gadget_binary_snapshot(
            binary_output,
            coordinates,
            velocities,
            masses,
            particle_ids,
            args.box_size_kpc,
        )
    plot_output = (
        args.plot_output
        if args.plot_output is not None
        else args.output.with_suffix(".png")
    )
    quicklook(
        plot_output,
        coordinates,
        starts,
        np.asarray(counts),
        centres,
        np.asarray(object_r200),
        labels,
        args.box_size_kpc,
        plot_rng,
    )

    print(f"Wrote {args.output}")
    if not args.no_binary:
        print(f"Wrote {binary_output}")
    print(f"Wrote {plot_output}")
    print(
        f"particle_mass={particle_mass:.8e} Msun/h "
        f"host_mass={host_mass:.8e} Msun/h host_R200={host_r200:.6f} kpc/h"
    )
    for label, ratio, fraction, count, centre, radius, infall_speed in zip(
        labels, ratios, radial_fractions, counts, centres, object_r200,
        relative_radial_speeds,
    ):
        print(
            f"{label}: ratio={ratio:g} radial_fraction={fraction:g} "
            f"count={count} R200={radius:.6f} "
            f"host_relative_radial_speed={infall_speed:.6f} km/s "
            f"centre=({centre[0]:.6f},{centre[1]:.6f},{centre[2]:.6f})"
        )


if __name__ == "__main__":
    main()
