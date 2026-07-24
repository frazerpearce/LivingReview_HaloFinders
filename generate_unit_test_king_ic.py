#!/usr/bin/env python3
"""
Generate a unit-test dark-matter IC snapshot for LR_halo_radius.py.

The output is a Gadget-like HDF5 snapshot with:
- PartType1/Coordinates in kpc/h-like code length units
- PartType1/Velocities in km/s
- PartType1/ParticleIDs
- PartType1/Masses in the same 1e10 Msun/h code units used by LR_halo_radius.py

Default setup:
- 10 Mpc/h periodic box, stored as 10000 kpc/h
- 1,000,000 final DM particles
- a quiet one-particle-per-cell background grid with small random jitter
- two truncated King-like halos with rho proportional to (1 + (r/rc)^2)^(-3/2)
- isotropic Jeans-equilibrium halo particle velocities plus a 4000 km/s
  translational velocity

The final particle count stays fixed: the script removes N_king particles per
King sphere from the quiet background and replaces them with King particles.
"""

from pathlib import Path
import argparse
import struct

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from plot_config import apply_plot_style


apply_plot_style(plt)


PARTICLE_MASS_UNIT_MSUN = 1.0e10
CRITICAL_DENSITY_MSUN_H_PER_MPC_H3 = 2.77519737e11
G_KPC_KMS2_PER_MSUN = 4.30091e-6
OMEGA_M = 0.307115
OMEGA_LAMBDA = 0.692885
OMEGA_BARYON = 0.048206
HUBBLE_PARAM = 0.6777
KING_CENTRES_KPC = np.array([
    [0.0, 0.0, 0.0],
    [5000.0, 5000.0, 5000.0],
], dtype=np.float64)
BOX_ROLL_KPC = np.array([5000.0, 5000.0, 5000.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a quiet-box plus moving King-sphere DM-only HDF5 snapshot."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("Simulation/data/unit_test_king_box.hdf5"),
        help="Output HDF5 snapshot path.",
    )
    parser.add_argument(
        "--binary-output",
        type=Path,
        default=None,
        help="Optional single-file old-style Gadget binary output path. Defaults to output path without .hdf5.",
    )
    parser.add_argument(
        "--no-binary",
        action="store_true",
        help="Only write the HDF5 snapshot, not the companion Gadget binary.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional x-y quick-look PNG path. Defaults to output path with .png suffix.",
    )
    parser.add_argument(
        "--plot-fraction",
        type=float,
        default=0.1,
        help="Fraction of particles to show in the quick-look plot.",
    )
    parser.add_argument("--n-total", type=int, default=1_000_000, help="Final total particle count.")
    parser.add_argument("--n-king", type=int, default=100_000, help="Number of particles per King sphere.")
    parser.add_argument("--box-size-kpc", type=float, default=10_000.0, help="Periodic box size in kpc/h.")
    parser.add_argument(
        "--king-mass-msun",
        type=float,
        default=None,
        help=(
            "Deprecated target King-sphere mass in Msun/h. The unit-test particle mass is "
            "set by Omega_m rho_crit box volume / n_total so Rockstar's FOF linking length "
            "has the intended cosmological-density normalization."
        ),
    )
    parser.add_argument("--king-core-kpc", type=float, default=60.0, help="King core radius in kpc/h.")
    parser.add_argument("--king-trunc-kpc", type=float, default=1000.0, help="King truncation radius in kpc/h.")
    parser.add_argument(
        "--jitter-fraction",
        type=float,
        default=0.5,
        help="Uniform background jitter as a fraction of one quiet-grid cell width.",
    )
    parser.add_argument("--translation-kms", type=float, default=4000.0, help="King centre-of-mass speed.")
    parser.add_argument(
        "--vcirc",
        action="store_true",
        help="Use the old circular-orbit velocity prescription instead of the default isotropic Jeans velocities.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    return parser.parse_args()


def require_valid_args(args: argparse.Namespace) -> None:
    n_side = round(args.n_total ** (1.0 / 3.0))
    if n_side ** 3 != args.n_total:
        raise ValueError("--n-total must be a perfect cube for the quiet grid")
    n_king_total = args.n_king * KING_CENTRES_KPC.shape[0]
    if args.n_king <= 0 or n_king_total >= args.n_total:
        raise ValueError("--n-king must be positive and leave at least one background particle")
    if args.box_size_kpc <= 0.0:
        raise ValueError("--box-size-kpc must be positive")
    if args.king_mass_msun is not None and args.king_mass_msun <= 0.0:
        raise ValueError("--king-mass-msun must be positive")
    if args.king_core_kpc <= 0.0 or args.king_trunc_kpc <= args.king_core_kpc:
        raise ValueError("--king-trunc-kpc must be larger than --king-core-kpc")
    if not (0.0 <= args.jitter_fraction <= 0.5):
        raise ValueError("--jitter-fraction must satisfy 0 <= jitter <= 0.5")
    if not (0.0 < args.plot_fraction <= 1.0):
        raise ValueError("--plot-fraction must satisfy 0 < fraction <= 1")


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    vec = rng.normal(size=(n, 3))
    norm = np.linalg.norm(vec, axis=1)
    while np.any(norm == 0.0):
        bad = norm == 0.0
        vec[bad] = rng.normal(size=(np.count_nonzero(bad), 3))
        norm = np.linalg.norm(vec, axis=1)
    return vec / norm[:, None]


def quiet_grid_positions(
    n_total: int,
    box_size: float,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_side = round(n_total ** (1.0 / 3.0))
    cell = box_size / n_side
    axis = (np.arange(n_side, dtype=np.float64) + 0.5) * cell
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    if jitter_fraction > 0.0:
        jitter = rng.uniform(-jitter_fraction, jitter_fraction, size=pos.shape) * cell
        pos = (pos + jitter) % box_size
    return pos


def king_mass_shape(x: np.ndarray) -> np.ndarray:
    return np.arcsinh(x) - x / np.sqrt(1.0 + x * x)


def cosmological_particle_mass(box_size_kpc: float, n_total: int) -> float:
    box_size_mpc = box_size_kpc / 1000.0
    total_mass = OMEGA_M * CRITICAL_DENSITY_MSUN_H_PER_MPC_H3 * box_size_mpc**3
    return total_mass / n_total


def sample_king_radii(
    n: int,
    core_radius: float,
    trunc_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x_trunc = trunc_radius / core_radius
    x_grid = np.linspace(0.0, x_trunc, 200_000, dtype=np.float64)
    cdf = king_mass_shape(x_grid)
    cdf /= cdf[-1]
    u = rng.random(n)
    x = np.interp(u, cdf, x_grid)
    return core_radius * x


def perpendicular_unit_vectors(rhat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    trial = rng.normal(size=rhat.shape)
    tangential = np.cross(rhat, trial)
    norm = np.linalg.norm(tangential, axis=1)
    bad = norm == 0.0
    while np.any(bad):
        trial[bad] = rng.normal(size=(np.count_nonzero(bad), 3))
        tangential[bad] = np.cross(rhat[bad], trial[bad])
        norm = np.linalg.norm(tangential, axis=1)
        bad = norm == 0.0
    return tangential / norm[:, None]


def king_jeans_sigma_r(
    radii: np.ndarray,
    total_mass: float,
    core_radius: float,
    trunc_radius: float,
) -> np.ndarray:
    x_trunc = trunc_radius / core_radius
    x_grid = np.linspace(0.0, x_trunc, 200_000, dtype=np.float64)
    r_grid = core_radius * x_grid
    rho_shape = (1.0 + x_grid * x_grid) ** -1.5
    mass_fraction = king_mass_shape(x_grid) / king_mass_shape(np.array(x_trunc))
    enclosed_mass = total_mass * mass_fraction

    integrand = np.zeros_like(x_grid)
    positive = r_grid > 0.0
    integrand[positive] = rho_shape[positive] * G_KPC_KMS2_PER_MSUN * enclosed_mass[positive] / r_grid[positive] ** 2

    # Isotropic, spherical Jeans solution with zero pressure at the truncation radius:
    # sigma_r^2(r) = rho(r)^-1 integral_r^R rho(s) G M(<s) / s^2 ds.
    rev_integral = np.zeros_like(x_grid)
    dr = np.diff(r_grid)
    trapezoids = 0.5 * (integrand[:-1] + integrand[1:]) * dr
    rev_integral[:-1] = np.cumsum(trapezoids[::-1])[::-1]
    sigma2_grid = rev_integral / rho_shape
    sigma2 = np.interp(radii, r_grid, sigma2_grid)
    return np.sqrt(np.maximum(sigma2, 0.0))


def king_escape_speed(
    radii: np.ndarray,
    total_mass: float,
    core_radius: float,
    trunc_radius: float,
) -> np.ndarray:
    radii = np.asarray(radii, dtype=np.float64)
    x = np.clip(radii / core_radius, 0.0, trunc_radius / core_radius)
    x_trunc = trunc_radius / core_radius
    norm = king_mass_shape(np.array(x_trunc))
    enclosed_mass = total_mass * king_mass_shape(x) / norm

    interior = np.zeros_like(radii)
    positive = radii > 0.0
    interior[positive] = enclosed_mass[positive] / radii[positive]

    shell = total_mass / (core_radius * norm) * (
        1.0 / np.sqrt(1.0 + x * x) - 1.0 / np.sqrt(1.0 + x_trunc * x_trunc)
    )
    return np.sqrt(np.maximum(2.0 * G_KPC_KMS2_PER_MSUN * (interior + shell), 0.0))


def sample_isotropic_jeans_velocities(
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.normal(scale=sigma[:, None], size=(sigma.size, 3))


def sample_bound_isotropic_jeans_velocities(
    sigma: np.ndarray,
    v_escape: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Draw the local Jeans Gaussian, rejecting every unbound velocity."""
    velocities = sample_isotropic_jeans_velocities(sigma, rng)
    redraw_count = 0
    for _ in range(10_000):
        # Keep each realised sphere at rest before adding its bulk translation.
        # Recentring can change the boundness test, so it is repeated after
        # every replacement draw.
        velocities -= velocities.mean(axis=0)
        speed = np.linalg.norm(velocities, axis=1)
        unbound = speed >= v_escape
        if not np.any(unbound):
            return velocities, redraw_count
        count = int(np.count_nonzero(unbound))
        velocities[unbound] = sample_isotropic_jeans_velocities(
            sigma[unbound], rng
        )
        redraw_count += count
    raise RuntimeError("King velocity rejection sampling did not converge")


def build_king_sphere(
    n_king: int,
    total_mass: float,
    core_radius: float,
    trunc_radius: float,
    box_size: float,
    centre: np.ndarray,
    translation_speed: float,
    velocity_model: str,
    position_rng: np.random.Generator,
    velocity_rng: np.random.Generator,
    translation_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = sample_king_radii(n_king, core_radius, trunc_radius, position_rng)
    rhat = random_unit_vectors(position_rng, n_king)
    rel_pos = radii[:, None] * rhat
    rel_pos -= rel_pos.mean(axis=0)
    bound_radii = np.linalg.norm(rel_pos, axis=1)

    x = radii / core_radius
    x_trunc = trunc_radius / core_radius
    mass_fraction = king_mass_shape(x) / king_mass_shape(np.array(x_trunc))
    enclosed_mass = total_mass * mass_fraction
    vcirc = np.zeros(n_king, dtype=np.float64)
    positive = radii > 0.0
    vcirc[positive] = np.sqrt(G_KPC_KMS2_PER_MSUN * enclosed_mass[positive] / radii[positive])

    if velocity_model == "vcirc":
        tangential = perpendicular_unit_vectors(rhat, velocity_rng)
        internal_vel = tangential * vcirc[:, None]
        internal_vel -= internal_vel.mean(axis=0)
    elif velocity_model == "jeans_isotropic":
        sigma_r = king_jeans_sigma_r(bound_radii, total_mass, core_radius, trunc_radius)
        internal_vel, redraw_count = sample_bound_isotropic_jeans_velocities(
            sigma_r,
            king_escape_speed(
                bound_radii, total_mass, core_radius, trunc_radius
            ),
            velocity_rng,
        )
        print(f"King velocity rejection redraws: {redraw_count}")
    else:
        raise ValueError(f"Unknown velocity model: {velocity_model}")

    translation_dir = random_unit_vectors(translation_rng, 1)[0]
    translation = translation_speed * translation_dir

    centre = np.asarray(centre, dtype=np.float64) % box_size
    pos = (centre[None, :] + rel_pos) % box_size
    vel = internal_vel + translation[None, :]
    return pos, vel, centre, translation


def assert_bound_king_particles(
    coordinates: np.ndarray,
    velocities: np.ndarray,
    king_start_indices: np.ndarray,
    king_centres: np.ndarray,
    king_translations: np.ndarray,
    n_king: int,
    total_mass: float,
    core_radius: float,
    trunc_radius: float,
    box_size: float,
) -> None:
    for i, start in enumerate(king_start_indices):
        sl = slice(int(start), int(start + n_king))
        rel = (coordinates[sl] - king_centres[i] + 0.5 * box_size) % box_size - 0.5 * box_size
        radii = np.linalg.norm(rel, axis=1)
        internal = velocities[sl] - king_translations[i]
        speed = np.linalg.norm(internal, axis=1)
        v_escape = king_escape_speed(radii, total_mass, core_radius, trunc_radius)
        unbound = speed >= v_escape
        if np.any(unbound):
            worst = int(np.argmax(speed / np.maximum(v_escape, 1.0e-30)))
            raise ValueError(
                f"King {i} has {np.count_nonzero(unbound)} unbound particles before writing; "
                f"worst speed/vesc={speed[worst] / v_escape[worst]:.6g}"
            )


def write_snapshot(
    path: Path,
    coordinates: np.ndarray,
    velocities: np.ndarray,
    masses_code: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
    king_start_indices: np.ndarray,
    king_centres: np.ndarray,
    king_velocities: np.ndarray,
    king_mass_msun: float,
    total_box_mass_msun: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_total = particle_ids.size
    npart = np.array([0, n_total, 0, 0, 0, 0], dtype=np.uint32)
    npart_total = np.array([0, n_total, 0, 0, 0, 0], dtype=np.uint64)

    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["BoxSize"] = float(box_size)
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["NumPart_ThisFile"] = npart
        header.attrs["NumPart_Total"] = npart_total
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["Omega0"] = OMEGA_M
        header.attrs["OmegaLambda"] = OMEGA_LAMBDA
        header.attrs["HubbleParam"] = HUBBLE_PARAM
        header.attrs["Redshift"] = 0.0
        header.attrs["Time"] = 1.0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_DoublePrecision"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_IC_Info"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_StellarAge"] = 0

        params = f.create_group("Parameters")
        params.attrs["BoxSize"] = float(box_size)
        params.attrs["ComovingIntegrationOn"] = 1
        params.attrs["HubbleParam"] = HUBBLE_PARAM
        params.attrs["Omega0"] = OMEGA_M
        params.attrs["OmegaBaryon"] = OMEGA_BARYON
        params.attrs["OmegaLambda"] = OMEGA_LAMBDA
        params.attrs["TimeBegin"] = 1.0
        params.attrs["TimeMax"] = 1.0
        params.attrs["ICFormat"] = 3
        params.attrs["SnapFormat"] = 3
        params.attrs["UnitLength_in_cm"] = 3.085678e21
        params.attrs["UnitMass_in_g"] = 1.989e43
        params.attrs["UnitVelocity_in_cm_per_s"] = 1.0e5

        config = f.create_group("Config")
        config.attrs["ICGenerator"] = b"generate_unit_test_king_ic.py"
        config.attrs["FinalParticleCount"] = int(args.n_total)
        config.attrs["KingSphereCount"] = int(king_centres.shape[0])
        config.attrs["KingParticleCountEach"] = int(args.n_king)
        config.attrs["KingParticleCountTotal"] = int(args.n_king * king_centres.shape[0])
        config.attrs["KingStartIndex"] = int(king_start_indices[0])
        config.attrs["KingStartIndices"] = king_start_indices.astype(np.int64)
        config.attrs["KingMassMsun"] = float(king_mass_msun)
        config.attrs["RequestedKingMassMsun"] = (
            np.nan if args.king_mass_msun is None else float(args.king_mass_msun)
        )
        config.attrs["CosmologicalBoxMassMsun"] = float(total_box_mass_msun)
        config.attrs["CosmologicalParticleMassMsun"] = float(total_box_mass_msun / args.n_total)
        config.attrs["KingCoreKpc"] = float(args.king_core_kpc)
        config.attrs["KingTruncKpc"] = float(args.king_trunc_kpc)
        config.attrs["KingCentreKpc"] = king_centres[0].astype(np.float64)
        config.attrs["KingCentresKpc"] = king_centres.astype(np.float64)
        config.attrs["KingTranslationKms"] = king_velocities[0].astype(np.float64)
        config.attrs["KingTranslationsKms"] = king_velocities.astype(np.float64)
        config.attrs["KingVelocityModel"] = b"vcirc" if args.vcirc else b"jeans_isotropic"
        config.attrs["Seed"] = int(args.seed)
        config.attrs["RNGStreamScheme"] = b"SeedSequence-spawn-component-v1"

        p1 = f.create_group("PartType1")
        p1.create_dataset("Coordinates", data=coordinates.astype(np.float32), compression="gzip", shuffle=True)
        p1.create_dataset("Velocities", data=velocities.astype(np.float32), compression="gzip", shuffle=True)
        p1.create_dataset("Masses", data=masses_code.astype(np.float32), compression="gzip", shuffle=True)
        p1.create_dataset("ParticleIDs", data=particle_ids.astype(np.uint32), compression="gzip", shuffle=True)


def write_fortran_record(f, payload: bytes) -> None:
    nbytes = len(payload)
    f.write(struct.pack("<I", nbytes))
    f.write(payload)
    f.write(struct.pack("<I", nbytes))


def gadget_binary_header(
    n_total: int,
    box_size: float,
) -> bytes:
    npart = np.array([0, n_total, 0, 0, 0, 0], dtype="<u4")
    massarr = np.zeros(6, dtype="<f8")
    npart_total = np.array([0, n_total, 0, 0, 0, 0], dtype="<u4")
    npart_highword = np.zeros(6, dtype="<u4")

    parts = [
        npart.tobytes(),
        massarr.tobytes(),
        struct.pack("<d", 1.0),          # time
        struct.pack("<d", 0.0),          # redshift
        struct.pack("<i", 0),            # flag_sfr
        struct.pack("<i", 0),            # flag_feedback
        npart_total.tobytes(),
        struct.pack("<i", 0),            # flag_cooling
        struct.pack("<i", 1),            # num_files
        struct.pack("<d", float(box_size)),
        struct.pack("<d", OMEGA_M),
        struct.pack("<d", OMEGA_LAMBDA),
        struct.pack("<d", HUBBLE_PARAM),
        struct.pack("<i", 0),            # flag_stellarage
        struct.pack("<i", 0),            # flag_metals
        npart_highword.tobytes(),
        struct.pack("<i", 0),            # flag_entropy_instead_u
        struct.pack("<i", 0),            # flag_doubleprecision
        struct.pack("<i", 0),            # flag_lpt_ics
        struct.pack("<f", 0.0),          # lpt_scalingfactor
    ]
    header = b"".join(parts)
    if len(header) > 256:
        raise ValueError(f"Gadget header is {len(header)} bytes, expected <= 256")
    return header + bytes(256 - len(header))


def write_gadget_binary_snapshot(
    path: Path,
    coordinates: np.ndarray,
    velocities: np.ndarray,
    masses_code: np.ndarray,
    particle_ids: np.ndarray,
    box_size: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        write_fortran_record(f, gadget_binary_header(particle_ids.size, box_size))
        write_fortran_record(f, coordinates.astype("<f4").tobytes())
        write_fortran_record(f, velocities.astype("<f4").tobytes())
        write_fortran_record(f, particle_ids.astype("<u4").tobytes())
        write_fortran_record(f, masses_code.astype("<f4").tobytes())


def r200_from_mass(
    mass_msun: float,
    core_radius: float,
    trunc_radius: float,
) -> float:
    grid = np.geomspace(max(core_radius * 1.0e-5, 1.0e-6), trunc_radius, 200_000)
    enclosed = mass_msun * king_mass_shape(grid / core_radius)
    enclosed /= king_mass_shape(np.array(trunc_radius / core_radius))
    density = enclosed / ((4.0 * np.pi / 3.0) * grid**3)
    target = 200.0 * CRITICAL_DENSITY_MSUN_H_PER_MPC_H3 / 1.0e9
    crossing = np.flatnonzero(density <= target)
    if crossing.size == 0:
        raise ValueError("King profile does not cross 200 times critical density")
    index = int(crossing[0])
    if index == 0:
        return float(grid[0])
    log_radius = np.interp(
        np.log(target),
        np.log(density[index - 1 : index + 1])[::-1],
        np.log(grid[index - 1 : index + 1])[::-1],
    )
    return float(np.exp(log_radius))


def write_quicklook_plot(
    path: Path,
    coordinates: np.ndarray,
    king_centres: np.ndarray,
    box_size: float,
    king_mass: float,
    core_radius: float,
    trunc_radius: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    centre = king_centres[0]
    relative = (
        coordinates - centre + 0.5 * box_size
    ) % box_size - 0.5 * box_size
    limits = (-5.0, 5.0, -5.0, 5.0)
    relative_mpc = relative / 1000.0
    counts, _, _ = np.histogram2d(
        relative_mpc[:, 0],
        relative_mpc[:, 1],
        bins=1000,
        range=((limits[0], limits[1]), (limits[2], limits[3])),
    )
    density = counts / ((limits[1] - limits[0]) / 1000.0) ** 2
    positive = density > 0.0
    floor = 0.1 * np.min(density[positive])
    log_density = np.log10(np.maximum(density, floor))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        log_density.T,
        origin="lower",
        extent=limits,
        interpolation="nearest",
        cmap="viridis",
        aspect="equal",
        norm=PowerNorm(
            gamma=5.0 / 3.0,
            vmin=float(np.percentile(log_density, 1.0)),
            vmax=float(np.percentile(log_density, 99.9)),
        ),
    )
    r200 = r200_from_mass(king_mass, core_radius, trunc_radius) / 1000.0
    for king_centre in king_centres:
        relative_centre = (
            king_centre - centre + 0.5 * box_size
        ) % box_size - 0.5 * box_size
        relative_centre_mpc = relative_centre[:2] / 1000.0
        box_mpc = box_size / 1000.0
        for dx in (-box_mpc, 0.0, box_mpc):
            for dy in (-box_mpc, 0.0, box_mpc):
                image_centre = relative_centre_mpc + (dx, dy)
                if (
                    limits[0] - r200 <= image_centre[0] <= limits[1] + r200
                    and limits[2] - r200 <= image_centre[1] <= limits[3] + r200
                ):
                    ax.add_patch(
                        plt.Circle(
                            image_centre,
                            r200,
                            fill=False,
                            edgecolor="white",
                            linewidth=0.8,
                        )
                    )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta x\ [{\rm Mpc}/h]$")
    ax.set_ylabel(r"$\Delta y\ [{\rm Mpc}/h]$")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def default_binary_output(path: Path) -> Path:
    if path.suffix == ".hdf5":
        return path.with_suffix("")
    return path.with_name(path.name + ".gadget")


def default_plot_output(path: Path) -> Path:
    return path.with_suffix(".png")


def main() -> None:
    args = parse_args()
    require_valid_args(args)

    n_spheres = KING_CENTRES_KPC.shape[0]
    # Component-specific streams make each realization invariant to how many
    # rejection draws another component or King sphere happens to consume.
    stream_sequences = np.random.SeedSequence(args.seed).spawn(3 + 3 * n_spheres)
    background_position_rng = np.random.default_rng(stream_sequences[0])
    background_selection_rng = np.random.default_rng(stream_sequences[1])
    quicklook_rng = np.random.default_rng(stream_sequences[2])
    king_position_rngs = [
        np.random.default_rng(stream_sequences[3 + 3 * i])
        for i in range(n_spheres)
    ]
    king_velocity_rngs = [
        np.random.default_rng(stream_sequences[4 + 3 * i])
        for i in range(n_spheres)
    ]
    king_translation_rngs = [
        np.random.default_rng(stream_sequences[5 + 3 * i])
        for i in range(n_spheres)
    ]

    n_king_total = args.n_king * n_spheres
    particle_mass_msun = cosmological_particle_mass(args.box_size_kpc, args.n_total)
    particle_mass_code = particle_mass_msun / PARTICLE_MASS_UNIT_MSUN
    total_box_mass_msun = particle_mass_msun * args.n_total
    king_mass_msun = particle_mass_msun * args.n_king
    velocity_model = "vcirc" if args.vcirc else "jeans_isotropic"

    background_pos = quiet_grid_positions(
        args.n_total,
        args.box_size_kpc,
        args.jitter_fraction,
        background_position_rng,
    )
    remove = np.zeros(args.n_total, dtype=bool)
    remove[
        background_selection_rng.choice(
            args.n_total, size=n_king_total, replace=False
        )
    ] = True
    background_pos = background_pos[~remove]
    background_vel = np.zeros_like(background_pos)

    king_pos_parts = []
    king_vel_parts = []
    king_centres = []
    king_translations = []
    for i, centre in enumerate(KING_CENTRES_KPC):
        king_pos, king_vel, king_centre, king_translation = build_king_sphere(
            args.n_king,
            king_mass_msun,
            args.king_core_kpc,
            args.king_trunc_kpc,
            args.box_size_kpc,
            centre,
            args.translation_kms,
            velocity_model,
            king_position_rngs[i],
            king_velocity_rngs[i],
            king_translation_rngs[i],
        )
        king_pos_parts.append(king_pos)
        king_vel_parts.append(king_vel)
        king_centres.append(king_centre)
        king_translations.append(king_translation)
    king_pos_all = np.vstack(king_pos_parts)
    king_vel_all = np.vstack(king_vel_parts)
    king_centres = np.vstack(king_centres)
    king_translations = np.vstack(king_translations)

    coordinates = np.vstack((background_pos, king_pos_all))
    velocities = np.vstack((background_vel, king_vel_all))
    masses_code = np.full(args.n_total, particle_mass_code, dtype=np.float64)
    particle_ids = np.arange(args.n_total, dtype=np.uint32)
    king_start = background_pos.shape[0]
    king_start_indices = king_start + np.arange(n_spheres, dtype=np.int64) * args.n_king
    coordinates = (coordinates + BOX_ROLL_KPC[None, :]) % args.box_size_kpc
    king_centres = (king_centres + BOX_ROLL_KPC[None, :]) % args.box_size_kpc

    assert_bound_king_particles(
        coordinates,
        velocities,
        king_start_indices,
        king_centres,
        king_translations,
        args.n_king,
        king_mass_msun,
        args.king_core_kpc,
        args.king_trunc_kpc,
        args.box_size_kpc,
    )

    write_snapshot(
        args.output,
        coordinates,
        velocities,
        masses_code,
        particle_ids,
        args.box_size_kpc,
        king_start_indices,
        king_centres,
        king_translations,
        king_mass_msun,
        total_box_mass_msun,
        args,
    )
    binary_output = args.binary_output if args.binary_output is not None else default_binary_output(args.output)
    if not args.no_binary:
        write_gadget_binary_snapshot(
            binary_output,
            coordinates,
            velocities,
            masses_code,
            particle_ids,
            args.box_size_kpc,
        )
    plot_output = args.plot_output if args.plot_output is not None else default_plot_output(args.output)
    write_quicklook_plot(
        plot_output,
        coordinates,
        king_centres,
        args.box_size_kpc,
        king_mass_msun,
        args.king_core_kpc,
        args.king_trunc_kpc,
    )

    print(f"Wrote {args.output}")
    if not args.no_binary:
        print(f"Wrote {binary_output}")
    print(f"Wrote {plot_output}")
    print(f"Total particles: {args.n_total}")
    print(f"Background particles kept: {background_pos.shape[0]}")
    print(f"King spheres: {n_spheres}")
    print(f"King particles per sphere: {args.n_king}")
    print(f"King particles total: {n_king_total}")
    print(f"Cosmological box mass: {total_box_mass_msun:.8e} Msun/h")
    print(f"Particle mass: {particle_mass_msun:.8e} Msun/h ({particle_mass_code:.8e} code units)")
    print(f"King mass from particle count: {king_mass_msun:.8e} Msun/h")
    print(f"King velocity model: {velocity_model}")
    print("King particle boundness: all internal speeds are below local truncated-King escape speed")
    if args.king_mass_msun is not None:
        print(f"Requested King mass (not used for particle mass): {args.king_mass_msun:.8e} Msun/h")
    for i, (centre, translation, start) in enumerate(zip(king_centres, king_translations, king_start_indices)):
        stop = int(start + args.n_king - 1)
        print(f"King {i} centre [kpc/h]: {centre[0]:.8e} {centre[1]:.8e} {centre[2]:.8e}")
        print(f"King {i} translation [km/s]: {translation[0]:.8e} {translation[1]:.8e} {translation[2]:.8e}")
        print(f"King {i} ParticleID range: {int(start)} .. {stop}")


if __name__ == "__main__":
    main()
