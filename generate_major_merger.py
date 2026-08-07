#!/usr/bin/env python3
"""Generate a controlled equal-mass major-merger snapshot sequence.

The background is the central 5 x 5 x 5 (Mpc/h)^3 region of the existing
King unit-test box, with its original King particles removed.  Two copies of
the large King sphere are then placed on the x-axis, initially just touching.
Successive snapshots move each centre 100 kpc/h toward the box centre, ending
with the two centres coincident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


G_KPC_KMS2_PER_MSUN = 4.30091e-6
DEFAULT_INPUT = Path("Simulation/data/unit_test_king_box.hdf5")
DEFAULT_OUTPUT_DIR = Path("Simulation/Major_Merger_ICs")
OUTPUT_BOX_KPC_H = 5000.0
INITIAL_HALF_SEPARATION_KPC_H = 500.0
STEP_PER_HALO_KPC_H = 50.0
SNAPSHOT_COUNT = 11
MIN_VELOCITY_SEPARATION_KPC_H = 2.0 * STEP_PER_HALO_KPC_H


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Existing King unit-test Gadget HDF5 snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving major_merger_000.hdf5 ... 010.hdf5.",
    )
    return parser.parse_args()


def periodic_offset(
    coordinates: np.ndarray,
    centre: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """Return minimum-image offsets from centre."""
    return (
        coordinates - centre[None, :] + 0.5 * box_size
    ) % box_size - 0.5 * box_size


def load_components(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    dict[str, float],
]:
    """Read the cropped background and two King internal realizations."""
    with h5py.File(path, "r") as handle:
        coordinates = handle["PartType1/Coordinates"][:].astype(np.float64)
        velocities = handle["PartType1/Velocities"][:].astype(np.float64)
        masses = handle["PartType1/Masses"][:].astype(np.float64)
        config = handle["Config"].attrs
        input_box = float(handle["Header"].attrs["BoxSize"])
        starts = np.asarray(config["KingStartIndices"], dtype=np.int64)
        count = int(config["KingParticleCountEach"])
        centres = np.asarray(config["KingCentresKpc"], dtype=np.float64)
        translations = np.asarray(
            config["KingTranslationsKms"], dtype=np.float64
        )
        king_mass = float(config["KingMassMsun"])
        core_radius = float(config["KingCoreKpc"])
        trunc_radius = float(config["KingTruncKpc"])
        omega_m = float(handle["Header"].attrs["Omega0"])
        omega_lambda = float(handle["Header"].attrs["OmegaLambda"])
        hubble = float(handle["Header"].attrs["HubbleParam"])

    if starts.size < 2:
        raise ValueError("The source snapshot must contain two King spheres")
    if not np.allclose(masses, masses[0]):
        raise ValueError("The source snapshot must use a common particle mass")

    first_king = int(starts.min())
    background_coordinates = coordinates[:first_king]
    background_velocities = velocities[:first_king]
    lower = 0.5 * (input_box - OUTPUT_BOX_KPC_H)
    upper = lower + OUTPUT_BOX_KPC_H
    inside = np.all(
        (background_coordinates >= lower)
        & (background_coordinates < upper),
        axis=1,
    )
    background_coordinates = background_coordinates[inside] - lower
    background_velocities = background_velocities[inside]

    king_offsets: list[np.ndarray] = []
    king_internal_velocities: list[np.ndarray] = []
    for start, centre, translation in zip(
        starts[:2], centres[:2], translations[:2]
    ):
        selection = slice(int(start), int(start) + count)
        king_offsets.append(
            periodic_offset(
                coordinates[selection],
                centre,
                input_box,
            )
        )
        king_internal_velocities.append(
            velocities[selection] - translation[None, :]
        )

    header_values = {
        "Omega0": omega_m,
        "OmegaLambda": omega_lambda,
        "HubbleParam": hubble,
    }
    return (
        background_coordinates,
        background_velocities,
        np.stack(king_offsets),
        np.stack(king_internal_velocities),
        float(masses[0]),
        king_mass,
        core_radius,
        {"TruncRadiusKpc": trunc_radius, **header_values},
    )


def write_snapshot(
    path: Path,
    coordinates: np.ndarray,
    velocities: np.ndarray,
    particle_mass_code: float,
    centres: np.ndarray,
    bulk_velocities: np.ndarray,
    background_count: int,
    king_count: int,
    king_mass_msun_h: float,
    core_radius: float,
    metadata: dict[str, float],
    snapshot_index: int,
) -> None:
    particle_count = coordinates.shape[0]
    particle_ids = np.arange(particle_count, dtype=np.uint32)
    masses = np.full(particle_count, particle_mass_code, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        npart = np.zeros(6, dtype=np.uint32)
        npart[1] = particle_count
        header.attrs["BoxSize"] = OUTPUT_BOX_KPC_H
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["NumPart_ThisFile"] = npart
        header.attrs["NumPart_Total"] = npart
        header.attrs["NumPart_Total_HighWord"] = np.zeros(
            6, dtype=np.uint32
        )
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 1.0
        header.attrs["Redshift"] = 0.0
        header.attrs["Omega0"] = metadata["Omega0"]
        header.attrs["OmegaLambda"] = metadata["OmegaLambda"]
        header.attrs["HubbleParam"] = metadata["HubbleParam"]
        for name in (
            "Flag_Cooling",
            "Flag_DoublePrecision",
            "Flag_Feedback",
            "Flag_Metals",
            "Flag_Sfr",
            "Flag_StellarAge",
        ):
            header.attrs[name] = 0

        parameters = handle.create_group("Parameters")
        parameters.attrs["BoxSize"] = OUTPUT_BOX_KPC_H
        parameters.attrs["ComovingIntegrationOn"] = 1
        parameters.attrs["HubbleParam"] = metadata["HubbleParam"]
        parameters.attrs["Omega0"] = metadata["Omega0"]
        parameters.attrs["OmegaLambda"] = metadata["OmegaLambda"]
        parameters.attrs["TimeBegin"] = 1.0
        parameters.attrs["TimeMax"] = 1.0
        parameters.attrs["ICFormat"] = 3
        parameters.attrs["SnapFormat"] = 3
        parameters.attrs["UnitLength_in_cm"] = 3.085678e21
        parameters.attrs["UnitMass_in_g"] = 1.989e43
        parameters.attrs["UnitVelocity_in_cm_per_s"] = 1.0e5

        config = handle.create_group("Config")
        config.attrs["ICGenerator"] = "Generate_Major_Merger.py"
        config.attrs["SourceSnapshot"] = str(DEFAULT_INPUT)
        config.attrs["SnapshotIndex"] = snapshot_index
        config.attrs["SnapshotCount"] = SNAPSHOT_COUNT
        config.attrs["BackgroundParticleCount"] = background_count
        config.attrs["KingSphereCount"] = 2
        config.attrs["KingParticleCountEach"] = king_count
        config.attrs["KingStartIndices"] = np.array(
            [background_count, background_count + king_count],
            dtype=np.int64,
        )
        config.attrs["KingMassMsun"] = king_mass_msun_h
        config.attrs["KingCoreKpc"] = core_radius
        config.attrs["KingTruncKpc"] = metadata["TruncRadiusKpc"]
        config.attrs["KingCentresKpc"] = centres
        config.attrs["KingTranslationsKms"] = bulk_velocities
        config.attrs["InitialSeparationKpc"] = (
            2.0 * INITIAL_HALF_SEPARATION_KPC_H
        )
        config.attrs["DisplacementPerHaloPerSnapshotKpc"] = (
            STEP_PER_HALO_KPC_H
        )
        config.attrs["VelocityPrescription"] = (
            "parabolic_equal_mass_at_snapshot_separation"
        )
        config.attrs["VelocitySeparationFloorKpc"] = (
            MIN_VELOCITY_SEPARATION_KPC_H
        )

        particles = handle.create_group("PartType1")
        for name, values in (
            ("Coordinates", coordinates.astype(np.float32)),
            ("Velocities", velocities.astype(np.float32)),
            ("Masses", masses),
            ("ParticleIDs", particle_ids),
        ):
            particles.create_dataset(
                name,
                data=values,
                compression="gzip",
                shuffle=True,
            )


def main() -> None:
    args = parse_args()
    (
        background_coordinates,
        background_velocities,
        king_offsets,
        king_internal_velocities,
        particle_mass_code,
        king_mass,
        core_radius,
        metadata,
    ) = load_components(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    centre = np.full(3, 0.5 * OUTPUT_BOX_KPC_H)
    king_count = king_offsets.shape[1]
    for snapshot_index in range(SNAPSHOT_COUNT):
        half_separation = (
            INITIAL_HALF_SEPARATION_KPC_H
            - snapshot_index * STEP_PER_HALO_KPC_H
        )
        separation = 2.0 * half_separation
        velocity_separation = max(
            separation,
            MIN_VELOCITY_SEPARATION_KPC_H,
        )
        speed_per_halo = np.sqrt(
            G_KPC_KMS2_PER_MSUN * king_mass / velocity_separation
        )
        bulk_velocities = np.array(
            [
                [speed_per_halo, 0.0, 0.0],
                [-speed_per_halo, 0.0, 0.0],
            ]
        )
        centres = np.array(
            [
                centre + [-half_separation, 0.0, 0.0],
                centre + [half_separation, 0.0, 0.0],
            ]
        )
        king_coordinates = (
            king_offsets + centres[:, None, :]
        ) % OUTPUT_BOX_KPC_H
        king_velocities = (
            king_internal_velocities + bulk_velocities[:, None, :]
        )
        coordinates = np.vstack(
            [background_coordinates, *king_coordinates]
        )
        velocities = np.vstack(
            [background_velocities, *king_velocities]
        )
        output = args.output_dir / (
            f"major_merger_{snapshot_index:03d}.hdf5"
        )
        write_snapshot(
            output,
            coordinates,
            velocities,
            particle_mass_code,
            centres,
            bulk_velocities,
            background_coordinates.shape[0],
            king_count,
            king_mass,
            core_radius,
            metadata,
            snapshot_index,
        )
        print(
            f"Wrote {output}: separation={separation:.1f} kpc/h, "
            f"speed/halo={speed_per_halo:.8f} km/s"
        )

    print(
        "The co-centred snapshot uses the final non-zero separation "
        f"({MIN_VELOCITY_SEPARATION_KPC_H:.1f} kpc/h) for its finite speed."
    )
    print(f"Background particles: {background_coordinates.shape[0]}")
    print(f"Particles per King sphere: {king_count}")


if __name__ == "__main__":
    main()
