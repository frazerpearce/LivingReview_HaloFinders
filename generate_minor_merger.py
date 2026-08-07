#!/usr/bin/env python3
"""Replace one major-merger King halo with a 1:100 King halo."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


G_KPC_KMS2_PER_MSUN = 4.30091e-6
SEPARATIONS_KPC_H = np.arange(500.0, -0.1, -50.0)
VELOCITY_SEPARATION_FLOOR_KPC_H = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--major-dir",
        type=Path,
        default=Path("Simulation/Major_Merger_ICs"),
    )
    parser.add_argument(
        "--king-infall",
        type=Path,
        default=Path("Simulation/data/king_infall.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Simulation/Minor_Merger_ICs"),
    )
    return parser.parse_args()


def periodic_offset(
    positions: np.ndarray, centre: np.ndarray, box_size: float
) -> np.ndarray:
    return (
        positions - centre[None, :] + 0.5 * box_size
    ) % box_size - 0.5 * box_size


def load_small_halo(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float, float]:
    with h5py.File(path, "r") as handle:
        labels = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["Config/ObjectLabel"][:]
        ]
        index = labels.index("control_0.01")
        starts = np.asarray(handle["Config/ParticleStart"], dtype=np.int64)
        counts = np.asarray(handle["Config/ParticleCount"], dtype=np.int64)
        centres = np.asarray(handle["Config/CentreKpc"], dtype=np.float64)
        translations = np.asarray(
            handle["Config/TranslationKms"], dtype=np.float64
        )
        core = float(handle["Config/CoreRadiusKpc"][index])
        trunc = float(handle["Config/TruncRadiusKpc"][index])
        r200 = float(handle["Config/R200Kpc"][index])
        box_size = float(handle["Header"].attrs["BoxSize"])
        start = int(starts[index])
        count = int(counts[index])
        selection = slice(start, start + count)
        positions = np.asarray(
            handle["PartType1/Coordinates"][selection], dtype=np.float64
        )
        velocities = np.asarray(
            handle["PartType1/Velocities"][selection], dtype=np.float64
        )
        masses = np.asarray(
            handle["PartType1/Masses"][selection], dtype=np.float64
        )
    if not np.allclose(masses, masses[0]):
        raise ValueError("The 1:100 template does not have a common mass")
    return (
        periodic_offset(positions, centres[index], box_size),
        velocities - translations[index],
        count,
        float(masses[0]),
        core,
        trunc,
        r200,
    )


def main() -> None:
    args = parse_args()
    (
        small_offsets,
        small_internal_velocities,
        small_count,
        small_particle_mass,
        small_core,
        small_trunc,
        small_r200,
    ) = load_small_halo(args.king_infall)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = args.major_dir / "major_merger_005.hdf5"
    with h5py.File(source, "r") as src:
        coordinates = np.asarray(
            src["PartType1/Coordinates"], dtype=np.float64
        )
        velocities = np.asarray(
            src["PartType1/Velocities"], dtype=np.float64
        )
        masses = np.asarray(src["PartType1/Masses"], dtype=np.float64)
        major_starts = np.asarray(
            src["Config"].attrs["KingStartIndices"], dtype=np.int64
        )
        major_count = int(src["Config"].attrs["KingParticleCountEach"])
        source_centres = np.asarray(
            src["Config"].attrs["KingCentresKpc"], dtype=np.float64
        )
        source_translations = np.asarray(
            src["Config"].attrs["KingTranslationsKms"], dtype=np.float64
        )
        primary_start = int(major_starts[0])
        primary_stop = primary_start + major_count
        box_size = float(src["Header"].attrs["BoxSize"])
        major_mass = float(src["Config"].attrs["KingMassMsun"])
        major_core = float(src["Config"].attrs["KingCoreKpc"])
        major_trunc = float(src["Config"].attrs["KingTruncKpc"])
        primary_offsets = periodic_offset(
            coordinates[primary_start:primary_stop],
            source_centres[0],
            box_size,
        )
        primary_internal_velocities = (
            velocities[primary_start:primary_stop] - source_translations[0]
        )
        background_coordinates = coordinates[:primary_start]
        background_velocities = velocities[:primary_start]
        header_attrs = dict(src["Header"].attrs)
        parameter_attrs = dict(src["Parameters"].attrs)

    if not np.isclose(masses[0], small_particle_mass):
        raise ValueError("Major and 1:100 particle masses differ")

    for snapshot_index, separation in enumerate(SEPARATIONS_KPC_H):
        output = args.output_dir / f"minor_merger_{snapshot_index:03d}.hdf5"
        centre = np.full(3, 0.5 * box_size)
        centres = np.array(
            [
                centre + [-0.5 * separation, 0.0, 0.0],
                centre + [0.5 * separation, 0.0, 0.0],
            ]
        )
        velocity_separation = max(
            float(separation), VELOCITY_SEPARATION_FLOOR_KPC_H
        )
        speed = np.sqrt(
            G_KPC_KMS2_PER_MSUN * major_mass / velocity_separation
        )
        translations = np.array(
            [[speed, 0.0, 0.0], [-speed, 0.0, 0.0]]
        )
        primary_coordinates = (
            centres[0][None, :] + primary_offsets
        ) % box_size
        primary_velocities = (
            primary_internal_velocities + translations[0][None, :]
        )
        small_coordinates = (
            centres[1][None, :] + small_offsets
        ) % box_size
        small_velocities = (
            small_internal_velocities + translations[1][None, :]
        )
        new_coordinates = np.vstack(
            [background_coordinates, primary_coordinates, small_coordinates]
        )
        new_velocities = np.vstack(
            [background_velocities, primary_velocities, small_velocities]
        )
        new_masses = np.full(
            new_coordinates.shape[0], masses[0], dtype=np.float64
        )
        new_ids = np.arange(new_masses.size, dtype=np.uint32)

        with h5py.File(output, "w") as dst:
                header = dst.create_group("Header")
                for key, value in header_attrs.items():
                    header.attrs[key] = value
                parameters = dst.create_group("Parameters")
                for key, value in parameter_attrs.items():
                    parameters.attrs[key] = value
                npart = np.zeros(6, dtype=np.uint32)
                npart[1] = new_masses.size
                dst["Header"].attrs["NumPart_ThisFile"] = npart
                dst["Header"].attrs["NumPart_Total"] = npart

                config = dst.create_group("Config")
                config.attrs["ICGenerator"] = "Generate_Minor_Merger.py"
                config.attrs["SourceMajorSnapshot"] = str(source)
                config.attrs["SnapshotIndex"] = snapshot_index
                config.attrs["SnapshotCount"] = 11
                config.attrs["BackgroundParticleCount"] = int(major_starts[0])
                config.attrs["KingSphereCount"] = 2
                config.attrs["KingParticleCounts"] = np.array(
                    [major_count, small_count], dtype=np.int64
                )
                config.attrs["KingStartIndices"] = np.array(
                    [major_starts[0], primary_stop], dtype=np.int64
                )
                config.attrs["KingMassRatios"] = np.array([1.0, 0.01])
                config.attrs["KingCentresKpc"] = centres
                config.attrs["KingTranslationsKms"] = translations
                config.attrs["KingCoreRadiiKpc"] = np.array(
                    [major_core, small_core]
                )
                config.attrs["KingTruncRadiiKpc"] = np.array(
                    [major_trunc, small_trunc]
                )
                config.attrs["KingR200Kpc"] = np.array(
                    [259.52436892, small_r200]
                )
                config.attrs["SeparationKpc"] = separation
                config.attrs["VelocityPrescription"] = (
                    "major-merger parabolic speed at refined separation"
                )
                config.attrs["VelocitySeparationFloorKpc"] = (
                    VELOCITY_SEPARATION_FLOOR_KPC_H
                )

                particles = dst.create_group("PartType1")
                for name, values in (
                    ("Coordinates", new_coordinates.astype(np.float32)),
                    ("Velocities", new_velocities.astype(np.float32)),
                    ("Masses", new_masses.astype(np.float32)),
                    ("ParticleIDs", new_ids),
                ):
                    particles.create_dataset(
                        name, data=values, compression="gzip", shuffle=True
                    )
        print(
            f"Wrote {output}: separation={separation:.1f} kpc/h, "
            f"speed/halo={speed:.8f} km/s"
        )


if __name__ == "__main__":
    main()
