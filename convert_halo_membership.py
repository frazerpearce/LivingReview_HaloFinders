#!/usr/bin/env python3
"""Convert halo-finder memberships to a minimal common HDF5 catalogue.

Output layout
-------------
/Header                       HDF5 group; input flags are stored as attrs
/Haloes/haloid                int64   [Nhalo]
/Haloes/centre                float64 [Nhalo, 3]
/Haloes/catalogue_mass        float64 [Nhalo]
/Haloes/catalogue_radius      float64 [Nhalo]
/Haloes/offset                int64   [Nhalo + 1]
/Haloes/particle_id           uint64  [Nmembership]

Halo i contains particle_id[offset[i]:offset[i + 1]]. The common physical units
are kpc/h for centres and radii and Msun/h for masses.

Supported inputs
----------------
--ahf
    An AHF catalogue prefix, directory, or .AHF_particles file. Standard
    AHF_particles blocks and the corresponding .AHF_halos table are read.

--rockstar
    A Rockstar catalogue file, directory, prefix, or glob matching
    halos_*.particles membership tables or halos_*.bin catalogues. Full-particle
    tables are preferred because their external_haloid field preserves
    Rockstar's inclusive parent memberships. Binary catalogues provide the
    finder-assigned per-halo memberships when full-particle tables were not
    written; descendant memberships are then unioned into their geometric
    parents. BGC2 SO particle spheres are deliberately not used as memberships.

--subfind
    A GADGET-4 SUBFIND HDF5 catalogue file, prefix, directory, or glob.
    Memberships are reconstructed from SubhaloOffsetType/SubhaloLenType and the
    corresponding HDF5 snapshot particle ordering.

AHF properties use Xc/Yc/Zc and Mhalo/Rhalo or Mvir/Rvir. Rockstar uses M200c
when it is positive and otherwise Mvir, and uses R200c when available and otherwise Rvir.
SUBFIND uses SubhaloPos and SubhaloMass. The first (central) subhalo in each
FoF group uses its parent Group_R_Crit200; satellites use SubhaloHalfmassRad.

The snapshot must be HDF5 and may be a single file, a multi-file prefix, a
single chunk, a directory, or a glob. Particle IDs are read from
PartType*/ParticleIDs (with common ID aliases accepted).
"""

import argparse
import glob
import json
import re
import shlex
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np

ROCKSTAR_MAGIC = np.uint64(0xFADEDACEC0C0D0D0)
ROCKSTAR_HEADER_SIZE = 256
ROCKSTAR_FORMAT_REVISION = 1
ROCKSTAR_STANDARD_HALO_SIZE = 264
ROCKSTAR_ID_OFFSET = 0
ROCKSTAR_POSITION_OFFSET = 8
ROCKSTAR_MASS_OFFSET = 56
ROCKSTAR_RADIUS_OFFSET = 60
ROCKSTAR_M200C_OFFSET = 120
ROCKSTAR_NUM_P_OFFSET = 200
ROCKSTAR_P_START_OFFSET = 216
SUBFIND_MASS_TO_MSUN_H = 1.0e10
MPC_TO_KPC = 1.0e3


class CatalogueError(RuntimeError):
    pass


# NEW FINDER CHECKLIST
# --------------------
# 1. Add ``read_<finder>()`` at the reader stub below. It must return:
#      haloid, centre, catalogue_mass, catalogue_radius,
#      offset, particle_id, catalogue_files
# 2. Add one mutually exclusive CLI option in ``build_parser()``.
# 3. Add one dispatch branch in ``main()`` and set ``finder`` to the same key
#    registered in finder_config.py.
# 4. Convert units before returning: centre/radius in kpc/h, mass in Msun/h,
#    haloid int64, offset int64, and particle_id uint64.
# 5. Let the existing validators and write_output() create the common HDF5.


def natural_key(path: Path) -> tuple[int | str, ...]:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(path)))


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in sorted((p.resolve() for p in paths), key=natural_key):
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def expand_path_set(spec: str, *, suffixes: Sequence[str] = ()) -> list[Path]:
    """Resolve a file, directory, prefix, or glob into an ordered file set."""
    path = Path(spec).expanduser()
    candidates: list[Path] = []

    if any(ch in spec for ch in "*?["):
        candidates.extend(Path(p) for p in glob.glob(str(path)))
    elif path.is_file():
        candidates.append(path)
    elif path.is_dir():
        for suffix in suffixes:
            candidates.extend(path.glob(f"*{suffix}"))
    else:
        for suffix in suffixes:
            exact = Path(f"{path}{suffix}")
            if exact.is_file():
                candidates.append(exact)
            candidates.extend(Path(p) for p in glob.glob(f"{path}.*{suffix}"))
            candidates.extend(Path(p) for p in glob.glob(f"{path}*{suffix}"))

    files = unique_paths(p for p in candidates if p.is_file())
    if not files:
        suffix_text = ", ".join(suffixes) if suffixes else "requested files"
        raise CatalogueError(f"No {suffix_text} found for: {spec}")
    return files


def resolve_snapshot_files(spec: str) -> list[Path]:
    path = Path(spec).expanduser()
    candidates: list[Path] = []

    if any(ch in spec for ch in "*?["):
        candidates.extend(Path(p) for p in glob.glob(str(path)))
    elif path.is_file():
        candidates.append(path)
        match = re.match(r"^(.*)\.\d+\.(hdf5|h5)$", str(path))
        if match:
            candidates.extend(
                Path(p) for p in glob.glob(f"{match.group(1)}.*.{match.group(2)}")
            )
    elif path.is_dir():
        candidates.extend(path.glob("*.hdf5"))
        candidates.extend(path.glob("*.h5"))
    else:
        for pattern in (
            str(path),
            f"{path}.hdf5",
            f"{path}.h5",
            f"{path}.*.hdf5",
            f"{path}.*.h5",
        ):
            candidates.extend(Path(p) for p in glob.glob(pattern))

    files = unique_paths(p for p in candidates if p.is_file())
    if not files:
        raise CatalogueError(f"No HDF5 snapshot files found for: {spec}")

    for file in files:
        if not h5py.is_hdf5(file):
            raise CatalogueError(f"Snapshot file is not HDF5: {file}")
    return files


def find_id_dataset(group: h5py.Group) -> h5py.Dataset | None:
    for name in ("ParticleIDs", "ParticleID", "Particle_IDs", "IDs", "ID"):
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def load_snapshot_ids_by_type(snapshot_files: Sequence[Path]) -> dict[int, np.ndarray]:
    chunks: defaultdict[int, list[np.ndarray]] = defaultdict(list)

    for filename in snapshot_files:
        with h5py.File(filename, "r") as handle:
            found = False
            for name, obj in handle.items():
                if not name.startswith("PartType") or not isinstance(obj, h5py.Group):
                    continue
                try:
                    ptype = int(name[len("PartType") :])
                except ValueError:
                    continue
                dataset = find_id_dataset(obj)
                if dataset is None:
                    continue
                values = np.asarray(dataset[...])
                if values.ndim != 1:
                    raise CatalogueError(f"{filename}:{dataset.name} is not one-dimensional")
                if values.dtype.kind not in "iu":
                    raise CatalogueError(f"{filename}:{dataset.name} is not an integer dataset")
                if values.dtype.kind == "i" and np.any(values < 0):
                    raise CatalogueError(f"{filename}:{dataset.name} contains negative particle IDs")
                chunks[ptype].append(values.astype(np.uint64, copy=False))
                found = True
            if not found:
                raise CatalogueError(f"No PartType*/ParticleIDs datasets found in {filename}")

    return {
        ptype: np.concatenate(parts) if len(parts) > 1 else parts[0].copy()
        for ptype, parts in sorted(chunks.items())
    }


def flatten_snapshot_ids(ids_by_type: dict[int, np.ndarray]) -> np.ndarray:
    if not ids_by_type:
        return np.empty(0, dtype=np.uint64)
    arrays = list(ids_by_type.values())
    result = np.concatenate(arrays) if len(arrays) > 1 else arrays[0].copy()
    result.sort()
    return result


def checked_uint64(value: str, context: str) -> np.uint64:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CatalogueError(f"Invalid integer {value!r} in {context}") from exc
    if parsed < 0 or parsed > np.iinfo(np.uint64).max:
        raise CatalogueError(f"Particle ID outside uint64 range in {context}: {parsed}")
    return np.uint64(parsed)


def checked_int64(value: str, context: str) -> np.int64:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CatalogueError(f"Invalid integer {value!r} in {context}") from exc
    if parsed < np.iinfo(np.int64).min or parsed > np.iinfo(np.int64).max:
        raise CatalogueError(f"Halo ID outside int64 range in {context}: {parsed}")
    return np.int64(parsed)


def resolve_ahf_halos_file(particles_file: Path) -> Path:
    name = particles_file.name
    if not name.endswith(".AHF_particles"):
        raise CatalogueError(f"Not an AHF particle catalogue: {particles_file}")
    halos_file = particles_file.with_name(name[: -len(".AHF_particles")] + ".AHF_halos")
    if not halos_file.is_file():
        raise CatalogueError(f"Corresponding AHF halo table not found: {halos_file}")
    return halos_file


def read_ahf_halo_table(
    particles_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    halos_file = resolve_ahf_halos_file(particles_file)
    header: list[str] | None = None
    rows: list[list[str]] = []

    with halos_file.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if header is None:
                    header = [re.sub(r"\(\d+\)$", "", name) for name in stripped[1:].split()]
                continue
            rows.append(stripped.split())

    if header is None:
        raise CatalogueError(f"AHF halo table has no column header: {halos_file}")
    columns = {name.lower(): index for index, name in enumerate(header)}
    required = ("id", "xc", "yc", "zc")
    missing = [name for name in required if name not in columns]
    mass_name = next((name for name in ("mhalo", "mvir") if name in columns), None)
    radius_name = next((name for name in ("rhalo", "rvir") if name in columns), None)
    if mass_name is None:
        missing.append("Mhalo/Mvir")
    if radius_name is None:
        missing.append("Rhalo/Rvir")
    if missing:
        raise CatalogueError(
            f"AHF halo table {halos_file} lacks columns: {', '.join(missing)}"
        )

    halo_ids = np.empty(len(rows), dtype=np.int64)
    centres = np.empty((len(rows), 3), dtype=np.float64)
    masses = np.empty(len(rows), dtype=np.float64)
    radii = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        context = f"{halos_file}:data row {index + 1}"
        try:
            halo_ids[index] = checked_int64(row[columns["id"]], context)
            centres[index] = [float(row[columns[name]]) for name in ("xc", "yc", "zc")]
            masses[index] = float(row[columns[mass_name]])
            radii[index] = float(row[columns[radius_name]])
        except IndexError as exc:
            raise CatalogueError(f"Truncated AHF halo row in {context}") from exc
        except ValueError as exc:
            raise CatalogueError(f"Invalid floating-point value in {context}") from exc
    return halo_ids, centres, masses, radii, halos_file


def read_ahf(
    spec: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[Path],
]:
    files = expand_path_set(spec, suffixes=(".AHF_particles",))
    if len(files) != 1:
        raise CatalogueError(
            f"AHF conversion expects one .AHF_particles file; resolved {len(files)} files"
    )
    filename = files[0]
    table_ids, table_centres, table_masses, table_radii, halos_file = (
        read_ahf_halo_table(filename)
    )

    with filename.open("rt", encoding="utf-8") as stream:
        lines = ((number, line.strip()) for number, line in enumerate(stream, 1))
        content = [(number, line) for number, line in lines if line and not line.startswith("#")]

    if not content:
        raise CatalogueError(f"Empty AHF particle catalogue: {filename}")

    cursor = 0
    declared_nhalo: int | None = None
    first_tokens = content[0][1].split()
    if len(first_tokens) == 1:
        declared_nhalo = int(first_tokens[0])
        if declared_nhalo < 0:
            raise CatalogueError(f"Negative halo count in {filename}:{content[0][0]}")
        cursor = 1

    halo_ids: list[np.int64] = []
    counts: list[int] = []
    memberships: list[np.ndarray] = []
    block_index = 0

    while cursor < len(content):
        line_number, header = content[cursor]
        cursor += 1
        tokens = header.split()
        if not tokens:
            continue
        try:
            count = int(tokens[0])
        except ValueError as exc:
            raise CatalogueError(f"Invalid AHF block count in {filename}:{line_number}") from exc
        if count < 0:
            raise CatalogueError(f"Negative AHF block count in {filename}:{line_number}")

        if len(tokens) >= 2:
            haloid = checked_int64(tokens[1], f"{filename}:{line_number}")
        elif block_index < table_ids.size:
            haloid = table_ids[block_index]
        else:
            raise CatalogueError(
                f"AHF block {block_index} has no halo ID and no matching .AHF_halos entry"
            )

        if cursor + count > len(content):
            raise CatalogueError(
                f"AHF block {haloid} declares {count} particles but the file ends early"
            )

        ids = np.empty(count, dtype=np.uint64)
        for local_index in range(count):
            particle_line_number, particle_line = content[cursor + local_index]
            particle_tokens = particle_line.split()
            if not particle_tokens:
                raise CatalogueError(f"Blank particle row in {filename}:{particle_line_number}")
            ids[local_index] = checked_uint64(
                particle_tokens[0], f"{filename}:{particle_line_number}"
            )
        cursor += count

        halo_ids.append(haloid)
        counts.append(count)
        memberships.append(ids)
        block_index += 1

    if declared_nhalo is not None and declared_nhalo != len(halo_ids):
        raise CatalogueError(
            f"AHF header declares {declared_nhalo} halos but {len(halo_ids)} blocks were read"
        )
    if table_ids.size < len(halo_ids):
        raise CatalogueError("AHF_halos contains fewer halo IDs than AHF_particles blocks")

    table_index = {int(haloid): index for index, haloid in enumerate(table_ids)}
    if len(table_index) != table_ids.size:
        raise CatalogueError(f"Duplicate halo IDs in {halos_file}")
    try:
        indices = np.asarray([table_index[int(haloid)] for haloid in halo_ids])
    except KeyError as exc:
        raise CatalogueError(f"AHF membership halo {exc.args[0]} is absent from {halos_file}") from exc
    packed = pack_memberships(halo_ids, counts, memberships)
    return (
        packed[0],
        table_centres[indices],
        table_masses[indices],
        table_radii[indices],
        packed[1],
        packed[2],
        [filename, halos_file],
    )


def rockstar_header_dtype() -> np.dtype:
    base = np.dtype(
        [
            ("magic", "<u8"),
            ("snap", "<i8"),
            ("chunk", "<i8"),
            ("scale", "<f4"),
            ("Om", "<f4"),
            ("Ol", "<f4"),
            ("h0", "<f4"),
            ("bounds", "<f4", (6,)),
            ("num_halos", "<i8"),
            ("num_particles", "<i8"),
            ("box_size", "<f4"),
            ("particle_mass", "<f4"),
            ("particle_type", "<i8"),
            ("format_revision", "<i4"),
            ("rockstar_version", "S12"),
        ],
        align=True,
    )
    padding = ROCKSTAR_HEADER_SIZE - base.itemsize
    if padding < 0:
        raise AssertionError("Rockstar header dtype exceeds 256 bytes")
    return np.dtype(base.descr + [("unused", "u1", (padding,))], align=True)


def rockstar_membership_dtype(record_size: int) -> np.dtype:
    """Return a minimal view of a Rockstar halo record.

    Rockstar binary files are native C-structure dumps. Some forks append
    additional halo properties without changing HALO_FORMAT_REVISION, so the
    complete record can be larger than the public 264-byte structure. The
    membership fields retain their revision-1 offsets.
    """
    minimum_size = ROCKSTAR_P_START_OFFSET + np.dtype("<i8").itemsize
    if record_size < minimum_size:
        raise CatalogueError(
            f"Rockstar halo record is {record_size} bytes; "
            f"at least {minimum_size} are required"
        )
    if record_size % 4:
        raise CatalogueError(
            f"Rockstar halo record size {record_size} is not four-byte aligned"
        )
    return np.dtype(
        {
            "names": [
                "id",
                "centre",
                "primary_mass",
                "catalogue_radius",
                "m200c",
                "num_p",
                "p_start",
            ],
            "formats": [
                "<i8",
                ("<f4", (3,)),
                "<f4",
                "<f4",
                "<f4",
                "<i8",
                "<i8",
            ],
            "offsets": [
                ROCKSTAR_ID_OFFSET,
                ROCKSTAR_POSITION_OFFSET,
                ROCKSTAR_MASS_OFFSET,
                ROCKSTAR_RADIUS_OFFSET,
                ROCKSTAR_M200C_OFFSET,
                ROCKSTAR_NUM_P_OFFSET,
                ROCKSTAR_P_START_OFFSET,
            ],
            "itemsize": record_size,
        }
    )


def infer_rockstar_halo_record_size(
    filename: Path, nhalo: int, nparticle: int
) -> int:
    """Infer the native halo structure size from exact file accounting."""
    file_size = filename.stat().st_size
    particle_bytes = nparticle * np.dtype("<i8").itemsize
    halo_bytes = file_size - ROCKSTAR_HEADER_SIZE - particle_bytes
    if halo_bytes < 0:
        raise CatalogueError(
            f"Rockstar file is too short for its header counts: {filename}"
        )

    if nhalo == 0:
        if halo_bytes:
            raise CatalogueError(
                f"Rockstar file {filename} has no halos but "
                f"{halo_bytes} unexplained bytes"
            )
        return ROCKSTAR_STANDARD_HALO_SIZE

    record_size, remainder = divmod(halo_bytes, nhalo)
    if remainder:
        raise CatalogueError(
            f"Cannot infer Rockstar halo record size in {filename}: "
            f"{halo_bytes} halo-table bytes are not divisible by {nhalo} halos"
        )
    return int(record_size)


def inclusive_memberships_from_children(
    halo_ids: np.ndarray,
    memberships: Sequence[np.ndarray],
    children: dict[int, list[int]],
) -> list[np.ndarray]:
    """Union each halo's assigned particles with all of its descendants."""
    id_to_index = {int(haloid): index for index, haloid in enumerate(halo_ids)}
    cache: dict[int, np.ndarray] = {}
    active: set[int] = set()

    def collect(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        if index in active:
            raise CatalogueError(
                f"Cycle in Rockstar hierarchy at halo {int(halo_ids[index])}"
            )
        active.add(index)
        parts = [memberships[index]] if memberships[index].size else []
        for child_id in children.get(int(halo_ids[index]), ()):
            child = collect(id_to_index[child_id])
            if child.size:
                parts.append(child)
        active.remove(index)
        if not parts:
            result = np.empty(0, dtype=np.uint64)
        elif len(parts) == 1:
            result = parts[0].copy()
        else:
            result = np.unique(np.concatenate(parts)).astype(np.uint64, copy=False)
        cache[index] = result
        return result

    return [collect(index) for index in range(halo_ids.size)]


def inclusive_rockstar_memberships(
    halo_ids: np.ndarray,
    centres: np.ndarray,
    hierarchy_masses: np.ndarray,
    radii: np.ndarray,
    memberships: Sequence[np.ndarray],
    box_size: float,
) -> list[np.ndarray]:
    """Reproduce the original geometric Rockstar parent/child expansion."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise CatalogueError(
            "Inclusive Rockstar membership requires scipy.spatial.cKDTree"
        ) from exc

    wrapped_centres = np.mod(centres, box_size)
    tree = cKDTree(wrapped_centres, boxsize=box_size)
    parent = np.full(halo_ids.size, -1, dtype=np.int64)
    parent_mass = np.full(halo_ids.size, np.inf, dtype=np.float64)
    order = np.argsort(np.nan_to_num(hierarchy_masses, nan=-np.inf))[::-1]

    for host_index in order:
        host_mass = hierarchy_masses[host_index]
        host_radius = radii[host_index]
        if not (
            np.isfinite(host_mass)
            and np.isfinite(host_radius)
            and host_radius > 0.0
        ):
            continue
        for object_index in tree.query_ball_point(
            wrapped_centres[host_index], host_radius
        ):
            if object_index == host_index:
                continue
            object_mass = hierarchy_masses[object_index]
            if not np.isfinite(object_mass) or object_mass >= host_mass:
                continue
            if host_mass < parent_mass[object_index]:
                parent[object_index] = host_index
                parent_mass[object_index] = host_mass

    children: defaultdict[int, list[int]] = defaultdict(list)
    for child_index, parent_index in enumerate(parent):
        if parent_index >= 0:
            children[int(halo_ids[parent_index])].append(int(halo_ids[child_index]))
    return inclusive_memberships_from_children(halo_ids, memberships, children)


def read_rockstar_binary(
    files: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    header_dtype = rockstar_header_dtype()
    all_halo_ids: list[np.ndarray] = []
    all_centres: list[np.ndarray] = []
    all_masses: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_memberships: list[np.ndarray] = []
    box_sizes: list[float] = []

    for filename in files:
        with filename.open("rb") as stream:
            header = np.fromfile(stream, dtype=header_dtype, count=1)
            if header.size != 1:
                raise CatalogueError(f"Truncated Rockstar header: {filename}")
            row = header[0]
            if row["magic"] != ROCKSTAR_MAGIC:
                raise CatalogueError(f"Bad Rockstar magic number in {filename}")
            if int(row["format_revision"]) != ROCKSTAR_FORMAT_REVISION:
                raise CatalogueError(
                    f"Unsupported Rockstar format revision "
                    f"{int(row['format_revision'])} in {filename}"
                )
            nhalo = int(row["num_halos"])
            nparticle = int(row["num_particles"])
            if nhalo < 0 or nparticle < 0:
                raise CatalogueError(f"Negative Rockstar counts in {filename}")
            box_sizes.append(float(row["box_size"]) * MPC_TO_KPC)

            record_size = infer_rockstar_halo_record_size(
                filename, nhalo, nparticle
            )
            halo_dtype = rockstar_membership_dtype(record_size)
            halos = np.fromfile(stream, dtype=halo_dtype, count=nhalo)
            if halos.size != nhalo:
                raise CatalogueError(f"Truncated Rockstar halo table: {filename}")
            particle_ids = np.fromfile(stream, dtype="<i8", count=nparticle)
            if particle_ids.size != nparticle:
                raise CatalogueError(f"Truncated Rockstar particle table: {filename}")
            if np.any(particle_ids < 0):
                raise CatalogueError(f"Negative particle IDs in {filename}")
            if stream.tell() != filename.stat().st_size:
                raise CatalogueError(
                    f"Rockstar file accounting failed for {filename}"
                )

        starts = halos["p_start"].astype(np.int64, copy=False)
        counts = halos["num_p"].astype(np.int64, copy=False)
        if np.any(starts < 0) or np.any(counts < 0):
            raise CatalogueError(f"Negative Rockstar p_start/num_p in {filename}")
        if np.any(starts > nparticle) or np.any(counts > nparticle - starts):
            raise CatalogueError(
                f"Rockstar membership range exceeds particle table in {filename}; "
                f"the inferred {record_size}-byte halo record may use a non-standard "
                "field layout"
            )

        all_halo_ids.append(halos["id"].astype(np.int64, copy=True))
        all_centres.append(
            halos["centre"].astype(np.float64, copy=True) * MPC_TO_KPC
        )
        primary_mass = halos["primary_mass"].astype(np.float64, copy=False)
        m200c = halos["m200c"].astype(np.float64, copy=False)
        all_masses.append(np.where(m200c > 0.0, m200c, primary_mass))
        all_radii.append(
            halos["catalogue_radius"].astype(np.float64, copy=True)
        )
        for start_index, count in zip(starts, counts):
            all_memberships.append(
                particle_ids[
                    int(start_index) : int(start_index + count)
                ].astype(np.uint64, copy=True)
            )

    halo_ids = np.concatenate(all_halo_ids) if all_halo_ids else np.empty(0, np.int64)
    centres = np.concatenate(all_centres) if all_centres else np.empty((0, 3), np.float64)
    masses = np.concatenate(all_masses) if all_masses else np.empty(0, np.float64)
    radii = np.concatenate(all_radii) if all_radii else np.empty(0, np.float64)
    hierarchy_masses = masses.copy()

    # Use the public catalogue for the standardized finder properties.  The
    # binary halo records remain the authoritative source of particle ranges,
    # but forks can lay out optional mass fields differently.  This also keeps
    # the common contract explicit: positive M200c (otherwise Mvir), and R200c
    # when present (otherwise Rvir).
    list_properties = read_rockstar_list_properties(files)
    if list_properties:
        missing = [int(haloid) for haloid in halo_ids if int(haloid) not in list_properties]
        if missing:
            raise CatalogueError(
                f"Rockstar list properties missing for binary IDs: {missing[:10]}"
            )
        centres = np.asarray([list_properties[int(haloid)][0] for haloid in halo_ids])
        masses = np.asarray([list_properties[int(haloid)][1] for haloid in halo_ids])
        radii = np.asarray([list_properties[int(haloid)][2] for haloid in halo_ids])
        hierarchy_masses = np.asarray(
            [list_properties[int(haloid)][3] for haloid in halo_ids]
        )

    if not box_sizes or not np.allclose(box_sizes, box_sizes[0]):
        raise CatalogueError("Rockstar binary files have inconsistent box sizes")
    all_memberships = inclusive_rockstar_memberships(
        halo_ids,
        centres,
        hierarchy_masses,
        radii,
        all_memberships,
        box_sizes[0],
    )
    counts = np.asarray([membership.size for membership in all_memberships], dtype=np.int64)
    packed = pack_memberships(halo_ids, counts, all_memberships)
    return packed[0], centres, masses, radii, packed[1], packed[2]


def read_rockstar_list_properties(
    files: Sequence[Path],
) -> dict[int, tuple[list[float], float, float, float]]:
    """Read standardized properties from companion out_<snapshot>.list files."""
    list_files: set[Path] = set()
    for filename in files:
        match = re.match(r"halos_(\d+)\.", filename.name)
        if match:
            candidate = filename.with_name(f"out_{match.group(1)}.list")
            if candidate.is_file():
                list_files.add(candidate)

    result: dict[int, tuple[list[float], float, float, float]] = {}
    for filename in sorted(list_files, key=natural_key):
        columns: dict[str, int] | None = None
        with filename.open("rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#ID "):
                    columns = {
                        name.lower(): index
                        for index, name in enumerate(stripped[1:].split())
                    }
                    continue
                if stripped.startswith("#"):
                    continue
                if columns is None or "id" not in columns:
                    raise CatalogueError(f"Rockstar list {filename} lacks an ID column")
                required = ("x", "y", "z")
                if any(name not in columns for name in required):
                    raise CatalogueError(f"Rockstar list {filename} lacks position columns")
                radius_name = next(
                    (name for name in ("r200c", "rvir") if name in columns), None
                )
                if radius_name is None:
                    raise CatalogueError(f"Rockstar list {filename} lacks R200c/Rvir")
                tokens = stripped.split()
                try:
                    haloid = int(
                        checked_int64(tokens[columns["id"]], f"{filename}:{line_number}")
                    )
                    m200c = (
                        float(tokens[columns["m200c"]])
                        if "m200c" in columns
                        else None
                    )
                    mvir = (
                        float(tokens[columns["mvir"]])
                        if "mvir" in columns
                        else None
                    )
                except IndexError as exc:
                    raise CatalogueError(
                        f"Malformed Rockstar list row {filename}:{line_number}"
                    ) from exc
                if m200c is not None and m200c > 0.0:
                    mass = m200c
                elif mvir is not None:
                    mass = mvir
                else:
                    raise CatalogueError(
                        f"Rockstar halo {haloid} has no positive M200c or Mvir"
                    )
                result[haloid] = (
                    [float(tokens[columns[name]]) * MPC_TO_KPC for name in required],
                    mass,
                    float(tokens[columns[radius_name]]),
                    float(mvir if mvir is not None else mass),
                )
    return result


def read_rockstar_particles(
    files: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read inclusive memberships grouped by external_haloid."""
    memberships: defaultdict[int, list[int]] = defaultdict(list)
    properties: dict[int, tuple[list[float], float, float, float]] = {}
    list_properties = read_rockstar_list_properties(files)

    for filename in files:
        in_halo_table = False
        in_particle_table = False
        halo_columns: dict[str, int] | None = None
        particle_columns: dict[str, int] | None = None
        with filename.open("rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#id "):
                    halo_columns = {
                        name.lower(): index
                        for index, name in enumerate(stripped[1:].split())
                    }
                    continue
                if stripped.startswith("#x "):
                    particle_columns = {
                        name.lower(): index
                        for index, name in enumerate(stripped[1:].split())
                    }
                    continue
                if stripped == "#Halo table begins here:":
                    in_halo_table = True
                    continue
                if stripped == "#Particle table begins here:":
                    in_halo_table = False
                    in_particle_table = True
                    continue

                if in_halo_table and stripped.startswith("#"):
                    required = ("id", "x", "y", "z", "m200c", "r200c")
                    if halo_columns is None or any(
                        name not in halo_columns for name in required
                    ):
                        raise CatalogueError(
                            f"Rockstar halo header in {filename} lacks required columns"
                        )
                    tokens = stripped[1:].split()
                    if len(tokens) < len(halo_columns):
                        raise CatalogueError(
                            f"Malformed Rockstar halo row {filename}:{line_number}"
                        )
                    haloid = int(
                        checked_int64(
                            tokens[halo_columns["id"]], f"{filename}:{line_number}"
                        )
                    )
                    if haloid < 0:
                        continue
                    if haloid in list_properties:
                        properties[haloid] = list_properties[haloid]
                        continue
                    m200c = float(tokens[halo_columns["m200c"]])
                    if m200c > 0.0:
                        mass = m200c
                    else:
                        raise CatalogueError(
                            f"Rockstar halo {haloid} has non-positive M200c and no Mvir"
                        )
                    properties[haloid] = (
                        [
                            float(tokens[halo_columns[name]]) * MPC_TO_KPC
                            for name in ("x", "y", "z")
                        ],
                        mass,
                        float(tokens[halo_columns["r200c"]]),
                        mass,
                    )
                    continue

                if not in_particle_table or stripped.startswith("#"):
                    continue
                required = ("particle_id", "external_haloid")
                if particle_columns is None or any(
                    name not in particle_columns for name in required
                ):
                    raise CatalogueError(
                        f"Rockstar particle header in {filename} lacks required columns"
                    )
                tokens = stripped.split()
                if len(tokens) < len(particle_columns):
                    raise CatalogueError(
                        f"Malformed Rockstar particle row {filename}:{line_number}"
                    )
                haloid = int(
                    checked_int64(
                        tokens[particle_columns["external_haloid"]],
                        f"{filename}:{line_number}",
                    )
                )
                if haloid >= 0:
                    memberships[haloid].append(
                        int(
                            checked_uint64(
                                tokens[particle_columns["particle_id"]],
                                f"{filename}:{line_number}",
                            )
                        )
                    )

    halo_ids = np.asarray(sorted(memberships), dtype=np.int64)
    missing = [int(haloid) for haloid in halo_ids if int(haloid) not in properties]
    if missing:
        raise CatalogueError(f"Rockstar halo properties missing for IDs: {missing[:10]}")
    arrays = [np.asarray(memberships[int(haloid)], dtype=np.uint64) for haloid in halo_ids]
    counts = np.asarray([array.size for array in arrays], dtype=np.int64)
    packed = pack_memberships(halo_ids, counts, arrays)
    centres = np.asarray([properties[int(haloid)][0] for haloid in halo_ids])
    masses = np.asarray([properties[int(haloid)][1] for haloid in halo_ids])
    radii = np.asarray([properties[int(haloid)][2] for haloid in halo_ids])
    return packed[0], centres, masses, radii, packed[1], packed[2]


def read_rockstar(
    spec: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[Path],
]:
    path = Path(spec).expanduser()
    if path.is_file() and path.name.endswith(".particles"):
        files = [path.resolve()]
    elif path.is_file() and path.suffix == ".bin":
        files = [path.resolve()]
    else:
        # A BGC2 ASCII path is accepted as a way of identifying the Rockstar
        # output set, but its SO spheres are not halo memberships.  Resolve the
        # corresponding .particles or .bin files in the same directory.
        lookup_spec = str(path.parent) if path.is_file() else spec
        try:
            candidates = expand_path_set(lookup_spec, suffixes=(".particles",))
        except CatalogueError:
            candidates = expand_path_set(lookup_spec, suffixes=(".bin",))
        files = [file for file in candidates if file.name.startswith("halos_")]
    if not files:
        raise CatalogueError(
            f"No Rockstar halos_*.particles or halos_*.bin files found for {spec}"
        )

    if all(file.suffix == ".particles" for file in files):
        result = read_rockstar_particles(files)
    elif all(file.suffix == ".bin" for file in files):
        result = read_rockstar_binary(files)
    else:
        raise CatalogueError("Do not mix Rockstar .particles and .bin inputs")
    return result + (files,)


def find_subfind_dataset(handle: h5py.File, names: Sequence[str]) -> h5py.Dataset:
    groups = ("Subhalo", "Subhalos")
    for group_name in groups:
        if group_name not in handle:
            continue
        group = handle[group_name]
        for name in names:
            if name in group:
                return group[name]
    raise CatalogueError(
        f"None of {', '.join(names)} found below Subhalo/Subhalos in {handle.filename}"
    )


def find_subfind_group_dataset(handle: h5py.File, names: Sequence[str]) -> h5py.Dataset:
    for group_name in ("Group", "Groups"):
        if group_name not in handle:
            continue
        group = handle[group_name]
        for name in names:
            if name in group:
                return group[name]
    raise CatalogueError(
        f"None of {', '.join(names)} found below Group/Groups in {handle.filename}"
    )


def resolve_subfind_files(spec: str) -> list[Path]:
    files = expand_path_set(spec, suffixes=(".hdf5", ".h5"))
    if len(files) == 1:
        match = re.match(r"^(.*)\.\d+\.(hdf5|h5)$", str(files[0]))
        if match:
            siblings = [
                Path(p) for p in glob.glob(f"{match.group(1)}.*.{match.group(2)}")
            ]
            files = unique_paths(p for p in siblings if p.is_file())
    files = [file for file in files if h5py.is_hdf5(file)]
    if not files:
        raise CatalogueError(f"No SUBFIND HDF5 catalogue files found for: {spec}")
    return files


def read_subfind(
    spec: str, snapshot_ids_by_type: dict[int, np.ndarray]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[Path],
]:
    files = resolve_subfind_files(spec)
    length_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = []
    centre_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []
    group_radius_parts: list[np.ndarray] = []
    halfmass_radius_parts: list[np.ndarray] = []

    for filename in files:
        with h5py.File(filename, "r") as handle:
            lengths = np.asarray(
                find_subfind_dataset(handle, ("SubhaloLenType", "LenType"))[...],
                dtype=np.int64,
            )
            offsets = np.asarray(
                find_subfind_dataset(handle, ("SubhaloOffsetType", "OffsetType"))[...],
                dtype=np.int64,
            )
            centres = np.asarray(
                find_subfind_dataset(handle, ("SubhaloPos", "Pos"))[...],
                dtype=np.float64,
            )
            masses = np.asarray(
                find_subfind_dataset(handle, ("SubhaloMass", "Mass"))[...],
                dtype=np.float64,
            )
            parents = np.asarray(
                find_subfind_dataset(handle, ("SubhaloGroupNr", "GroupNr"))[...],
                dtype=np.int64,
            )
            halfmass_radii = np.asarray(
                find_subfind_dataset(
                    handle, ("SubhaloHalfmassRad", "HalfmassRad")
                )[...],
                dtype=np.float64,
            )
            group_radii = np.asarray(
                find_subfind_group_dataset(handle, ("Group_R_Crit200",))[...],
                dtype=np.float64,
            )
            if lengths.ndim != 2 or offsets.ndim != 2 or lengths.shape != offsets.shape:
                raise CatalogueError(
                    f"SUBFIND length/offset arrays have incompatible shapes in {filename}"
                )
            nhalo = lengths.shape[0]
            if (
                centres.shape != (nhalo, 3)
                or masses.shape != (nhalo,)
                or parents.shape != (nhalo,)
                or halfmass_radii.shape != (nhalo,)
                or group_radii.ndim != 1
            ):
                raise CatalogueError(
                    f"SUBFIND halo-property arrays have incompatible shapes in {filename}"
                )
            length_parts.append(lengths)
            offset_parts.append(offsets)
            centre_parts.append(centres)
            mass_parts.append(masses)
            parent_parts.append(parents)
            group_radius_parts.append(group_radii)
            halfmass_radius_parts.append(halfmass_radii)

    lengths = np.concatenate(length_parts, axis=0)
    offsets_by_type = np.concatenate(offset_parts, axis=0)
    centres = np.concatenate(centre_parts, axis=0)
    masses = np.concatenate(mass_parts) * SUBFIND_MASS_TO_MSUN_H
    parents = np.concatenate(parent_parts)
    group_radii = np.concatenate(group_radius_parts)
    radii = np.concatenate(halfmass_radius_parts)
    if np.any(parents < 0) or np.any(parents >= group_radii.size):
        raise CatalogueError("SUBFIND SubhaloGroupNr references an absent FoF group")
    if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
        raise CatalogueError("SUBFIND SubhaloHalfmassRad is negative or non-finite")

    # SUBFIND orders the central subhalo first within each FoF group. Give that
    # main object the group's R200c aperture; retain each satellite's own
    # half-mass radius as its catalogue boundary.
    _, central_indices = np.unique(parents, return_index=True)
    radii[central_indices] = group_radii[parents[central_indices]]
    if np.any(lengths < 0) or np.any(offsets_by_type < 0):
        raise CatalogueError("SUBFIND contains negative lengths or offsets")

    ntypes = lengths.shape[1]
    memberships: list[np.ndarray] = []
    counts = lengths.sum(axis=1, dtype=np.int64)

    for halo_index in range(lengths.shape[0]):
        parts: list[np.ndarray] = []
        for ptype in range(ntypes):
            count = int(lengths[halo_index, ptype])
            if count == 0:
                continue
            if ptype not in snapshot_ids_by_type:
                raise CatalogueError(
                    f"SUBFIND subhalo {halo_index} references absent snapshot PartType{ptype}"
                )
            start = int(offsets_by_type[halo_index, ptype])
            stop = start + count
            source = snapshot_ids_by_type[ptype]
            if stop > source.size:
                raise CatalogueError(
                    f"SUBFIND subhalo {halo_index} PartType{ptype} range [{start}:{stop}] "
                    f"exceeds snapshot size {source.size}"
                )
            parts.append(source[start:stop])
        if parts:
            memberships.append(np.concatenate(parts) if len(parts) > 1 else parts[0].copy())
        else:
            memberships.append(np.empty(0, dtype=np.uint64))

    # Match the original analysis convention: the first (central) subhalo in
    # every FoF group with satellites carries the inclusive union of all
    # subhalo memberships in that group. Satellite memberships remain
    # exclusive. This supplies the complete central profile without requiring
    # SUBFIND hierarchy metadata downstream.
    group_members: defaultdict[int, list[int]] = defaultdict(list)
    for halo_index, group_index in enumerate(parents):
        group_members[int(group_index)].append(halo_index)
    for member_indices in group_members.values():
        if len(member_indices) < 2:
            continue
        parts = [memberships[index] for index in member_indices if memberships[index].size]
        if parts:
            memberships[member_indices[0]] = np.unique(
                np.concatenate(parts)
            ).astype(np.uint64, copy=False)

    halo_ids = np.arange(lengths.shape[0], dtype=np.int64)
    counts = np.asarray([membership.size for membership in memberships], dtype=np.int64)
    packed = pack_memberships(halo_ids, counts, memberships)
    return packed[0], centres, masses, radii, packed[1], packed[2], files


def pack_memberships(
    halo_ids: Sequence[int] | np.ndarray,
    counts: Sequence[int] | np.ndarray,
    memberships: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    halo_array = np.asarray(halo_ids, dtype=np.int64)
    count_array = np.asarray(counts, dtype=np.int64)
    if halo_array.ndim != 1 or count_array.ndim != 1:
        raise CatalogueError("Halo IDs and counts must be one-dimensional")
    if halo_array.size != count_array.size or halo_array.size != len(memberships):
        raise CatalogueError("Halo ID, count, and membership block counts disagree")
    if np.any(count_array < 0):
        raise CatalogueError("Negative membership count")

    offsets = np.empty(halo_array.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(count_array, out=offsets[1:])
    if offsets[-1] < 0:
        raise CatalogueError("Membership count overflow")

    particle_ids = np.empty(int(offsets[-1]), dtype=np.uint64)
    for index, (expected, values) in enumerate(zip(count_array, memberships)):
        array = np.asarray(values)
        if array.ndim != 1:
            raise CatalogueError(f"Membership block {index} is not one-dimensional")
        if array.size != expected:
            raise CatalogueError(
                f"Membership block {index} has {array.size} IDs; expected {expected}"
            )
        if array.dtype.kind not in "iu":
            raise CatalogueError(f"Membership block {index} is not integer-valued")
        if array.dtype.kind == "i" and np.any(array < 0):
            raise CatalogueError(f"Membership block {index} contains negative particle IDs")
        particle_ids[offsets[index] : offsets[index + 1]] = array.astype(
            np.uint64, copy=False
        )
    return halo_array, offsets, particle_ids


# NEW FINDER READER STUB
# ----------------------
# Copy this template, remove the leading comments, and implement catalogue
# parsing above ``pack_memberships`` (or alongside the other finder readers):
#
# def read_newfinder(
#     spec: str,
#     snapshot_ids_by_type: dict[int, np.ndarray],
# ) -> tuple[
#     np.ndarray,  # haloid             int64   [Nhalo]
#     np.ndarray,  # centre            float64 [Nhalo, 3], kpc/h
#     np.ndarray,  # catalogue_mass    float64 [Nhalo], Msun/h
#     np.ndarray,  # catalogue_radius  float64 [Nhalo], kpc/h
#     np.ndarray,  # offset            int64   [Nhalo + 1]
#     np.ndarray,  # particle_id       uint64  [Nmembership]
#     list[Path],  # resolved input catalogue files
# ]:
#     files = expand_path_set(spec, suffixes=(".newfinder",))
#     # Parse properties and build one uint64 membership array per halo.
#     # packed = pack_memberships(halo_ids, membership_counts, memberships)
#     # return packed[0], centres, masses, radii, packed[1], packed[2], files
#     raise NotImplementedError


def validate_halo_ids(halo_ids: np.ndarray) -> None:
    if halo_ids.ndim != 1:
        raise CatalogueError("haloid must be one-dimensional")
    if halo_ids.size < 2:
        return
    ordered = np.sort(halo_ids)
    duplicate_mask = ordered[1:] == ordered[:-1]
    if np.any(duplicate_mask):
        duplicates = np.unique(ordered[1:][duplicate_mask])
        preview = ", ".join(str(int(value)) for value in duplicates[:10])
        suffix = " ..." if duplicates.size > 10 else ""
        raise CatalogueError(f"Duplicate halo IDs: {preview}{suffix}")


def validate_halo_properties(
    halo_ids: np.ndarray,
    centres: np.ndarray,
    masses: np.ndarray,
    radii: np.ndarray,
) -> None:
    nhalo = halo_ids.size
    if centres.shape != (nhalo, 3):
        raise CatalogueError(f"centre has shape {centres.shape}; expected ({nhalo}, 3)")
    if masses.shape != (nhalo,):
        raise CatalogueError(f"catalogue_mass has shape {masses.shape}; expected ({nhalo},)")
    if radii.shape != (nhalo,):
        raise CatalogueError(
            f"catalogue_radius has shape {radii.shape}; expected ({nhalo},)"
        )
    if not np.all(np.isfinite(centres)):
        raise CatalogueError("centre contains non-finite values")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise CatalogueError("catalogue_mass contains negative or non-finite values")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise CatalogueError("catalogue_radius contains negative or non-finite values")


def validate_offsets(halo_ids: np.ndarray, offsets: np.ndarray, particle_ids: np.ndarray) -> None:
    if offsets.ndim != 1 or particle_ids.ndim != 1:
        raise CatalogueError("offset and particle_id must be one-dimensional")
    if offsets.size != halo_ids.size + 1:
        raise CatalogueError(
            f"offset has length {offsets.size}; expected {halo_ids.size + 1}"
        )
    if offsets.size == 0 or offsets[0] != 0:
        raise CatalogueError("offset[0] must be zero")
    if np.any(offsets < 0):
        raise CatalogueError("offset contains negative values")
    if np.any(offsets[1:] < offsets[:-1]):
        raise CatalogueError("offset is not monotonically non-decreasing")
    if int(offsets[-1]) != particle_ids.size:
        raise CatalogueError(
            f"offset[-1]={int(offsets[-1])} but particle_id has length {particle_ids.size}"
        )
    counts = np.diff(offsets)
    if np.any(counts < 0) or int(counts.sum(dtype=np.int64)) != particle_ids.size:
        raise CatalogueError("Offsets do not exactly partition particle_id")


def validate_particle_ids_exist(
    particle_ids: np.ndarray, sorted_snapshot_ids: np.ndarray, chunk_size: int = 5_000_000
) -> None:
    if sorted_snapshot_ids.size == 0 and particle_ids.size:
        raise CatalogueError("Snapshot contains no particle IDs")

    missing_examples: list[int] = []
    missing_count = 0
    for start in range(0, particle_ids.size, chunk_size):
        values = particle_ids[start : start + chunk_size]
        positions = np.searchsorted(sorted_snapshot_ids, values)
        in_range = positions < sorted_snapshot_ids.size
        matched = np.zeros(values.size, dtype=bool)
        matched[in_range] = sorted_snapshot_ids[positions[in_range]] == values[in_range]
        if np.all(matched):
            continue
        missing = values[~matched]
        missing_count += missing.size
        if len(missing_examples) < 10:
            missing_examples.extend(int(value) for value in missing[: 10 - len(missing_examples)])

    if missing_count:
        preview = ", ".join(str(value) for value in missing_examples)
        suffix = " ..." if missing_count > len(missing_examples) else ""
        raise CatalogueError(
            f"{missing_count} catalogue particle-ID entries do not exist in the snapshot: "
            f"{preview}{suffix}"
        )


def write_output(
    output: Path,
    halo_ids: np.ndarray,
    centres: np.ndarray,
    masses: np.ndarray,
    radii: np.ndarray,
    offsets: np.ndarray,
    particle_ids: np.ndarray,
    args: argparse.Namespace,
    finder: str,
    catalogue_files: Sequence[Path],
    snapshot_files: Sequence[Path],
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise CatalogueError(f"Output exists; use --force to replace it: {output}")

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    input_flags = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "force"} and value is not None
    }

    string_dtype = h5py.string_dtype(encoding="utf-8")

    try:
        with h5py.File(temporary, "w") as handle:
            header = handle.create_group("Header")
            header.attrs["command_line"] = " ".join(
                shlex.quote(arg) for arg in sys.argv
            )
            header.attrs["input_flags"] = json.dumps(input_flags, sort_keys=True)
            header.attrs["finder"] = finder
            header.attrs.create(
                "catalogue_files",
                [str(path) for path in catalogue_files],
                dtype=string_dtype,
            )
            header.attrs.create(
                "snapshot_files",
                [str(path) for path in snapshot_files],
                dtype=string_dtype,
            )
            haloes = handle.create_group("Haloes")
            haloes.create_dataset("haloid", data=halo_ids, dtype="i8")
            centre_dataset = haloes.create_dataset("centre", data=centres, dtype="f8")
            mass_dataset = haloes.create_dataset(
                "catalogue_mass", data=masses, dtype="f8"
            )
            radius_dataset = haloes.create_dataset(
                "catalogue_radius", data=radii, dtype="f8"
            )
            centre_dataset.attrs["units"] = "kpc/h"
            mass_dataset.attrs["units"] = "Msun/h"
            radius_dataset.attrs["units"] = "kpc/h"
            haloes.create_dataset("offset", data=offsets, dtype="i8")
            haloes.create_dataset("particle_id", data=particle_ids, dtype="u8")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    with h5py.File(output, "r") as handle:
        haloes = handle["Haloes"]
        written_halo_ids = np.asarray(haloes["haloid"][...], dtype=np.int64)
        written_centres = np.asarray(haloes["centre"][...], dtype=np.float64)
        written_masses = np.asarray(haloes["catalogue_mass"][...], dtype=np.float64)
        written_radii = np.asarray(haloes["catalogue_radius"][...], dtype=np.float64)
        written_offsets = np.asarray(haloes["offset"][...], dtype=np.int64)
        written_particle_ids = np.asarray(haloes["particle_id"][...], dtype=np.uint64)
        validate_halo_ids(written_halo_ids)
        validate_halo_properties(
            written_halo_ids, written_centres, written_masses, written_radii
        )
        validate_offsets(written_halo_ids, written_offsets, written_particle_ids)
        if not np.array_equal(written_halo_ids, halo_ids):
            raise CatalogueError("Written haloid dataset differs from input")
        if not np.array_equal(written_offsets, offsets):
            raise CatalogueError("Written offset dataset differs from input")
        if not np.array_equal(written_particle_ids, particle_ids):
            raise CatalogueError("Written particle_id dataset differs from input")
        if not np.array_equal(written_centres, centres):
            raise CatalogueError("Written centre dataset differs from input")
        if not np.array_equal(written_masses, masses):
            raise CatalogueError("Written catalogue_mass dataset differs from input")
        if not np.array_equal(written_radii, radii):
            raise CatalogueError("Written catalogue_radius dataset differs from input")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert halo-finder particle memberships to minimal HDF5."
    )
    finder = parser.add_mutually_exclusive_group(required=True)
    finder.add_argument("--ahf", metavar="CATALOGUE_SET")
    finder.add_argument("--rockstar", metavar="CATALOGUE_SET")
    finder.add_argument("--subfind", metavar="CATALOGUE_SET")
    # NEW FINDER CLI STUB:
    # finder.add_argument("--newfinder", metavar="CATALOGUE_SET")
    parser.add_argument("--snapshot", required=True, metavar="SNAPSHOT")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        snapshot_files = resolve_snapshot_files(args.snapshot)
        snapshot_ids_by_type = load_snapshot_ids_by_type(snapshot_files)

        if args.ahf is not None:
            finder = "ahf"
            (
                halo_ids,
                centres,
                masses,
                radii,
                offsets,
                particle_ids,
                catalogue_files,
            ) = read_ahf(args.ahf)
        elif args.rockstar is not None:
            finder = "rockstar"
            (
                halo_ids,
                centres,
                masses,
                radii,
                offsets,
                particle_ids,
                catalogue_files,
            ) = read_rockstar(args.rockstar)
        elif args.subfind is not None:
            finder = "subfind"
            (
                halo_ids,
                centres,
                masses,
                radii,
                offsets,
                particle_ids,
                catalogue_files,
            ) = read_subfind(args.subfind, snapshot_ids_by_type)
        # NEW FINDER DISPATCH STUB:
        # elif args.newfinder is not None:
        #     finder = "newfinder"
        #     (
        #         halo_ids,
        #         centres,
        #         masses,
        #         radii,
        #         offsets,
        #         particle_ids,
        #         catalogue_files,
        #     ) = read_newfinder(args.newfinder, snapshot_ids_by_type)
        else:
            raise AssertionError("argparse did not select a finder")

        validate_halo_ids(halo_ids)
        validate_halo_properties(halo_ids, centres, masses, radii)
        validate_offsets(halo_ids, offsets, particle_ids)
        sorted_snapshot_ids = flatten_snapshot_ids(snapshot_ids_by_type)
        validate_particle_ids_exist(particle_ids, sorted_snapshot_ids)
        write_output(
            args.output,
            halo_ids,
            centres,
            masses,
            radii,
            offsets,
            particle_ids,
            args,
            finder,
            catalogue_files,
            snapshot_files,
        )
    except (CatalogueError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"Wrote {args.output}: {halo_ids.size} halos, "
        f"{particle_ids.size} particle memberships"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
