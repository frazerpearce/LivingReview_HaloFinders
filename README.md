# LivingReview Halo-Finder Tests

This repository provides three common tests for comparing halo finders:

1. **King unit test** — recovery of two isolated, equal-mass King-model haloes.
2. **King infall test** — recovery of lower-mass King haloes at different radii
   from a central host, compared particle by particle with their isolated
   counterparts.
3. **CosmoBox test** — comparison of halo and subhalo populations in a
   cosmological dark-matter simulation.

The aim is not to require every halo finder to expose the same native catalogue
format. Instead, each finder is run normally and its output is converted to the
small common HDF5 format described below. The analysis scripts consume only
that common format.

## What is supplied

The two compact test snapshots are included:

```text
Simulation/data/unit_test_king_box.hdf5
Simulation/data/king_infall.hdf5
```

Common-format results from AHF, SUBFIND, and ROCKSTAR are included in
`LRdata/` for comparison.

The CosmoBox initial conditions/snapshot and the corresponding comparison
catalogues are too large for GitHub. Request them from
[frazer.pearce@nottingham.ac.uk](mailto:frazer.pearce@nottingham.ac.uk).
The cosmological outputs from the other finders are supplied with the snapshot
data so that a new finder can be added directly to the comparison figures.

## Requirements

Python 3 with:

```bash
python -m pip install numpy scipy h5py matplotlib
```

Run all commands below from the repository root.

## Input snapshot format

All three tests use Gadget-style HDF5 snapshots. Dark-matter particles are in
`/PartType1`:

```text
/Header
    attrs: BoxSize, NumPart_ThisFile, NumPart_Total, MassTable, ...

/PartType1/Coordinates    [N, 3]
/PartType1/Velocities     [N, 3]
/PartType1/ParticleIDs    [N]
/PartType1/Masses         [N]
```

The supplied King snapshots use positions and `BoxSize` in
`h^{-1} kpc`, velocities in `km s^{-1}`, and explicit particle masses. Particle
IDs are the persistent identifiers used to compare memberships; a converter
must preserve them exactly. The CosmoBox data supplied separately use the same
Gadget HDF5 conventions.

Your finder may require a different input representation. It is fine to
translate the snapshot for the finder, provided that its final membership lists
can be mapped back to the original Gadget `ParticleIDs`.

## Common output format

Run `convert_halo_membership.py` on each native finder catalogue. It writes:

```text
/Header
    attrs:
        finder
        catalogue_files
        snapshot_files
        command_line
        input_flags

/Haloes/haloid             int64    [Nhalo]
/Haloes/centre             float64  [Nhalo, 3]       h^-1 kpc
/Haloes/catalogue_mass     float64  [Nhalo]          h^-1 Msun
/Haloes/catalogue_radius   float64  [Nhalo]          h^-1 kpc
/Haloes/offset             int64    [Nhalo + 1]
/Haloes/particle_id        uint64   [Nmembership]
```

Memberships use a packed/CSR layout. The particle IDs belonging to halo `i`
are:

```python
particle_id[offset[i]:offset[i + 1]]
```

`offset[0]` must be zero and `offset[-1]` must equal the length of
`particle_id`. Every membership ID must occur in the input snapshot. Return the
finder's intended halo membership lists (including subhalo members where that
is the native catalogue convention), rather than constructing new spherical
particle selections in the analysis scripts.

Catalogue centres and radii must be converted to `h^{-1} kpc`, and catalogue
masses to `h^{-1} Msun`, before being returned by the reader. The converter
validates the array shapes, offsets, unique halo IDs, membership-ID range, and
presence of every membership ID in the snapshot before writing the result.

## Adding another finder

`convert_halo_membership.py` contains three marked extension points. Search for
`NEW FINDER` to find them.

### 1. Implement the reader

Copy the `read_newfinder()` template below the `NEW FINDER READER STUB`. Parse
the native halo properties and one particle-ID array per halo, then return:

```python
(
    haloid,             # int64   [Nhalo]
    centre,             # float64 [Nhalo, 3], h^-1 kpc
    catalogue_mass,     # float64 [Nhalo], h^-1 Msun
    catalogue_radius,   # float64 [Nhalo], h^-1 kpc
    offset,             # int64   [Nhalo + 1]
    particle_id,        # uint64  [Nmembership]
    catalogue_files,    # resolved native input files
)
```

The supplied `pack_memberships()` helper builds `offset` and `particle_id`
from per-halo membership arrays and checks their lengths.

### 2. Add the command-line option

At `NEW FINDER CLI STUB`, add a mutually exclusive option:

```python
finder.add_argument("--newfinder", metavar="CATALOGUE_SET")
```

Use a short, lowercase key in place of `newfinder`.

### 3. Add the dispatch branch

At `NEW FINDER DISPATCH STUB`, enable the template branch and call the reader.
Set `finder` to exactly the same lowercase key used on the command line and in
`finder_config.py`.

### 4. Register its name and plotting style

Add an entry to `FINDERS` in `finder_config.py`:

```python
"newfinder": {
    "label": "NEWFINDER",
    "color": "C3",
    "linestyle": "-.",
    "derived": "LRdata/cosmological_newfinder_128_derived.txt",
    "catalogue": "LRdata/cosmological_newfinder_128.hdf5",
    "king_catalogue": "LRdata/unit_test_king_newfinder.hdf5",
},
```

Dictionary order controls plot order. Do not change `REFERENCE_FINDER` unless
you deliberately want another finder to seed the cosmological cross-matching.
The King-infall analysis automatically looks for
`LRdata/king_infall_<finder>.hdf5`.

## Test 1: King unit test

Run your finder on:

```text
Simulation/data/unit_test_king_box.hdf5
```

Convert its native catalogue:

```bash
python convert_halo_membership.py \
  --newfinder path/to/unit_test/native_catalogue \
  --snapshot Simulation/data/unit_test_king_box.hdf5 \
  --output LRdata/unit_test_king_newfinder.hdf5
```

Replace `--newfinder` with the option added for your reader. If an output
already exists and should be replaced, add `--force`.

After registering the finder in `finder_config.py`, create the comparison
figures and summary with:

```bash
python analyze_king.py
```

Outputs are written to `output_king/`. The analysis compares recovered centres,
particle membership, density profiles, circular-velocity profiles,
`r(Vmax)`, and `Vmax` with the injected King haloes.

## Test 2: King infall test

Run the same finder configuration, as far as practical, on:

```text
Simulation/data/king_infall.hdf5
```

Convert its output:

```bash
python convert_halo_membership.py \
  --newfinder path/to/king_infall/native_catalogue \
  --snapshot Simulation/data/king_infall.hdf5 \
  --output LRdata/king_infall_newfinder.hdf5
```

Then run:

```bash
python analyze_king_infall.py
```

Outputs are written to `output_king_infall/`. This test matches objects using
particle-ID overlap and reports how satellite recovery changes with host-centric
radius relative to the corresponding isolated King model.

## Test 3: CosmoBox

First obtain the CosmoBox snapshot and existing comparison outputs from
[frazer.pearce@nottingham.ac.uk](mailto:frazer.pearce@nottingham.ac.uk). Place
them at the paths supplied with the data, or pass explicit paths in the commands
below.

Run your finder on the supplied Gadget HDF5 snapshot and convert the result:

```bash
python convert_halo_membership.py \
  --newfinder path/to/cosmological/native_catalogue \
  --snapshot Simulation/data/snap_128.hdf5 \
  --output LRdata/cosmological_newfinder_128.hdf5
```

Next calculate the particle-derived quantities required by the cosmological
analysis:

```bash
python LR_base_analysis.py \
  LRdata/cosmological_newfinder_128.hdf5 \
  Simulation/data/snap_128.hdf5 \
  --output LRdata/cosmological_newfinder_128_derived.txt \
  --no-plots
```

After adding the paths to `finder_config.py`, generate the full comparison:

```bash
python analyze_cosmo.py --no-show
```

Figures are written to `output_cosmo/`. The analysis automatically includes
every finder listed in `finder_config.py`, using the common memberships and the
derived text catalogue. It produces mass and `Vmax` functions, spatial and
radial comparisons, and matched-member plots. The supplied snapshot package
contains the other finders' cosmological outputs needed for these comparisons.

## What to return

Please return the three standardized HDF5 files produced by the converter:

```text
LRdata/unit_test_king_<finder>.hdf5
LRdata/king_infall_<finder>.hdf5
LRdata/cosmological_<finder>_128.hdf5
```

These files are the primary test deliverables. They contain the catalogue
properties and particle memberships needed to reproduce all comparisons.

Please also supply the finder name and version, the configuration/parameter file used
for each test, and any preprocessing needed to run the finder together with a link to the downloadable source if this exists. Native catalogues
and generated figures are useful for diagnosis but are not substitutes for the
three standardized HDF5 files.

Before returning the files, verify that all three analysis commands complete
without errors and inspect the generated summaries and figures for obviously
incorrect units, missing memberships, or duplicate objects.

