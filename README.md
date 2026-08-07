# LivingReview Halo-Finder Tests

This repository provides five common tests for comparing halo finders:

1. **King unit test** — recovery of two isolated, equal-mass King-model haloes.
2. **King infall test** — recovery of lower-mass King haloes at different radii
   from a central host, compared particle by particle with their isolated
   counterparts.
3. **Major-merger test** — recovery of two equal-mass King haloes through a
   controlled sequence of decreasing separations.
4. **Minor-merger test** — recovery of a 1:100 King satellite as it approaches
   and overlaps a primary King halo.
5. **CosmoBox test** — comparison of halo and subhalo populations in a
   cosmological dark-matter simulation.

The aim is not to require every halo finder to expose the same native catalogue
format. Instead, each finder is run normally and its output is converted to the
small common HDF5 format described below. The analysis scripts consume only
that common format.

## What is supplied

The two compact seed snapshots are included:

```text
Simulation/data/unit_test_king_box.hdf5
Simulation/data/king_infall.hdf5
```

Common-format results from AHF, SUBFIND, and ROCKSTAR are included in
`LRdata/` for comparison.

The major- and minor-merger generators use these seed snapshots to build two
eleven-snapshot controlled sequences under `Simulation/Major_Merger_ICs/` and
`Simulation/Minor_Merger_ICs/`. These configurations scan halo separation;
they are not successive outputs of a dynamical time integration.

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

All five tests use Gadget-style HDF5 snapshots. Dark-matter particles are in
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

All finder-specific file-reading code belongs in `Import_new_finder.py`. The
main converter should not need to be edited.

### 1. Set the finder key

At the top of `Import_new_finder.py`, replace:

```python
FINDER_KEY = "newfinder"
```

with a short, lowercase identifier for the finder. Use exactly the same key in
`finder_config.py`.

### 2. Implement the native-file reader

Edit only the clearly marked finder-specific block inside
`import_new_finder()`. It must construct and return five objects:

```python
return halo_ids, centres, masses, radii, memberships
```

These are:

```text
halo_ids       one-dimensional array with one integer ID per halo
centres        array [Nhalo, 3], in h^-1 kpc
masses         one-dimensional array [Nhalo], in h^-1 Msun
radii          one-dimensional array [Nhalo], in h^-1 kpc
memberships    Python list of length Nhalo
```

`memberships[i]` is a one-dimensional list or array containing the original
snapshot particle IDs assigned to halo `i`. All five objects must use the same
halo ordering.

An intentionally simple `numpy.loadtxt` example is included in the marked
block. Replace it with the code needed for the new finder's files and delete
the placeholder `NotImplementedError`. Do not construct HDF5 offsets or packed
membership arrays in this file. `convert_halo_membership.py` does that and
performs all shape, type, offset, unit-layout, and snapshot-ID validation.

### 3. Register its name and plotting style

Add an entry to `FINDERS` in `finder_config.py`:

```python
"newfinder": {
    "label": "NEWFINDER",
    "color": "C3",
    "linestyle": "-.",
    "derived": "LRdata/cosmological_newfinder_128_derived.txt",
    "catalogue": "LRdata/cosmological_newfinder_128.hdf5",
    "king_catalogue": "LRdata/unit_test_king_newfinder.hdf5",
    "infall_catalogue": "LRdata/king_infall_newfinder.hdf5",
},
```

Dictionary order controls plot order. Do not change `REFERENCE_FINDER` unless
you deliberately want another finder to seed the cosmological cross-matching.
`analyze_king.py`, `analyze_king_infall.py`, and `analyze_cosmo.py` obtain their
default catalogue paths from `king_catalogue`, `infall_catalogue`, and
`catalogue`, respectively.

## Test 1: King unit test

Run your finder on:

```text
Simulation/data/unit_test_king_box.hdf5
```

Convert its native catalogue:

```bash
python convert_halo_membership.py \
  --new-finder path/to/unit_test/native_catalogue \
  --snapshot Simulation/data/unit_test_king_box.hdf5 \
  --output LRdata/unit_test_king_newfinder.hdf5
```

If an output already exists and should be replaced, add `--force`.

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
  --new-finder path/to/king_infall/native_catalogue \
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

## Test 3: Controlled major-merger test

Generate the eleven equal-mass configurations with:

```bash
python generate_major_merger.py
```

This places two identical King haloes in a cropped
$5\,h^{-1}{\rm Mpc}$ box. Their centre separation decreases from
$1000$ to $0\,h^{-1}{\rm kpc}$ in $100\,h^{-1}{\rm kpc}$ steps while their
internal particle realizations and persistent IDs remain fixed. Run the same
finder configuration, as far as practical, on:

```text
Simulation/Major_Merger_ICs/major_merger_000.hdf5
...
Simulation/Major_Merger_ICs/major_merger_010.hdf5
```

Convert every native catalogue using its corresponding snapshot. Use the
following common-output naming convention:

```text
LRdata/Major_Merger/major_merger_<snapshot>_<finder>.hdf5
```

For example:

```bash
python convert_halo_membership.py \
  --new-finder path/to/major_merger_000/native_catalogue \
  --snapshot Simulation/Major_Merger_ICs/major_merger_000.hdf5 \
  --output LRdata/Major_Merger/major_merger_000_newfinder.hdf5
```

After all eleven catalogues have been converted, add the new finder key to
`FINDER_ORDER` in `analyze_major_merger.py`;
`analyze_minor_merger.py` imports and uses the same list. The corresponding
label and colour continue to come from `finder_config.py`. Then run:

```bash
python analyze_major_merger.py \
  --snapshot-dir Simulation/Major_Merger_ICs \
  --catalogue-dir LRdata/Major_Merger
```

The analysis assigns distinct recovered objects to the two injected
progenitors by maximising particle-ID overlap at each separation. It plots the
recovered centres and catalogue radii and annotates each recovered mass relative
to that progenitor's well-separated value. The figure is written to
`output_major_merger/major_merger_recovered_halo_sequence.png`.

## Test 4: Controlled minor-merger test

First generate the major-merger sequence and the King-infall seed snapshot,
then create the 1:100 sequence with:

```bash
python generate_minor_merger.py
```

The eleven configurations combine the primary King halo with a self-similar
satellite containing one hundredth of its mass. Their centre separation
decreases from $500$ to $0\,h^{-1}{\rm kpc}$ in
$50\,h^{-1}{\rm kpc}$ steps. The common particle mass means that the satellite
contains one hundredth as many particles as the primary. Run the finder on:

```text
Simulation/Minor_Merger_ICs/minor_merger_000.hdf5
...
Simulation/Minor_Merger_ICs/minor_merger_010.hdf5
```

Convert every result using the corresponding snapshot and write it as:

```text
LRdata/Minor_Merger/minor_merger_<snapshot>_<finder>.hdf5
```

For example:

```bash
python convert_halo_membership.py \
  --new-finder path/to/minor_merger_000/native_catalogue \
  --snapshot Simulation/Minor_Merger_ICs/minor_merger_000.hdf5 \
  --output LRdata/Minor_Merger/minor_merger_000_newfinder.hdf5
```

Analyse all eleven common catalogues with:

```bash
python analyze_minor_merger.py \
  --snapshot-dir Simulation/Minor_Merger_ICs \
  --catalogue-dir LRdata/Minor_Merger
```

The matching again uses the exact injected particle-ID sets, now distinguishing
the primary and 1:100 satellite. The resulting sequence is written to
`output_minor_merger/minor_merger_recovered_halo_sequence.png`.

## Test 5: CosmoBox

First obtain the CosmoBox snapshot and existing comparison outputs from
[frazer.pearce@nottingham.ac.uk](mailto:frazer.pearce@nottingham.ac.uk). Place
them at the paths supplied with the data, or pass explicit paths in the commands
below.

Run your finder on the supplied Gadget HDF5 snapshot and convert the result:

```bash
python convert_halo_membership.py \
  --new-finder path/to/cosmological/native_catalogue \
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

Please return the standardized HDF5 files produced by the converter:

```text
LRdata/unit_test_king_<finder>.hdf5
LRdata/king_infall_<finder>.hdf5
LRdata/Major_Merger/major_merger_000_<finder>.hdf5 ... major_merger_010_<finder>.hdf5
LRdata/Minor_Merger/minor_merger_000_<finder>.hdf5 ... minor_merger_010_<finder>.hdf5
LRdata/cosmological_<finder>_128.hdf5
```

These files are the primary test deliverables. They contain the catalogue
properties and particle memberships needed to reproduce all comparisons.

Please also supply the finder name and version, the configuration/parameter file used
for each test, and any preprocessing needed to run the finder together with a link to the downloadable source if this exists. Native catalogues
and generated figures are useful for diagnosis but are not substitutes for the
standardized HDF5 files.

Before returning the files, verify that all five analysis commands complete
without errors and inspect the generated summaries and figures for obviously
incorrect units, missing memberships, or duplicate objects.
