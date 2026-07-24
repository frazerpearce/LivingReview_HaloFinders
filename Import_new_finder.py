#!/usr/bin/env python3
"""Read one new halo finder's native output.

This is the only file that normally needs finder-specific catalogue-reading
code. Do not create the project's final HDF5 layout here. Return the simple
arrays and membership lists described below; convert_halo_membership.py will
check them and write the standardized file.
"""

from pathlib import Path

import numpy as np


# EDIT THIS: use the same short, lowercase key in finder_config.py.
FINDER_KEY = "newfinder"


def import_new_finder(catalogue_path):
    """Read native finder output and return five required objects.

    Parameters
    ----------
    catalogue_path : str
        File, directory, prefix, or other location passed to --new-finder.

    Returns
    -------
    halo_ids
        One-dimensional array containing one integer ID for every halo.
    centres
        Array with shape (Nhalo, 3). Coordinates must be in kpc/h.
    masses
        One-dimensional array of halo masses in Msun/h.
    radii
        One-dimensional array of halo radii in kpc/h.
    memberships
        A Python list with one entry per halo. memberships[i] must contain the
        original snapshot particle IDs assigned to halo i.

    The five objects must have the same halo ordering. In particular,
    halo_ids[i], centres[i], masses[i], radii[i], and memberships[i] must all
    describe the same halo.
    """
    path = Path(catalogue_path).expanduser()

    # =====================================================================
    # BEGIN FINDER-SPECIFIC IMPORT CODE
    # =====================================================================
    #
    # Replace this block with basic code that reads your finder's files.
    # For example:
    #
    # halo_table = np.loadtxt(path / "haloes.txt")
    # halo_ids = halo_table[:, 0]
    # centres = halo_table[:, 1:4]
    # masses = halo_table[:, 4]
    # radii = halo_table[:, 5]
    #
    # memberships = []
    # for halo_id in halo_ids:
    #     member_file = path / ("members_%d.txt" % int(halo_id))
    #     particle_ids = np.loadtxt(member_file, dtype=np.uint64, ndmin=1)
    #     memberships.append(particle_ids)
    #
    # Delete the following exception after implementing the reader.
    raise NotImplementedError(
        "Add the finder-specific file reader to Import_new_finder.py"
    )
    #
    # =====================================================================
    # END FINDER-SPECIFIC IMPORT CODE
    # =====================================================================

    # Convert ordinary Python lists to arrays. The main converter performs
    # the detailed shape, type, membership, and snapshot-ID validation.
    halo_ids = np.asarray(halo_ids)
    centres = np.asarray(centres)
    masses = np.asarray(masses)
    radii = np.asarray(radii)
    memberships = [np.asarray(member_ids) for member_ids in memberships]

    return halo_ids, centres, masses, radii, memberships
