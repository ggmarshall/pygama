"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

from lgdo import lh5, types

from .. import utils
from . import cross_talk
import json

def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    energy_observable: types.VectorOfVectors,
    rawids: types.VectorOfVectors,
    xtalk_matrix_filename: str,
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.

    The format of `xtalk_matrix_filename` should be...

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    energy_observable
        array of energy values to correct, one event per row. The detector
        identifier is stored in `rawids`, which has the same layout.
    rawids
        array of detector identifiers for each energy in `energy_observable`.
    xtalk_matrix_filename
        name of the file containing the cross-talk matrices.
    """
    # read in xtalk matrices (currently a json file)

    with open(xtalk_matrix_filename, 'r') as file:
            cross_talk_matrix = json.load(file)

    # do the correction
    energies_corr = cross_talk.cross_talk_corrected_energy_awkard_slow(energies=energy_observable,
                                                                       rawids=rawids,
                                                                       matrix=cross_talk_matrix,
                                                                       allow_non_existing=False
                                                                       )

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(energy_observable)
    )
