"""
Module for cross talk correction of energies.
"""

import awkward as ak
import numpy as np


def cross_talk_corrected_energy_awkard_slow(
    energies: ak.Array,
    rawids: ak.Array,
    matrix: dict,
    allow_non_existing: bool = True,
    threshold: float = None,
):
    """
    Function to perform the cross talk correction on awkward arrays of energy and rawid.
    The energies are first sorted from largest to smallest, a term is then added to the
        other energies of the cross talk matrix element multipled by the largest energy.
    .. math::
        E_{i,\text{cor}} = E_{1}\times M[c_i,c_1]+E_{i},
        where $c_i$ is the raw-id of the $i$th energy.

    This process is repeated recursively for the 2nd largest energy etc.
    This implementation is called 'slow' since it uses loops over events and energies

    Parameters
        - energies (ak.Array): array of energies
        - rawids   (ak.Array): array of rawids
        - matrix (dict)      : python dictonary of the cross talk correction matrix
        - allow_non_existing (bool): A boolean to control what happens if a rawid is not present
            in the matrix, if True, this matrix element is set to 0, if False an exception is raised.
        - threshold (float)  : energy threshold below which hits are not used to correct the other hits
    Returns:
        ak.Array of corrected energies

    """

    # some exceptions
    # check types
    if not isinstance(energies, ak.Array):
        raise TypeError("energies must be an awkward array")

    if not isinstance(rawids, ak.Array):
        raise TypeError("rawids must be an awkward array")

    if not isinstance(matrix, dict):
        raise TypeError("matrix must be a python dictonary")

    if not isinstance(allow_non_existing, bool):
        raise TypeError("allow_non_existing must be a Boolean")

    # first check that energies and rawids have the same dimensions
    if ak.any(ak.num(energies, axis=-1) != ak.num(rawids, axis=-1)):
        raise ValueError(
            "Error: the length of each subarray of energies and rawids must be equal"
        )

    if ak.num(energies, axis=-2) != ak.num(rawids, axis=-2):
        raise ValueError("Error: the number of energies is not equal to rawids")

    # check that the matrix elements exist
    for c1 in np.unique(ak.flatten(rawids).to_numpy()):
        if c1 not in matrix.keys():

            if allow_non_existing == True:
                matrix[c1] = {}
            else:
                raise ValueError(
                    f"Error allow_non_existing is set to False and {c1} isnt present in the matrix"
                )

        for c2 in np.unique(ak.flatten(rawids).to_numpy()):
            if c1 == c2:
                continue
            else:
                if c2 not in matrix[c1].keys():
                    if allow_non_existing == True:
                        matrix[c1][c2] = 0
                    else:
                        raise ValueError(
                            f"Error allow_non_existing is set to False and {c2} isnt present in the matrix[{c1}]"
                        )

    ## add a check that the matrix is symmetric

    for c1 in matrix.keys():
        for c2 in matrix[c1].keys():
            if abs(matrix[c1][c2] - matrix[c2][c1]) > 1e-6:
                raise ValueError(
                    f"Error input cross talk matrix is not symmetric for {c1},{c2}"
                )

    ## sort the energies and rawids
    args = ak.argsort(energies, ascending=False)
    energies = energies[args]
    rawids = rawids[args]

    ## run the correction
    ## --------------------

    energies_corrected = []

    ## we should try to speed this up
    for energy_vec_tmp, rawid_vec_tmp in zip(energies, rawids):

        energies_corrected_tmp = list(energy_vec_tmp)
        for id_main, (energy_main, rawid_main) in enumerate(
            zip(energy_vec_tmp, rawid_vec_tmp)
        ):
            if threshold is not None and energy_main < threshold:
                break
            for id_other, (energy_other, rawid_other) in enumerate(
                zip(energy_vec_tmp, rawid_vec_tmp)
            ):
                if id_main != id_other:
                    energies_corrected_tmp[id_other] += (
                        matrix[rawid_main][rawid_other] * energy_main
                    )

        energies_corrected.append(energies_corrected_tmp)

    ## convert to awkward array and unsort
    return ak.Array(energies_corrected)[args]
