"""Event processors for HPGe data."""

from __future__ import annotations

import json
from collections.abc import Sequence

from lgdo import types

from .. import utils
from . import cross_talk
from legendmeta import LegendMetadata


def read_ctx_matrix(xtalk_matrix_filename:str,positive_xtalk_matrix_filename:str,det_names:bool):
    """
    Function to read in and manipulate the cross talk matrix.
    Parameters
    ----------
    xtalk_matrix_filename (str)
         Path to the xtalk matrix
    positive_xtalk_matrix_filename (str)
        Path to the positive polarity cross talk matrix
    det_names
        bool to convert det names to rawids
     
    """
    # read in xtalk matrices (currently a json file)
    try:
        with open(xtalk_matrix_filename, 'r') as file:
                cross_talk_matrix = json.load(file)
    except FileNotFoundError:
         raise ValueError(f"path to x-talk matrix {xtalk_matrix_filename} does not exist")
    
    if (det_names==False):
        cross_talk_matrix=convert_matrix_det_names_to_rawid(cross_talk_matrix)

    # read the positive matrix
    if (positive_xtalk_matrix_filename is not None):
        try:
            with open(xtalk_matrix_filename, 'r') as file:
                positive_cross_talk_matrix = json.load(file)
        except FileNotFoundError:
            raise ValueError(f"path to x-talk matrix {positive_xtalk_matrix_filename} does not exist")

        if (det_names==False):
            positive_cross_talk_matrix=convert_matrix_det_names_to_rawid(positive_cross_talk_matrix)
        
        # merge +ive and -ive matrix
        matrix_merge={}
        for key_row,row in cross_talk_matrix.items():
            matrix_merge[key_row]={}
            for key_col,data in row.items():
                
                positive_ctx = positive_cross_talk_matrix[key_row][key_col]
                negative_ctx = data

                if (positive_ctx>negative_ctx):
                    matrix_merge[key_row][key_col]=-positive_ctx
        cross_talk_matrix=matrix_merge

    return cross_talk_matrix

def convert_matrix_det_names_to_rawid(matrix:dict)->dict:
    """
    Converts a cross talk matrix with keys of detector names to one with keys rawids
    Parameters
    ----------
    matrix
        dictonary of the cross talk matrix
    """
    metadb = LegendMetadata()
    chmap = metadb.channelmap("20230323T000000Z")

    geds_mapping = {
         _name :f"ch{_dict['daq']['rawid']}"
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    matrix_conv={}
    for key_row,row in matrix.items():
        matrix_conv[geds_mapping[key_row]]={}
        for key_col,data in row.items():
            matrix_conv[geds_mapping[key_row]][geds_mapping[key_col]]=data
    
    return matrix_conv


def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    energy_observable: types.VectorOfVectors,
    rawids: types.VectorOfVectors,
    xtalk_matrix_filename: str,
    threshold: float,
    det_names: bool=False,
    energy_observable_negative: types.VectorOfVectors=None,
    postive_xtalk_matrix_filename: str=None

) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a JSON file.

    The correction is appplied recursively, the energies are sorted and then the correction
    is applied from the largest to smallest (above the threshold).

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
    threshold
        threshold used for cross talk correction, hits below this energy will not
        be used to correct the other hits.
    det_names
        bool to say the x-talk matrices use the detector names (default False)
    energy_observable_negative
        array of negative energy values to correct, one event per row. The detector
        identifier is stored in `rawids`, which has the same layout. Default None
    positive_xtalk_matrix_filename
        name of the file containing the positive cross-talk matrices.

    """
    
    cross_talk_matrix=read_ctx_matrix(xtalk_matrix_filename,postive_xtalk_matrix_filename,det_names)

    # do the correction
    energies_corr = cross_talk.cross_talk_corrected_energy_awkard_slow(energies=energy_observable.view_as("ak"),
                                                                       rawids=rawids.view_as("ak"),
                                                                       matrix=cross_talk_matrix,
                                                                       allow_non_existing=False,
                                                                       threshold=threshold
                                                                       )

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(energy_observable)
    )
