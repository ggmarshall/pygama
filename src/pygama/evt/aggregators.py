"""
This module provides aggregators to build the `evt` tier.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, VectorOfVectors, lh5
from lgdo.lh5 import LH5Store
from numpy.typing import NDArray

from . import utils


def evaluate_to_first_or_last(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    channels: list,
    channels_rm: list,
    expr: str,
    exprl: list,
    query: str | NDArray,
    n_rows: int,
    sorter: tuple,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
    is_first: bool = True,
) -> Array:
    """Aggregates across channels by returning the expression of the channel
    with value of `sorter`.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output array.
    sorter
       tuple of field in `hit/dsp/evt` tier to evaluate ``(tier, field)``.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    is_first
       defines if sorted by smallest or largest value of `sorter`
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    # define dimension of output array
    out = np.full(n_rows, default_value, dtype=type(default_value))
    outt = np.zeros(len(out))

    store = LH5Store()

    for ch in channels:
        # get index list for this channel to be loaded
        idx_ch = tcm.idx[tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch))[0],
            "right",
        )

        # evaluate at channel
        res = utils.get_data_at_channel(
            datainfo=datainfo,
            ch=ch,
            tcm=tcm,
            expr=expr,
            exprl=exprl,
            pars_dict=pars_dict,
            is_evaluated=ch not in channels_rm,
            default_value=default_value,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        # find if sorter is in hit or dsp
        t0 = store.read(
            f"{ch}/{sorter[0]}/{sorter[1]}",
            f.hit.file if f"{f.hit.group}" == sorter[0] else f.dsp.file,
            idx=idx_ch,
        )[0].view_as("np")

        if t0.ndim > 1:
            raise ValueError(f"sorter '{sorter[0]}/{sorter[1]}' must be a 1D array")

        if is_first:
            if ch == channels[0]:
                outt[:] = np.inf

            out[evt_ids_ch] = np.where(
                (t0 < outt[evt_ids_ch]) & (limarr), res, out[evt_ids_ch]
            )
            outt[evt_ids_ch] = np.where(
                (t0 < outt[evt_ids_ch]) & (limarr), t0, outt[evt_ids_ch]
            )

        else:
            out[evt_ids_ch] = np.where(
                (t0 > outt[evt_ids_ch]) & (limarr), res, out[evt_ids_ch]
            )
            outt[evt_ids_ch] = np.where(
                (t0 > outt[evt_ids_ch]) & (limarr), t0, outt[evt_ids_ch]
            )

    return Array(nda=out, dtype=type(default_value))


def evaluate_to_scalar(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    mode: str,
    channels: list,
    channels_rm: list,
    expr: str,
    exprl: list,
    query: str | NDArray,
    n_rows: int,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
) -> Array:
    """Aggregates by summation across channels.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    mode
       aggregation mode.
    channels
       list of channels to be aggregated.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output array
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    # define dimension of output array
    out = np.full(n_rows, default_value, dtype=type(default_value))

    for ch in channels:
        # get index list for this channel to be loaded
        idx_ch = tcm.idx[tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch))[0],
            "right",
        )

        res = utils.get_data_at_channel(
            datainfo=datainfo,
            ch=ch,
            tcm=tcm,
            expr=expr,
            exprl=exprl,
            pars_dict=pars_dict,
            is_evaluated=ch not in channels_rm,
            default_value=default_value,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        # switch through modes
        if "sum" == mode:
            if res.dtype == bool:
                res = res.astype(int)
            out[evt_ids_ch] = np.where(limarr, res + out[evt_ids_ch], out[evt_ids_ch])
        if "any" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[evt_ids_ch] = out[evt_ids_ch] | (res & limarr)
        if "all" == mode:
            if res.dtype != bool:
                res = res.astype(bool)
            out[evt_ids_ch] = out[evt_ids_ch] & res & limarr

    return Array(nda=out, dtype=type(default_value))


def evaluate_at_channel(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    channels_rm: list,
    expr: str,
    exprl: list,
    ch_comp: Array,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
) -> Array:
    """Aggregates by evaluating the expression at a given channel.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of rawids at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    out = np.full(len(ch_comp.nda), default_value, dtype=type(default_value))

    for ch in np.unique(ch_comp.nda.astype(int)):
        # skip default value
        if utils.get_table_name_by_pattern(table_id_fmt, ch) not in lh5.ls(f.hit.file):
            continue
        idx_ch = tcm.idx[tcm.id == ch]
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length, np.where(tcm.id == ch)[0], "right"
        )
        res = utils.get_data_at_channel(
            datainfo=datainfo,
            ch=utils.get_table_name_by_pattern(table_id_fmt, ch),
            tcm=tcm,
            expr=expr,
            exprl=exprl,
            pars_dict=pars_dict,
            is_evaluated=utils.get_table_name_by_pattern(table_id_fmt, ch)
            not in channels_rm,
            default_value=default_value,
        )

        out[evt_ids_ch] = np.where(ch == ch_comp.nda[idx_ch], res, out[evt_ids_ch])

    return Array(nda=out, dtype=type(default_value))


def evaluate_at_channel_vov(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    expr: str,
    exprl: list,
    ch_comp: VectorOfVectors,
    channels_rm: list,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
) -> VectorOfVectors:
    """Same as :func:`evaluate_at_channel` but evaluates expression at non
    flat channels :class:`.VectorOfVectors`.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    # blow up vov to aoesa
    out = ak.Array([[] for _ in range(len(ch_comp))])

    channels = np.unique(ch_comp.flattened_data.nda).astype(int)
    ch_comp = ch_comp.view_as("ak")

    type_name = None
    for ch in channels:
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length, np.where(tcm.id == ch)[0], "right"
        )
        res = utils.get_data_at_channel(
            datainfo=datainfo,
            ch=utils.get_table_name_by_pattern(table_id_fmt, ch),
            tcm=tcm,
            expr=expr,
            exprl=exprl,
            pars_dict=pars_dict,
            is_evaluated=utils.get_table_name_by_pattern(table_id_fmt, ch)
            not in channels_rm,
            default_value=default_value,
        )

        # see in which events the current channel is present
        mask = ak.to_numpy(ak.any(ch_comp == ch, axis=-1), allow_missing=False)
        cv = np.full(len(ch_comp), np.nan)
        cv[evt_ids_ch] = res
        cv[~mask] = np.nan
        cv = ak.drop_none(ak.nan_to_none(ak.Array(cv)[:, None]))

        out = ak.concatenate((out, cv), axis=-1)

        if ch == channels[0]:
            type_name = res.dtype

    return VectorOfVectors(ak.values_astype(out, type_name), dtype=type_name)


def evaluate_to_aoesa(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    channels: list,
    channels_rm: list,
    expr: str,
    exprl: list,
    query: str | NDArray,
    n_rows: int,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
    missv=np.nan,
) -> ArrayOfEqualSizedArrays:
    """Aggregates by returning an :class:`.ArrayOfEqualSizedArrays` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawid"s at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    missv
       missing value.
    sorter
       sorts the entries in the vector according to sorter expression.
    """
    f = utils.make_files_config(datainfo)
    table_id_fmt = f.hit.table_fmt

    # define dimension of output array
    out = np.full((n_rows, len(channels)), missv)

    i = 0
    for ch in channels:
        idx_ch = tcm.idx[tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch)]
        evt_ids_ch = np.searchsorted(
            tcm.cumulative_length,
            np.where(tcm.id == utils.get_tcm_id_by_pattern(table_id_fmt, ch))[0],
            "right",
        )
        res = utils.get_data_at_channel(
            datainfo=datainfo,
            ch=ch,
            tcm=tcm,
            expr=expr,
            exprl=exprl,
            pars_dict=pars_dict,
            is_evaluated=ch not in channels_rm,
            default_value=default_value,
        )

        # get mask from query
        limarr = utils.get_mask_from_query(
            datainfo=datainfo,
            query=query,
            length=len(res),
            ch=ch,
            idx_ch=idx_ch,
        )

        out[evt_ids_ch, i] = np.where(limarr, res, out[evt_ids_ch, i])

        i += 1

    return ArrayOfEqualSizedArrays(nda=out)


def evaluate_to_vector(
    datainfo: utils.TierData,
    tcm: utils.TCMData,
    channels: list,
    channels_rm: list,
    expr: str,
    exprl: list,
    query: str | NDArray,
    n_rows: int,
    pars_dict: dict = None,
    default_value: bool | int | float = np.nan,
    sorter: str = None,
) -> VectorOfVectors:
    """Aggregates by returning a :class:`.VectorOfVector` of evaluated
    expressions of channels that fulfill a query expression.

    Parameters
    ----------
    datainfo
        input and output LH5 datainfo with HDF5 groups where tables are found.
    tcm
        TCM data arrays in an object that can be accessed by attribute.
    channels
       list of channels to be aggregated.
    channels_rm
       list of channels to be skipped from evaluation and set to default value.
    expr
       expression string to be evaluated.
    exprl
       list of `dsp/hit/evt` parameter tuples in expression ``(tier, field)``.
    query
       query expression to mask aggregation.
    n_rows
       length of output :class:`.VectorOfVectors`.
    ch_comp
       array of "rawids" at which the expression is evaluated.
    pars_dict
       dictionary of `evt` and additional parameters and their values.
    default_value
       default value.
    sorter
       sorts the entries in the vector according to sorter expression.
       ``ascend_by:<hit|dsp.field>`` results in an vector ordered ascending,
       ``decend_by:<hit|dsp.field>`` sorts descending.
    """
    out = evaluate_to_aoesa(
        datainfo=datainfo,
        tcm=tcm,
        channels=channels,
        channels_rm=channels_rm,
        expr=expr,
        exprl=exprl,
        query=query,
        n_rows=n_rows,
        pars_dict=pars_dict,
        default_value=default_value,
        missv=np.nan,
    ).view_as("np")

    # if a sorter is given sort accordingly
    if sorter is not None:
        md, fld = sorter.split(":")
        s_val = evaluate_to_aoesa(
            datainfo=datainfo,
            tcm=tcm,
            channels=channels,
            channels_rm=channels_rm,
            expr=fld,
            exprl=[tuple(fld.split("."))],
            query=None,
            n_rows=n_rows,
            missv=np.nan,
        ).view_as("np")
        if "ascend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(s_val)]

        elif "descend_by" == md:
            out = out[np.arange(len(out))[:, None], np.argsort(-s_val)]
        else:
            raise ValueError(
                "sorter values can only have 'ascend_by' or 'descend_by' prefixes"
            )

    return VectorOfVectors(
        ak.values_astype(
            ak.drop_none(ak.nan_to_none(ak.Array(out))), type(default_value)
        ),
        dtype=type(default_value),
    )
