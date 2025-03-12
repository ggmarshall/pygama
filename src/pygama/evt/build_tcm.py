from __future__ import annotations

import re

import awkward as ak
import lgdo
from lgdo import lh5
from lgdo.types import Table

from . import tcm as ptcm


def _concat_tables(tbls):
    # this will drop attrs, need to fix
    out_tbl = tbls[0].view_as("ak")
    for tbl in tbls[1:]:
        out_tbl = ak.concatenate([out_tbl, tbl.view_as("ak")], axis=0)
    return Table(col_dict=out_tbl)


def build_tcm(
    input_tables: list[tuple[str, str | list[str]]],
    coin_cols: str,
    hash_func: str = r"\d+",
    coin_windows: float = 0,
    window_refs: str = "last",
    out_file: str = None,
    out_name: str = "tcm",
    wo_mode: str = "write_safe",
    buffer_len: int = None,
    out_fields=None,
) -> lgdo.Table:
    r"""Build a Time Coincidence Map (TCM).

    Given a list of input tables, create an output table containing an entry
    list of coincidences among the inputs. Uses
    :func:`.evt.tcm.generate_tcm_cols`. For use with the
    :class:`~.flow.data_loader.DataLoader`.

    Parameters
    ----------
    input_tables
        each entry is ``(filename, table_name_pattern)``. All tables matching
        ``table_name_pattern`` in ``filename`` will be added to the list of
        input tables. ``table_name_pattern`` can be replaced with a list of
        patterns to be searched for in the file
    coin_col
        the name of the column in each tables used to build coincidences. All
        tables must contain a column with this name.
    hash_func
        mapping of table names to integers for use in the TCM.  `hash_func` is
        a regexp pattern that acts on each table name. The default `hash_func`
        ``r"\d+"`` pulls the first integer out of the table name. Setting to
        ``None`` will use a table's index in `input_tables`.
    coin_window
        the clustering window width.
    window_ref
        Configuration for the clustering window.
    out_file
        name (including path) for the output file. If ``None``, no file will be
        written; the TCM will just be returned in memory.
    out_name
        name for the TCM table in the output file.
    wo_mode
        mode to send to :meth:`~.lgdo.lh5.LH5Store.write`.

    See Also
    --------
    .tcm.generate_tcm_cols
    """
    # hash_func: later can add list or dict or a function(str) --> int.

    if not isinstance(coin_cols, list):
        coin_cols = [coin_cols]
    if not isinstance(coin_windows, list):
        coin_windows = [coin_windows]
    if not isinstance(window_refs, list):
        window_refs = [window_refs]

    iterators = []
    array_ids = []
    all_tables = []

    if buffer_len is None:
        ntables = 0
        for filename, patterns in input_tables:
            if isinstance(patterns, str):
                patterns = [patterns]
            for pattern in patterns:
                ntables += len(lh5.ls(filename, lh5_group=pattern))

        n_fields = (
            2 + len(set(coin_cols + out_fields))
            if out_fields is not None
            else 2 + len(set(coin_cols))
        )
        buffer_len = int(10**7 / (ntables * n_fields))

    for filename, patterns in input_tables:
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            tables = lh5.ls(filename, lh5_group=pattern)
            for table in tables:
                all_tables.append(table)
                array_id = len(array_ids)
                if hash_func is not None:
                    if isinstance(hash_func, str):
                        array_id = int(re.search(hash_func, table).group())
                    else:
                        raise NotImplementedError(
                            f"hash_func of type {type(hash_func).__name__}"
                        )
                else:
                    array_id = len(all_tables) - 1
                iterators.append(
                    lh5.LH5Iterator(
                        filename, table, field_mask=coin_cols, buffer_len=buffer_len
                    )
                )
                array_ids.append(array_id)

    coin_windows = [
        ptcm.coin_groups(n, w, r)
        for n, w, r in zip(coin_cols, coin_windows, window_refs)
    ]

    tcm_gen = ptcm.generate_tcm_cols(
        iterators, coin_windows=coin_windows, array_ids=array_ids, fields=out_fields
    )

    if out_file is not None:
        sto = lh5.LH5Store()
        while True:
            try:
                out_tbl = tcm_gen.__next__()
                out_tbl.attrs.update(
                    {"tables": str(all_tables), "hash_func": str(hash_func)}
                )
                sto.write(out_tbl, out_name, out_file, wo_mode=wo_mode)
            except StopIteration:
                break
    else:
        tcm = []
        while True:
            try:
                tcm.append(tcm_gen.__next__())
            except StopIteration:
                break
        total_length = 0
        for item in tcm:
            total_length += len(item)
        out_tbl = _concat_tables(tcm)
        out_tbl.attrs.update({"tables": str(all_tables), "hash_func": str(hash_func)})
        return out_tbl
