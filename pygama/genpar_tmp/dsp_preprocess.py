import argparse, os
import json
from collections import OrderedDict
import h5py
import numpy as np

import pygama
from pygama.genpar_tmp.tau import get_decay_constant
from pygama.genpar_tmp.cuts import generate_cuts, get_cut_indexes
from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.units import *
from pygama import lh5
from pygama.utils import update_progress
import pygama.git as git
from pygama.dsp.build_processing_chain import *
from pygama.dsp.errors import DSPFatal

def dsp_preprocess_decay_const(f_raw, dsp_config, database_file, database=None, lh5_tables=None,
                               outputs=None, n_max=np.inf, buffer_len=3200,
                               block_width=16, verbose=1, overwrite=False):
    

    if isinstance(dsp_config, str):
        with open(dsp_config, 'r') as config_file:
            dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)
            
    if not isinstance(dsp_config, dict):
        raise Exception('Error, dsp_config must be an dict')
    raw_store = lh5.Store()
    lh5_file = raw_store.gimme_file(f_raw, 'r')
    if lh5_file is None:
        print(f'raw_to_dsp: input file not found: {f_raw}')
        return
    else: print(f'Opened file {f_raw}')

    # if no group is specified, assume we want to decode every table in the file
    if lh5_tables is None:
        lh5_tables = []
        lh5_keys = raw_store.ls(f_raw)

        # sometimes 'raw' is nested, e.g g024/raw
        for tb in lh5_keys:
            if "raw" not in tb:
                tbname = raw_store.ls(lh5_file[tb])[0]
                if "raw" in tbname:
                    tb = tb +'/'+ tbname # g024 + /raw
            lh5_tables.append(tb)

    # make sure every group points to waveforms, if not, remove the group
    for tb in lh5_tables:
        if 'raw' not in tb:
            lh5_tables.remove(tb)
    if len(lh5_tables) == 0:
        print("Empty lh5_tables, exiting...")
        sys.exit(1)

    # get the database parameters. For now, this will just be a dict in a json
    # file, but eventually we will want to interface with the metadata repo
    if isinstance(database, str):
        with open(database, 'r') as db_file:
            database = json.load(db_file)

    if database and not isinstance(database, dict):
        database = None
        print('database is not a valid json file or dict. Using default db values.')


    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max<tot_n_rows: tot_n_rows=n_max

        chan_name = tb.split('/')[0]
        db_dict = database.get(chan_name) if database else None
        lh5_in, n_rows_read = raw_store.read_object(tb, f_raw, start_row=0, n_rows=buffer_len)
        pc, tb_out = build_processing_chain(lh5_in, dsp_config, db_dict, outputs, verbose, block_width)

        print(f'Processing table: {tb} ...')
        for start_row in range(0, tot_n_rows, buffer_len):
            if verbose > 0:
                update_progress(start_row/tot_n_rows)
            lh5_in, n_rows = raw_store.read_object(tb, f_raw, start_row=start_row, n_rows=buffer_len, obj_buf=lh5_in)
            n_rows = min(tot_n_rows-start_row, n_rows)
            try:
                pc.execute(0, n_rows)
            except DSPFatal as e:
                # Update the wf_range to reflect the file position
                e.wf_range = "{}-{}".format(e.wf_range[0]+start_row, e.wf_range[1]+start_row)
                raise e

    cut_dict = generate_cuts(tb_out, parameters = {'bl_mean':4, 'bl_std':4,'bl_slope':4})
    idxs = get_cut_indexes(tb_out,cut_dict, verbose=False)
    slopes = tb_out['tail_slope'].nda
    get_decay_constant(slopes[idxs],database_file)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
    """Process a single tier 1 LH5 file and save a value for the decay constant to the database file""")

    arg = parser.add_argument
    arg('file', help="Input (tier 1) LH5 file.")

    arg('-v', '--verbose', default=1, type=int,
        help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")

    arg('-b', '--block', default=16, type=int,
        help="Number of waveforms to process simultaneously. Default is 8")

    arg('-n', '--nevents', default=None, type=int,
        help="Number of waveforms to process. By default do the whole file")
    arg('-g', '--group', default=None, action='append', type=str,
        help="Name of group in LH5 file. By default process all base groups. Supports wildcards.")

    arg('-j', '--jsonconfig', default=None, type=str,
        help="Name of json file used by raw_to_dsp to construct the processing routines used. By default use dsp_config in pygama/apps.")

    arg('-d', '--dbfile', default=None, type=str,
        help="JSON file to write DB parameters to.")

    arg('-d', '--measurement', default=None, type=str,
        help="Measurement that will be used as the key for the output dictionary JSON")

    args = parser.parse_args()

    dsp_preprocess_decay_const(args.file,  args.jsonconfig, args.measurement, lh5_tables=None,
                                outputs=None, n_max=np.inf, buffer_len=3200,
                                block_width=16, verbose=1)

    
