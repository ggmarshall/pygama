import argparse, os
import json
from collections import OrderedDict
import numpy as np
import pathlib

from pygama.analysis import histograms as pgh
from pygama.dsp.units import *
from pygama import lh5
from pygama.utils import tqdm_range
from pygama.dsp.build_processing_chain import *
from pygama.dsp.errors import DSPFatal
from pygama.pargen.cuts import generate_cuts, get_cut_indexes

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

def get_decay_constant(slopes,  dict_file, wfs, plot_path = None, overwrite=False):
    
    """
    Finds the decay constant from the modal value of the tail slope after cuts
    and saves it to the specified json.
    Parameters
    ----------
    slopes : array
             tail slope array
    
    dict_file ; str
                path to json file to save decay constant value to. 
                It will be saved as a dictionary of form {'pz': {'tau': decay_constant}}
    """


    if os.path.isfile(dict_file) == True:
        tau_dict = json.load(open(dict_file,'r'))

    else:
        tau_dict = {}


    pz = tau_dict.get("pz")
    if overwrite == False and pz is not None:
        print('Tau already Calculated and Overwrite is False')
        return

    counts, bins, var = pgh.get_hist(slopes, bins=50000, range=(-0.01,0))
    bin_centres = pgh.get_bin_centers(bins)
    tau = round(-1/(bin_centres[np.argmax(counts)]),1)

    tau_dict["pz"] = {"tau":tau}
    with open(dict_file,'w') as fp:
        json.dump(tau_dict,fp, indent=4)
    if plot_path is None:
        return
    else:
        pathlib.Path(os.path.dirname(plot_path)).mkdir(parents=True, exist_ok=True)
        with PdfPages(plot_path) as pdf:
            mpl.use('pdf')
            plt.rcParams['figure.figsize'] = (20, 12)
            plt.rcParams['font.size'] = 12
            fig,ax = plt.subplots()
            bins = 10000 #change if needed
            counts, bins, bars = ax.hist(slopes, bins=bins, histtype='step')
            plot_max = np.argmax(counts)
            in_min = plot_max - 10
            if in_min <0:
                in_min = 0
            in_max = plot_max + 11
            if in_max >= len(bins):
                in_min = len(bins)-1
            plt.xlabel("Slope")
            plt.ylabel("Counts")
            plt.yscale('log')
            axins = ax.inset_axes([0.5, 0.45, 0.47, 0.47])
            axins.hist(slopes[(slopes>bins[in_min])&(slopes<bins[in_max])], bins=200, histtype='step')
            axins.set_xlim(bins[in_min], bins[in_max])
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels = labels, rotation=45)
            pdf.savefig()
            plt.close()

            wf_idxs = np.random.choice(len(wfs),100)
            plt.figure()
            for wf_idx in wf_idxs:
                plt.plot(np.arange(0,len(wfs[wf_idx]), 1), wfs[wf_idx])
            plt.xlabel("Samples")
            plt.ylabel("ADU")
            pdf.savefig()
            plt.close()



def dsp_preprocess_decay_const(f_raw, dsp_config, database_file, database=None, plot_path = None, verbose=0, overwrite=False):
    """
    This function calculates the pole zero constant for the input data

    f_raw : str 
            The raw file to run the macro on

    dsp_config: str
            Path to the dsp config file, this is a stripped down version which just includes cuts and slope of decay tail

    database_file:  str
            Path to the output file, the macro will output this as a json file.
    """

    tb_out = raw_to_dsp_no_save(f_raw,  dsp_config, database_file=database_file,  verbose=verbose, database=database)
    if verbose>0: print("Processed Data")
    cut_dict = generate_cuts(tb_out, parameters = {'bl_mean':4, 'bl_std':4,'bl_slope':4})
    if verbose>0: 
        print("Generated Cuts:", cut_dict)
    idxs = get_cut_indexes(tb_out,cut_dict, verbose=False)
    slopes = tb_out['tail_slope'].nda
    wfs = tb_out['wf_blsub'].nda
    get_decay_constant(slopes[idxs],database_file, wfs, plot_path=plot_path, overwrite=overwrite)



def raw_to_dsp_no_save(f_raw, dsp_config, database_file, database=None, lh5_tables=None,
                               outputs=None, n_max=np.inf, buffer_len=3200,
                               block_width=16, verbose=1, chan_config=None):

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

    # loop over tables to run DSP on
    for tb in lh5_tables:
        # load primary table and build processing chain and output table
        tot_n_rows = raw_store.read_n_rows(tb, f_raw)
        if n_max and n_max<tot_n_rows: tot_n_rows=n_max

        # if we have separate DSP files for each table, read them in here
        if chan_config is not None:
            f_config = chan_config[tb]
            with open(f_config, 'r') as config_file:
                dsp_config = json.load(config_file, object_pairs_hook=OrderedDict)
            print('Processing table:', tb, 'with DSP config file:\n  ', f_config)

        if not isinstance(dsp_config, dict):
            raise Exception('Error, dsp_config must be an dict')

        chan_name = tb.split('/')[0]
        db_dict = database.get(chan_name) if database else None
        lh5_in, n_rows_read = raw_store.read_object(tb, f_raw, start_row=0, n_rows=buffer_len)
        pc, mask, tb_out = build_processing_chain(lh5_in, dsp_config, db_dict, outputs, verbose, block_width)

        print(f'Processing table: {tb} ...')

        for start_row in tqdm_range(0, int(tot_n_rows), buffer_len, verbose):
            lh5_in, n_rows = raw_store.read_object(tb, f_raw, start_row=start_row, n_rows=buffer_len, field_mask = mask, obj_buf=lh5_in)
            n_rows = min(tot_n_rows-start_row, n_rows)
            try:
                pc.execute(0, n_rows)
            except DSPFatal as e:
                # Update the wf_range to reflect the file position
                e.wf_range = "{}-{}".format(e.wf_range[0]+start_row, e.wf_range[1]+start_row)
                raise e
    return tb_out

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