import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import glob

def generate_cuts(data, parameters):
    """
    Finds double sided cut boundaries for a file for the parameters specified 
    
    Parameters
    ----------
    data : lh5 table or dictionary of arrays
                data to calculate cuts on
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    """

    output_dict = {}
    for pars in parameters.keys():
        num_sigmas = parameters[pars]
        par_array = data[pars]
        if not isinstance(par_array, np.ndarray):
            par_array = par_array.nda
        counts, bins, var = pgh.get_hist(par_array,10**5)
        bin_centres = pgh.get_bin_centers(bins)
        fwhm = pgh.get_fwhm(counts, bins)[0]
        mean = float(bin_centres[np.argmax(counts)])
        std = fwhm/2.355
        if isinstance(num_sigmas, (int, float)):
            num_sigmas_left = num_sigmas
            num_sigmas_right = num_sigmas
        elif isinstance(num_sigmas, dict):
            num_sigmas_left = num_sigmas["left"]
            num_sigmas_right = num_sigmas["right"]
        upper =float( (num_sigmas_right *std)+mean)
        lower = float((-num_sigmas_left *std)+mean)
        output_dict.update({pars : {'Mean Value': mean, 'Sigmas Cut': num_sigmas, 'Upper Boundary' : upper, 'Lower Boundary': lower}})
    return output_dict
    
def get_cut_boundaries(file_path, cut_file, lh5_group, parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}, overwrite=False):
    
    """
    Finds double sided cut boundaries for a file for the parameters specified 
    Parameters
    ----------
    file_path : str
                path to data file
    cut_file : str
                path to json file of cuts will save cuts in the form:
                {detector-source-run : {parameter : {Mean_value : mean_value, Sigmas cut : sigmas, 
                                                     Upper Boundary : value, Lower Boundary : value}}
    lh5_group : str
                lh5_path (e.g. 'raw')
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    overwrite : bool
                True to overwrite existing cuts in json file
    """


    if os.path.isfile(cut_file) == True:
        cut_dict = json.load(open(cut_file,'r'))

    else:
        cut_dict = {}
    base=os.path.basename(file_path)

    file_name = os.path.splitext(base)[0]
    datatype, detector, measurement, run, timestamp = file_name.split('-')
    det_run = datatype+'-'+detector+'-'+measurement+'-'+run

    if det_run in cut_dict and overwrite == False:
        print('Cut Parameters already Calculated and Overwrite is False')
        return

    all_pars_array = lh5.load_nda(file_path, parameters.keys(),  lh5_group)
    output_dict = generate_cuts(all_pars_array, parameters)
    cut_dict[det_run] = output_dict
    with open(cut_file,'w') as fp:
        json.dump(cut_dict,fp, indent=4)
    return


def get_cut_indexes(all_data, cut_dict, verbose=False):

    """
    Returns a mask of the data, for a single file, that passes cuts based on dictionary of cuts 
    in form of cut boundaries above
    Parameters
    ----------
    File : dict or lh5_table
           dictionary of parameters + array such as load_nda or lh5 table of params
    Cut_dict : string
                Dictionary file with cuts
    """
    
    indexes = None
    keys = cut_dict.keys()
    for cut in keys:
        data = all_data[cut]
        if not isinstance(data, np.ndarray):
            data = data.nda
        upper = cut_dict[cut]['Upper Boundary']
        lower = cut_dict[cut]['Lower Boundary']
        idxs = (data<upper) & (data>lower) 
        if indexes is not None:
            indexes = indexes & idxs
            
        else:
            indexes = idxs
        if verbose: print(cut, ' loaded')

    return indexes

def load_df_with_cuts(files, lh5_group, cut_file=None, cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}, verbose=True):

    """

    This function loads all data after applying cuts, by default it will simply load the data according to the default cut parameters. 
    You can specify these cut parameters with a dictionary. If you supply a cut file json it will load and save the cuts to this 
    file in the form of get_cut_boundaries
    Parameters
    ----------
    files : List
            list of file paths

    lh5_group : string
                lh5 file path e.g. 'raw/'
    
    cut_file : string
                Path to json dictionary file with cuts, if provided cuts will be saved/loaded from here.
    
    cut_parameters :
                Dictionary in the form {cut:no_sigmas}, can be double sided {cut:{"left":no_sigmas,"right":no_sigmas}}
                if cut_file provided and cuts found there will overrule cut_parameters, if none found will save these to cut_file
    
    """

    sto = lh5.Store()
    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    base=os.path.basename(files[0])
    file_name = os.path.splitext(base)[0]
    datatype, detector, measurement, run, timestamp = file_name.split('-')
    run1 = datatype+'-'+detector+'-'+measurement+'-'+run

    if cut_file is None:
        data = lh5.load_nda(files[0], cut_parameters.keys(), lh5_group)
        cut_dict = generate_cuts(data, cut_parameters)
        print(cut_dict)
        idxs = []
        for file in files:
            base=os.path.basename(files[0])
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = datatype+'-'+detector+'-'+measurement+'-'+run
            if det_run != run1:
                print ("Files not all in same run")
                run1 = det_run
                data = lh5.load_nda(files, cut_parameters.keys(), lh5_group)
                cut_dict = generate_cuts(data, cut_parameters)
                print(cut_dict)
            keys = cut_dict.keys()
            par_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, verbose)
            idxs.append(idx)

    else:
        if os.path.isfile(cut_file) == False:
            get_cut_boundaries(files[0], cut_file, lh5_group, parameters= cut_parameters)
        with open(cut_file,'r') as f:
            full_cut_dict = json.load(f)
        try:
            cut_dict = full_cut_dict[run1]
            if verbose:
                print('Loaded Cut Dictionary')
        except KeyError:
            print("Cuts haven't been calculated yet, getting cut boundaries")
            get_cut_boundaries(files[0], cut_file, lh5_group, cut_parameters)
            with open(cut_file,'r') as f:
                full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]

        idxs = []
        for file in files:
            if verbose:
                print("loading data for", file)

            base=os.path.basename(files[0])
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = datatype+'-'+detector+'-'+measurement+'-'+run
            if det_run != run1:
                print ("Files not all in same run")
                run1 = det_run
                try:
                    cut_dict = full_cut_dict[run1]
                except KeyError:
                    print("Cuts haven't been calculated yet, getting cut boundaries")
                    get_cut_boundaries(files[0], cut_file, lh5_group, cut_parameters)
                    with open(cut_file,'r') as f:
                        full_cut_dict = json.load(f)
                        cut_dict = full_cut_dict[run1]
            keys = cut_dict.keys()
            par_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, verbose)
            idxs.append(idx)

    mask = np.concatenate(idxs)
    
    print(len(np.where(mask)[0])/ len(mask) *100, "% passed cuts")
    tb = sto.read_object(lh5_group, files)[0]
    data = lh5.Table.get_dataframe(tb)
        
    cut_data = data.iloc[mask]
    failed_cuts = data.iloc[~mask]

    cut_data.reset_index(inplace= True)
    failed_cuts.reset_index(inplace= True)
    return cut_data, failed_cuts



def load_nda_with_cuts(files, lh5_group, parameters, cut_file=None,  cut_parameters= {'bl_mean':4,'bl_std':4, 'pz_std':4}, verbose=True):

    """

    This function loads data after applying cuts, by default it will simply load the data according to the default cut parameters. 
    You can specify these cut parameters with a dictionary. If you supply a cut file json it will load and save the cuts to this 
    file in the form of get_cut_boundaries
    Parameters
    ----------
    files : List
            list of file paths

    lh5_group : string
                lh5 file path e.g. 'raw/'

    parameters : str, list
                parameters to load
    
    cut_file : string
                Path to json dictionary file with cuts, if provided cuts will be saved/loaded from here.
    
    cut_parameters :
                Dictionary in the form {cut:no_sigmas}, can be double sided {cut:{"left":no_sigmas,"right":no_sigmas}}
                if cut_file provided and cuts found there will overrule cut_parameters, if none found will save these to cut_file
    
    """

    sto = lh5.Store()

    if isinstance(files, str): files = [files]
    if isinstance(parameters, str): parameters = [parameters]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    base=os.path.basename(files[0])
    file_name = os.path.splitext(base)[0]
    datatype, detector, measurement, run, timestamp = file_name.split('-')
    run1 = datatype+'-'+detector+'-'+measurement+'-'+run

    if cut_file is None:
        data = lh5.load_nda(files[0], cut_parameters.keys(), lh5_group)
        cut_dict = generate_cuts(data, cut_parameters)
        print(cut_dict)
        idxs = []
        for file in files:
            base=os.path.basename(file)
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = datatype+'-'+detector+'-'+measurement+'-'+run
            if det_run != run1:
                run1 = det_run
                print ("Files not all in same run")
                data = lh5.load_nda(files, cut_parameters.keys(), lh5_group)
                cut_dict = generate_cuts(data, cut_parameters)
                print(cut_dict)
            keys = cut_dict.keys()
            par_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, verbose)
            idxs.append(idx)

    else:
        if os.path.isfile(cut_file) == False:
            get_cut_boundaries(files[0], cut_file, lh5_group, parameters= cut_parameters)
        with open(cut_file,'r') as f:
            full_cut_dict = json.load(f)
        try:
            cut_dict = full_cut_dict[run1]
            if verbose:
                print('Loaded Cut Dictionary')
        except KeyError:
            print("Cuts haven't been calculated yet, getting cut boundaries")
            get_cut_boundaries(files[0], cut_file, lh5_group, cut_parameters)
            with open(cut_file,'r') as f:
                full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]

        idxs = []
        for file in files:
            if verbose:
                print("loading data for", file)

            base=os.path.basename(files[0])
            file_name = os.path.splitext(base)[0]
            datatype, detector, measurement, run, timestamp = file_name.split('-')
            det_run = datatype+'-'+detector+'-'+measurement+'-'+run
            if det_run != run1:
                print ("Files not all in same run")
                run1 = det_run
                try:
                    cut_dict = full_cut_dict[run1]
                except KeyError:
                    print("Cuts haven't been calculated yet, getting cut boundaries")
                    get_cut_boundaries(files[0], cut_file, lh5_group, cut_parameters)
                    with open(cut_file,'r') as f:
                        full_cut_dict = json.load(f)
                        cut_dict = full_cut_dict[run1]
            keys = cut_dict.keys()
            par_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
            idx = get_cut_indexes(par_data, cut_dict, verbose)
            idxs.append(idx)
    
    mask = np.concatenate(idxs)
    print(len(np.where(mask)[0])/ len(mask) *100, "% passed cuts")
    #Concat dataframes together and return
    par_data = lh5.load_nda(files, parameters, lh5_group, verbose=verbose)
    failed_cuts={}
    passed_cuts={}
    for par in par_data:
        passed_cuts[par] = par_data[par][mask]
        failed_cuts[par] = par_data[par][~mask]
    return passed_cuts, failed_cuts

