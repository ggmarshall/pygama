import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as hist
import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import glob

def get_cut_boundaries(file_path, cut_file, lh5_group, parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}, overwrite=False):
    
    """
    Finds cut boundaries for a file pass parameters as a dictionary with the parameter to be cut and the number of 
    sigmas to cut
    """
    if os.path.isfile(cut_file) == True:
        cut_dict = json.load(open(cut_file,'r'))

    else:
        cut_dict = {}


    output_dict = {}

    base=os.path.basename(file_path)

    file_name = os.path.splitext(base)[0]
    parts = file_name.split('-')
    run = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]  #Probably a nicer way to do this

    if run in cut_dict and overwrite == False:
        print('Cut Parameters already Calculated and Overwrite is False')
        return
    elif run in cut_dict and overwrite == True:
        cut_dict.pop(run)
    all_pars_array = lh5.load_nda(file_path, parameters,  lh5_group)
    for pars in parameters.keys():
        num_sigmas = parameters[pars]
        par_array = all_pars_array[pars]
        counts, bins, var = hist.get_hist(par_array,10**5)
        bin_centres = hist.get_bin_centers(bins)
        fwhm = hist.get_fwhm(counts, bins)[0]
        mean = float(bin_centres[np.argmax(counts)])
        std = fwhm/2.355
        upper =float( (num_sigmas*std)+mean)
        lower = float((-num_sigmas*std)+mean)
        output_dict.update({pars : {'Mean Value': mean, 'Sigmas Cut': num_sigmas, 'Upper Boundary' : upper, 'Lower Boundary': lower}})

    cut_dict.update({run:output_dict})
    with open(cut_file,'w') as fp:
        json.dump(cut_dict,fp, indent=4)
    return

def get_cut_indexes(file, cut_dict, lh5_group, verbose):

    """
    Returns a mask of the data that passes cuts takes in dictionary of cuts and a single file

    Parameters
    ----------
    File : str 
           File path
    Cut_dict : string
                Dictionary file with cuts
    lh5_group : string
                lh5 file path e.g. 'raw/'
    """
    
    indexes = None
    keys = cut_dict.keys()
    all_cut_data = lh5.load_nda(file, keys, lh5_group, verbose=verbose)
    for cut in keys:
        data = all_cut_data[cut]
        upper = cut_dict[cut]['Upper Boundary']
        lower = cut_dict[cut]['Lower Boundary']
        idxs = (data<upper) & (data>lower) 
        if indexes is not None:
            indexes = indexes & idxs
            
        else:
            indexes = idxs

    return indexes

def load_df_with_cuts(files, cut_file_path, lh5_group, verbose=True):

    """
    This function loads data after applying cuts specified in cut_file

    Parameters
    ----------
    Files : str or list of str's
        A list of files. Can contain wildcards
    Cut_file_path : string
                    Path to json dictionary file with cuts
    lh5_group : string
                lh5 file path e.g. 'raw/'
    
    """

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    sto = lh5.Store()
    if os.path.isfile(cut_file_path) == False:
        get_cut_boundaries(files[0], cut_file_path, lh5_group)
    with open(cut_file_path,'r') as f:
            full_cut_dict = json.load(f)
    #get first file name

    base=os.path.basename(files[0])
    file_name = os.path.splitext(base)[0]
    parts = file_name.split('-')
    run1 = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]
    
    try:
        cut_dict = full_cut_dict[run1]
        if verbose:
            print('Loaded Cut Dictionary')
    except KeyError:
        print("Cuts haven't been calculated yet, getting cut boundaries")
        get_cut_boundaries(files[0], cut_file_path, lh5_group)
        with open(cut_file_path,'r') as f:
            full_cut_dict = json.load(f)
        cut_dict = full_cut_dict[run1]
    idxs = []
    for file in files:
        if verbose:
                print("loading data for", file)

        base=os.path.basename(file)
        file_name = os.path.splitext(base)[0]
        parts = file_name.split('-')
        run = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]

        if run != run1:
            print ("Files not all in same run")
            run1 = run
            try:
                cut_dict = full_cut_dict[run1]
            except KeyError:
                print("Cuts haven't been calculated yet, getting cut boundaries")
                get_cut_boundaries(file, cut_file_path, lh5_group)
                with open(cut_file,'r') as f:
                    full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]
        
        idx = get_cut_indexes(file, cut_dict, lh5_group, verbose)
        idxs.append(idx)
    mask = np.concatenate(idxs)
    
    tb = sto.read_object(lh5_group, files)[0]
    data = lh5.Table.get_dataframe(tb)
        
    cut_data = data.iloc[mask]

    cut_data.reset_index(inplace= True)
    return cut_data



def load_nda_with_cuts(files, cut_file_path, lh5_group, parameters, verbose=True):

    """

    This function loads data after applying cuts specified in cut_file

    Parameters
    ----------
    files : List
            list of file paths
    cut_file_path : string
                    Path to json dictionary file with cuts
    lh5_group : string
                lh5 file path e.g. 'raw/'
    parameters : list
                list of parameters to load, must be different to cut parameters
    
    """


    
    sto = lh5.Store()

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    if os.path.isfile(cut_file_path) == False:
        get_cut_boundaries(files[0], cut_file_path, lh5_group)
    with open(cut_file_path,'r') as f:
            full_cut_dict = json.load(f)

    #get first file name

    base=os.path.basename(files[0])
    file_name = os.path.splitext(base)[0]
    parts = file_name.split('-')
    run1 = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]
    
    try:
        cut_dict = full_cut_dict[run1]
        if verbose:
            print('Loaded Cut Dictionary')
    except KeyError:
        print("Cuts haven't been calculated yet, getting cut boundaries")
        get_cut_boundaries(files[0], cut_file_path, lh5_group)
        with open(cut_file_path,'r') as f:
            full_cut_dict = json.load(f)
        cut_dict = full_cut_dict[run1]
    if (set(parameters) & set(cut_dict.keys())):
        raise KeyError("Input Parameters must be different to cut parameters otherwise use load_df_with_cuts")
    idxs = []
    for file in files:

        base=os.path.basename(file)
        file_name = os.path.splitext(base)[0]
        parts = file_name.split('-')
        run = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]

        if run != run1:
            print ("Files not all in same run")
            run1 = run
            try:
                cut_dict = full_cut_dict[run1]
            except KeyError:
                print("Cuts haven't been calculated yet, getting cut boundaries")
                get_cut_boundaries(file, cut_file_path, lh5_group)
                with open(cut_file_path,'r') as f:
                    full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]
        idx = get_cut_indexes(file, cut_dict, lh5_group, verbose)
        idxs.append(idx)
    mask = np.concatenate(idxs)
    #Concat dataframes together and return
    par_data = lh5.load_nda(files, parameters, lh5_group, verbose=verbose)
    for par in par_data:
        par_data[par] = par_data[par][mask]
    return par_data

