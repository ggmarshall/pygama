import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as hist
import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import glob

def get_cut_boundaries(file_path, cut_file, lh5_group, parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}, overwrite=False)
    """
    Finds cut boundaries for a file pass parameters as a dictionary with the parameter to be cut and the number of$
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
        output_dict.update({pars : {'Mean Value': mean, 'Sigmas Cut': num_sigmas, 'Upper Boundary' : upper, 'Lower$

    cut_dict.update({run:output_dict})
    with open(cut_file,'w') as fp:
        json.dump(cut_dict,fp, indent=4)
    return



def load_df_with_cuts(files, cut_file_path, lh5_group, verbose=True):

    '''
    This function loads data after applying cuts specified in cut_file

    Parameters
    ----------
    Files : str or list of str's
        A list of files. Can contain wildcards
    Cut_file_path : string
                    Path to json dictionary file with cuts
    lh5_group : string
                lh5 file path e.g. 'raw/'
    
    '''

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    def get_cut_indices(df, cut_dict):
    
        indexes = None
        keys = cut_dict.keys()
        for cut in keys:
            data = df[cut].to_numpy()
            upper = cut_dict[cut]['Upper Boundary']
            lower = cut_dict[cut]['Lower Boundary']
            idxs = (data<upper) & (data>lower) 
            if indexes is not None:
                indexes = indexes & idxs
            
            else:
                indexes = idxs

        return np.where(indexes)[0]

    sto = lh5.Store()
    if os.path.isfile(cut_file) == False:
        get_cut_boundaries(files[0], cut_file)
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
        get_cut_boundaries(files[0], cut_file)
        with open(cut_file_path,'r') as f:
            full_cut_dict = json.load(f)
        cut_dict = full_cut_dict[run1]
    all_data = {}
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
            except IndexError:
                print("Cuts haven't been calculated yet, getting cut boundaries")
                get_cut_boundaries(files[0], cut_file_path)
                with open(cut_file,'r') as f:
                    full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]

        tb = sto.read_object(lh5_group, file)[0]
        df = lh5.Table.get_dataframe(tb)
        idxs = get_cut_indices(df, cut_dict)
        df = df.iloc[idxs]
        all_data.update({os.path.basename(file):df})

    #Concat dataframes together and return
    all_data = pd.concat(all_data)
    all_data.reset_index(inplace= True)
    all_data.rename(columns = {'level_0':'File','level_1':'waveform_no'}, inplace = True)
    return all_data



def load_nda_with_cuts(files, cut_file_path, lh5_group, parameters, verbose=True):

    '''
    FUNCTIONALITY NOT YET IMPLEMENTED

    This function loads data after applying cuts specified in cut_file

    Parameters
    ----------
    Files : List
            list of file paths
    Cut_file_path : string
                    Path to json dictionary file with cuts
    lh5_group : string
                lh5 file path e.g. 'raw/'
    parameters : list
                list of parameters to load, must be different to cut parameters
    
    '''

    def get_cut_indexes(file, cut_dict, lh5_group):
    
        indexes = None
        keys = cut_dict.keys()
        all_cut_data = lh5.load_nda(file, keys, lh5_group)
        for cut in keys:
            data = all_cut_data[cut]
            upper = cut_dict[cut]['Upper Boundary']
            lower = cut_dict[cut]['Lower Boundary']
            idxs = (data<upper) & (data>lower) 
            if indexes is not None:
                indexes = indexes & idxs
            
            else:
                indexes = idxs

        return np.where(indexes)[0]
    
    sto = lh5.Store()

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

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
        get_cut_boundaries(files[0], cut_file)
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
            except IndexError:
                print("Cuts haven't been calculated yet, getting cut boundaries")
                get_cut_boundaries(files[0], cut_file_path)
                with open(cut_file,'r') as f:
                    full_cut_dict = json.load(f)
                cut_dict = full_cut_dict[run1]
        idx = get_cut_indexes(file, cut_dict, lh5_group)
        idxs.append(idx)
    #Concat dataframes together and return
    all_data = lh5.load_nda(files, parameters, lh5_group, idx_list=idxs)
    return all_data
