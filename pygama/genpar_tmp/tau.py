import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as hist
import pygama.lh5 as lh5
import matplotlib.pyplot as plt
from pygama.genpar_tmp import cuts as cts

def get_decay_constant(file_path, cut_path, lh5_group, dict_file):
    """
    Gets the decay constant as saves it to the specified json 
    """
    if os.path.isfile(dict_file) == True:
        tau_dict = json.load(open(dict_file,'r'))

    else:
        tau_dict = {}


    base=os.path.basename(file_path)

    file_name = os.path.splitext(base)[0]
    parts = file_name.split('-')
    run = parts[0]+'-'+parts[1]+'-'+parts[2]+'-'+parts[3]  #Probably a nicer way to do this

    if run in tau_dict and overwrite == False:
        print('Tau already Calculated and Overwrite is False')
        return
    elif run in tau_dict and overwrite == True:
        tau_dict.pop(run)
    
    slopes = cts.load_nda_with_cuts(file_path,cut_path, lh5_group, ['tail_slope'], verbose=False)['tail_slope']
    counts, bins, var = hist.get_hist(slopes, bins=50000, range=(-0.01,0))
    bin_centres = hist.get_bin_centers(bins)
    tau = round(-1/(bin_centres[np.argmax(counts)]),1)

    tau_dict.update({run:tau})
    with open(dict_file,'w') as fp:
        json.dump(tau_dict,fp, indent=4)
    return tau


