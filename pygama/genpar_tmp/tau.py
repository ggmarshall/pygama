import numpy as np
import pandas as pd
import os
import json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import matplotlib.pyplot as plt
from pygama.genpar_tmp import cuts as cts

def get_decay_constant(slopes,  dict_file, overwrite=False):
    
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

    try: 
        tau_dict["pz"]["tau"]
        if overwrite == False:
            print('Tau already Calculated and Overwrite is False')
            return
    except: 
        pass
    
    counts, bins, var = pgh.get_hist(slopes, bins=50000, range=(-0.01,0))
    bin_centres = pgh.get_bin_centers(bins)
    tau = round(-1/(bin_centres[np.argmax(counts)]),1)

    tau_dict["pz"] = {"tau":tau}
    with open(dict_file,'w') as fp:
        json.dump(tau_dict,fp, indent=4)
    return
