import numpy as np
import os,json
from pygama.analysis import histograms as hist
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
#import pygama.genpar_tmp.cuts as cuts
sto = lh5.Store()

def run_optimisation(file,opt_config,dsp_config,fom,cuts):
    grid = set_par_space(list(opt_config.keys())[0],opt_config[list(opt_config.keys())[0]],init_args = True)
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts,verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms } )
    return opt.run_grid(tb_data,dsp_config,grid,fom,verbosity=0)

def set_par_space(processor, par_values,init_args=False):
    par_space = opt.ParGrid()
    for ind in par_values.keys():
        str_vals = set_values(par_values[ind])
        par_space.add_dimension(processor,int(ind),str_vals,init_arg = init_args)
    return par_space
    
def set_values(par_values):
    string_values=np.linspace(par_values['start'],par_values['end'],par_values['frequency'])
    string_values = [ f'{val:.2f}{par_values["unit"]}' for val in string_values]
    return string_values

def build_energy_cuts(file,Emin,Emax):
    wf_maxes = sto.read_object('/raw/wf_max', file,verbosity=0)[0].nda
    fpga_bl = sto.read_object('/raw/energy',file,verbosity=0)[0].nda
    wf_maxes -=fpga_bl
    energy_cuts = []
    for i in range(len(wf_maxes)):
        if wf_maxes[i] > Emin and wf_maxes[i] < Emax:
            energy_cuts.extend([i])
    return energy_cuts    

def fom_FWHM(tb_in,verbosity):
    #change to use pygama hist
    Energies=tb_in['zacEmax'].nda
    heights,bins,_ = plt.hist(Energies, bins=np.linspace(40500,41500,251),histtype='step')
    peak = max(heights)
    halfmax = peak/2
    for i in range(1,len(heights)):
        if heights[i]<halfmax and heights[i+1]>=halfmax:
            left = bins[i+1]
            break
    for i in range(len(heights)):
        if heights[-i]<halfmax and heights[-i-1]>=halfmax:
            right = bins[-i-1]
            break
    plt.close()
    return (100*(right-left)/(bins[np.argmax(heights)]))