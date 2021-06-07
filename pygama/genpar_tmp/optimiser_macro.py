import numpy as np
import os,json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
#import pygama.genpar_tmp.cuts as cuts
sto = lh5.Store()

def run_optimisation(file,opt_config,dsp_config, cuts, fom, **fom_kwargs):
    """
    Runs optimisation on .lh5 file
    
    Parameters
    ----------
    file: string
        path to raw .lh5 file
    opt_config: str
        path to JSON dictionary to configure optimisation
    dsp_config: str
        path to JSON dictionary specifying dsp configuration
    fom: function
        When given the output lh5 table of a DSP iteration, the
        fom_function must return a scalar figure-of-merit value upon which the
        optimization will be based. Should accept verbosity as a second argument  
    fom_kwargs: any
        Any keyword arguments to be passed to the fom  
    """
    grid = set_par_space(list(opt_config.keys())[0],opt_config[list(opt_config.keys())[0]])
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts,verbosity=0)[0]
    baseline = sto.read_object('/raw/baseline', file,idx=cuts,verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms , 'baseline':baseline} )
    return opt.run_grid(tb_data,dsp_config,grid,fom, **fom_kwargs, verbosity=0)

def set_par_space(processor, par_values,init_args=False):
    par_space = opt.ParGrid()
    for arg_t in par_values.keys():
        p_values = par_values[arg_t]
        for ind in p_values.keys():
            str_vals = set_values(p_values[ind])
            par_space.add_dimension(processor, int(ind),str_vals,arg_type=arg_t)
    return par_space
    
def set_values(par_values):
    string_values=np.linspace(par_values['start'],par_values['end'],par_values['frequency'])
    string_values = [ f'{val:.2f}*{par_values["unit"]}' for val in string_values]
    return string_values

def build_energy_mask(file, Emin, Emax):

    """
    creates and returns array mask for specified energy range using maxes of
    raw waveforms

    Parameters
    ----------
    File: str
        path to raw .lh5 file
    Emin: float
        energy selection lower bound
    Emax: float
        energy selection upper bound

    """
    wf_maxes = sto.read_object('/raw/wf_max', file, verbosity=0)[0].nda
    fpga_bl = sto.read_object('/raw/energy', file, verbosity=0)[0].nda
    wf_maxes -= fpga_bl
    return (wf_maxes >= Emin) & (wf_maxes <= Emax)

def fom_FWHM(tb_in,verbosity, parameter= None):
    Energies=tb_in[parameter].nda
    hist, bins, var = pgh.get_hist(Energies, dx = 2, range = (40000,42000))
    fwhm, uncertainty = pgh.get_fwhm(hist,bins,var)
    return fwhm
