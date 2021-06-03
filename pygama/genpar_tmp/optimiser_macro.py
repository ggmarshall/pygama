import numpy as np
import os,json
from pygama.analysis import histograms as hist
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
#import pygama.genpar_tmp.cuts as cuts
sto = lh5.Store()

def run_optimisation(file,opt_config,dsp_config,fom,cuts):
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
    """
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

def fom_FWHM(tb_in,verbosity):
    Energies=tb_in['zacEmax'].nda
    hist, bins, var = pgh.get_hist(Energies, dx = 2, range = (40000,42000))
    fwhm, uncertainty = pgh.get_fwhm(hist,bins,var)
    return fwhm