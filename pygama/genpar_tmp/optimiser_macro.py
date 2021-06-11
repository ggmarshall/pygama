import numpy as np
import os,json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
#import pygama.genpar_tmp.cuts as cuts
sto = lh5.Store()

def run_optimisation(file,opt_config,dsp_config, cuts, fom, db_dict=None, **fom_kwargs):
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
    grid = set_par_space(opt_config)
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts,verbosity=0)[0]
    baseline = sto.read_object('/raw/baseline', file,idx=cuts,verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms, 'baseline':baseline } )
    return opt.run_grid(tb_data,dsp_config,grid, fom, db_dict, verbosity=0, **fom_kwargs)

def set_par_space(opt_config):
    par_space = opt.ParGrid()
    for name in opt_config.keys():
        p_values = opt_config[name]
        for param in p_values.keys():
            str_vals = set_values(p_values[param])
            par_space.add_dimension(name, param,str_vals)
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

def fom_FWHM(tb_in,verbosity, parameter):
    Energies=tb_in[parameter].nda
    bin_width = 0.5
    lower_bound = (np.amin(Energies)//bin_width) * bin_width
    upper_bound = ((np.amax(Energies)//bin_width)+1) * bin_width
    hist, bins, var = pgh.get_hist(Energies, dx = bin_width, range = (lower_bound,upper_bound))
    fwhm, uncertainty = pgh.get_fwhm(hist,bins,var)
    max_idx = np.argmax(hist)
    bin_centres = pgh.get_bin_centers(bins)
    max_val = bin_centres[max_idx]
    return fwhm/max_val

def fom_FWHM_with_dt_corr(tb_in,verbosity, parameter):
    Energies=tb_in[parameter].nda
    dt = tb_in['wf_max'].nda - tb_in['tp_0_est'].nda
    if np.isnan(Energies).any(): return np.nan
    if np.isnan(dt).any(): return np.nan
    alphas = np.linspace(0,4*10**-6,num=1000)
    fwhms = np.empty(len(alphas))
    for i,alpha in enumerate(alphas):
        ct_energy = (1+alpha*dt)*Energies
        bin_width = 0.5
        lower_bound = (np.amin(ct_energy)//bin_width) * bin_width
        upper_bound = ((np.amax(ct_energy)//bin_width)+1) * bin_width
        hist, bins, var = pgh.get_hist(ct_energy, dx = bin_width, range = (lower_bound,upper_bound))
        max_idx = np.argmax(hist)
        bin_centres = pgh.get_bin_centers(bins)
        max_val = bin_centres[max_idx]
        try:
            fwhm, uncertainty = pgh.get_fwhm(hist,bins,var)
            fwhms[i] = fwhm/max_val
        except:
            return np.amin(fwhms)    
    return np.amin(fwhms)
