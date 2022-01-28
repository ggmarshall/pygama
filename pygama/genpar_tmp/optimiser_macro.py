import numpy as np
import os,json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
import pygama.analysis.peak_fitting as pgf
import pygama.analysis.calibration as pgc
import pygama.genpar_tmp.cuts as cts
import pickle as pkl
import glob
from iminuit import Minuit, cost, util
import sys
from scipy.optimize import minimize, curve_fit, minimize_scalar, brentq
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import chisquare

sto = lh5.Store()

def run_optimisation(file,opt_config,dsp_config, cuts, fom, db_dict=None, n_events=8000, **fom_kwargs):
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
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts, n_rows = n_events, verbosity=0)[0]
    baseline = sto.read_object('/raw/baseline', file,idx=cuts, n_rows = n_events, verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms, 'baseline':baseline } )
    return opt.run_grid(tb_data,dsp_config,grid, fom, db_dict, verbosity=0, **fom_kwargs)

def run_optimisation_multiprocessed(file,opt_config,dsp_config, cuts, fom, db_dict=None, processes=5, verbosity=0, n_events=8000, **fom_kwargs):
    """
    Runs optimisation on .lh5 file, this version multiprocesses the grid points, it also can handle multiple grids being passed 
    as long as they are the same dimensions.
    
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
    n_events : int
        Number of events to run over
    processes : int
        Number of speperate processes to run for the multiprocessing 
    """
    def form_dict(in_dict, length):
        keys = list(in_dict.keys())
        out_list =[]
        for i in range(length):
            out_list.append({keys[0]:0}) 
        for key in keys:
            if isinstance(in_dict[key],list):
                if len(in_dict[key]) == length:
                    for i in range(length):
                        out_list[i][key] = in_dict[key][i]
                else:
                    for i in range(length):
                        out_list[i][key] = in_dict[key] 
            else:
                for i in range(length):
                    out_list[i][key] = in_dict[key]
        return out_list  

    if not isinstance(opt_config, list):
        opt_config = [opt_config]
    grid = []
    for i,opt_conf in enumerate(opt_config):
        grid.append(set_par_space(opt_conf)) 
    if fom_kwargs:
        if 'fom_kwargs' in fom_kwargs:
            fom_kwargs = fom_kwargs['fom_kwargs']
        fom_kwargs = form_dict(fom_kwargs, len(grid))    
    sto=lh5.Store()
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts, n_rows = n_events, verbosity=0)[0]
    baseline = sto.read_object('/raw/baseline', file,idx=cuts, n_rows = n_events, verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms, 'baseline':baseline } )
    return opt.run_grid_multiprocess_parallel(tb_data,dsp_config,grid, fom, 
                                 db_dict=db_dict,processes=processes, 
                                 verbosity=verbosity, fom_kwargs=fom_kwargs)

def set_par_space(opt_config):
    """
    Generates grid for optimizer from dictionary of form {param : {start: , end: , frequency: }}
    """
    par_space = opt.ParGrid()
    for name in opt_config.keys():
        p_values = opt_config[name]
        for param in p_values.keys():
            str_vals = set_values(p_values[param])
            par_space.add_dimension(name, param,str_vals)
    return par_space
    
def set_values(par_values):
    """
    Finds values for grid
    """
    string_values=np.linspace(par_values['start'],par_values['end'],par_values['frequency'])
    try:
        string_values = [ f'{val:.2f}*{par_values["unit"]}' for val in string_values]
    except:
        string_values = [ f'{val:.2f}' for val in string_values]
    return string_values


def simple_guess(hist, bins, var, func_i):
    """
    Simple guess for peak fitting
    """
    if func_i == pgf.radford_pdf:
        bin_cs = (bins[1:]+bins[:-1])/2
        #hist = (hist[1:-1]+hist[2:]+hist[:-2])/3
        _, sigma, amp = pgh.get_gaussian_guess(hist,bins)
        i_0 = np.nanargmax(hist)
        mu = bin_cs[i_0]
        height = hist[i_0]
        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1./5
        tau = sigma
        height -= (bg0 + step/2)
        amp = height / (htail*0.87/35 + (1-htail)/(sigma*np.sqrt(2*np.pi)))
        hstep = step/(2*amp)
        dx = np.diff(bins)[0]
        nsig_guess = np.sum(hist[int(i_0-4*dx*sigma):int(i_0+4*dx*sigma)])
        nbkg_guess = np.sum(hist)-nsig_guess
        parguess = [nsig_guess,nbkg_guess ,mu, sigma, hstep,  tau] #htail,
        return parguess
    
    elif func_i == pgf.gauss_step_pdf:
        mu, sigma, amp = pgh.get_gaussian_guess(hist,bins)
        i_0 = np.argmax(hist)
        height = hist[i_0]
        bg = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg
        tau = 6.*sigma
        height -= (bg + step/2)
        amp = height * sigma * np.sqrt(2 * np.pi)
        dx = np.diff(bins)[0]
        nsig_guess = np.sum(hist[int(i_0-4*dx*sigma):int(i_0+4*dx*sigma)])
        nbkg_guess = np.sum(hist)-nsig_guess
        return [nsig_guess,nbkg_guess, mu, sigma, step]

def unbinned_energy_fit(energy, func, verbose=False, display=0):
    bin_width = 1
    lower_bound = (np.nanmin(energy)//bin_width) * bin_width
    upper_bound = ((np.nanmax(energy)//bin_width)+1) * bin_width
    hist, bins,var = pgh.get_hist(energy,dx=bin_width, range= (lower_bound, upper_bound))
    bin_cs = (bins[:-1]+bins[1:])/2
    x0 = simple_guess(hist, bins, var, func)
    if verbose:print(x0)
    c = cost.ExtendedUnbinnedNLL(energy, func)
    m = Minuit(c, *x0)
    m.migrad()
    m.hesse()
    
    m_fit = func(bin_cs, *m.values)[1]
    cs = chisquare(hist, m_fit)[0]/len(m.values)
    m2 = Minuit(c, *x0)
    m2.simplex().migrad()
    m2.hesse()
    m2_fit = func(bin_cs, *m2.values)[1]
    cs2 = chisquare(hist, m2_fit)[0]/len(m2.values)
    
    if display >1:
        print(m.errors)
        print(m2.errors)
        plt.figure()
        plt.plot(bin_cs, hist, label='data')
        plt.plot(bin_cs, m2_fit, label='1 simplex')
        plt.plot(bin_cs, m_fit, label='0 simplex')
        plt.legend()
        plt.show()
        
    frac_errors1 = np.sum(np.abs(np.array(m.errors)/np.array(m.values)))
    frac_errors2 = np.sum(np.abs(np.array(m2.errors)/np.array(m2.values)))
    
    if (np.isnan(m.errors).all() and np.isnan(m2.errors).all()):
        print("extra simplex needed")
        m = Minuit(c, *x0)
        m.simplex().simplex().migrad()
        m.hesse()
        m_fit = func(bin_cs, *m.values)[1]
        cs = chisquare(hist, m_fit)[0]/len(m.values)
        
        return m.values, m.errors, m.covariance, cs
    
    elif (frac_errors1>30 and frac_errors2>30):
        
        m3 = Minuit(c, *x0)
        m3.simplex().simplex().migrad()
        m3.hesse()
        m3_fit = func(bin_cs, *m.values)[1]
        cs3 = chisquare(hist, m_fit)[0]/len(m.values)
        frac_errors3 = np.sum(np.abs(np.array(m3.errors)/np.array(m3.values)))
        
        if (frac_errors3<frac_errors1*0.9) and (frac_errors3<frac_errors2*0.9):
            return m3.values, m3.errors, m3.covariance, cs3

    if np.isnan(m2.errors).all() or cs*1.1 < cs2:
        return m.values, m.errors, m.covariance, cs
    
    elif np.isnan(m.errors).all() or cs2*1.1 < cs:
        return m2.values, m2.errors, m2.covariance, cs2

    elif frac_errors1 < frac_errors2:
        return m.values, m.errors, m.covariance, cs
    
    elif frac_errors1 > frac_errors2:
        return m2.values, m2.errors, m2.covariance, cs2
    

    else:
        raise RuntimeError

def get_peak_fwhm_with_dt_corr(Energies, alpha,dt, func, peak, kev_width, kev=False, display=0):
    """
    Applies the drift time correction and fits the peak returns the fwhm, max, err. Can return result in ADC or keV
    """

    correction = np.multiply(np.multiply(alpha,dt, dtype='float64'),Energies, dtype='float64')
    ct_energy = np.add(correction, Energies)
    
    bin_width = 1
    lower_bound = (np.nanmin(ct_energy)//bin_width) * bin_width
    upper_bound = ((np.nanmax(ct_energy)//bin_width)+1) * bin_width
    hist, bins, var = pgh.get_hist(ct_energy, dx = bin_width, range = (lower_bound,upper_bound))  
    mu = bins[np.nanargmax(hist)]
    adc_to_kev = mu/peak
    # Making the window slightly smaller removes effects where as mu moves edge can be outside bin width
    lower_bound = mu - (kev_width[0]* adc_to_kev) 
    upper_bound = mu + (kev_width[1]* adc_to_kev)
    win_idxs = (ct_energy>lower_bound) &(ct_energy<upper_bound)
    try:   
    
        if display >0:
            energy_pars, energy_err, cov, chisqr = unbinned_energy_fit(ct_energy[win_idxs], func, verbose=True, display=display)
            print(energy_pars)
            print(energy_err) 
            print(cov)
            plt.figure()
            xs = np.arange(lower_bound, upper_bound, bin_width)
            hist, bins, var = pgh.get_hist(ct_energy, dx = bin_width, range = (lower_bound,upper_bound))
            plt.plot((bins[1:]+bins[:-1])/2, hist)
            plt.plot(xs , func(xs, *energy_pars)[1])
            plt.show()
        else:
            energy_pars, energy_err, cov, chisqr = unbinned_energy_fit(ct_energy[win_idxs], func)
        if func == pgf.radford_pdf:
            xs = np.arange(lower_bound,upper_bound+1, 1)
            #try: 
            fwhm= pgf.radford_pdf_fwhm(energy_pars[3], np.abs(energy_pars[5]))

        elif func == pgf.gauss_step_pdf:
            fwhm = energy_pars[3]*2*np.sqrt(2*np.log(2))
            fwhm_err = np.sqrt(cov[3][3])*2*np.sqrt(2*np.log(2))

        xs = np.arange(lower_bound, upper_bound, 0.1)
        y = func(xs, *energy_pars)[1]
        max_val = np.amax(y)
        
        fwhm_o_max = fwhm/max_val
        
        rng = np.random.default_rng(1)
        # generate set of bootstrapped parameters
        par_b = rng.multivariate_normal(energy_pars, cov, size=100)
        y_max = np.array([func(xs, *p)[1] for p in par_b])
        maxs = np.nanmax(y_max, axis=1)
        
        yerr_boot = np.nanstd(y_max, axis=0)

        if func == pgf.radford_pdf:
            y_b = np.zeros(len(par_b))
            for i,p in enumerate(par_b):
                y_b[i] = pgf.radford_pdf_fwhm(p[3],np.abs(p[5]))
            fwhm_err = np.nanstd(y_b, axis=0)
            fwhm_o_max_err = np.nanstd(y_b/maxs, axis=0)
        else:
            
            max_err = np.nanstd(maxs)

            fwhm_o_max_err = fwhm_o_max*np.sqrt((np.array(fwhm_err)/np.array(fwhm))**2+(np.array(max_err)/np.array(max_val))**2)

        if display >1:
            plt.figure()
            plt.plot((bins[1:]+bins[:-1])/2, hist)
            for i in range(100):
                plt.plot(xs, y_max[i,:])
            plt.show()
        
        if display >0:
            plt.figure()
            hist, bins, var = pgh.get_hist(ct_energy, dx = bin_width, range = (lower_bound,upper_bound))
            plt.plot((bins[1:]+bins[:-1])/2, hist)
            plt.plot(xs , func(xs, *energy_pars)[1])
            plt.fill_between(xs, y - yerr_boot, y + yerr_boot, facecolor="C1", alpha=0.5)
            plt.show()

    except:
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    if kev==True:
        fwhm *= (peak/energy_pars[2])
        fwhm_err *= (peak/energy_pars[2])

    return fwhm, fwhm_o_max, fwhm_err,fwhm_o_max_err, chisqr, energy_pars[0], energy_err[0]

def fom_FWHM_with_dt_corr_fit(tb_in,verbosity, kwarg_dict, ctc_parameter, display=0):
    """
    Fom for sweeping over ctc values to find the best value, returns the best found fwhm 
    """
    parameter = kwarg_dict['parameter']
    func = kwarg_dict['func']
    Energies=tb_in[parameter].nda
    Energies = Energies.astype('float64')
    peak = kwarg_dict['peak']
    kev_width = kwarg_dict['kev_width']
    min_alpha=0
    max_alpha = 3.5e-06
    astep= 1.25e-07
    if ctc_parameter == 'QDrift':
        dt = tb_in['dt_eff'].nda 
    elif ctc_parameter == 'dt':
        dt = np.subtract(tb_in['tp_99'].nda , tb_in['tp_0_est'].nda, dtype='float64')
    elif ctc_parameter == 'rt':
        dt = np.subtract(tb_in['tp_99'].nda,tb_in['tp_01'].nda, dtype='float64')
    if np.isnan(Energies).any(): return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan}
    if np.isnan(dt).any(): 
        return {'fwhm':np.nan,'fwhm_err':np.nan, 'alpha':np.nan}
    alphas = np.linspace(0,3.5*10**-6,num=29) # np.arange(min_alpha,max_alpha+astep, astep,  dtype='float64')
    fwhms = np.array([])
    final_alphas = np.array([])
    fwhm_errs = np.array([])
    for alpha in alphas:
        _, fwhm_o_max,_,fwhm_o_max_err,_,_,_ = get_peak_fwhm_with_dt_corr(Energies, alpha,dt, func,peak,kev_width) #,_,fwhm_o_max_err
        if not np.isnan(fwhm_o_max):
            fwhms = np.append(fwhms,fwhm_o_max)
            final_alphas = np.append(final_alphas, alpha) 
            fwhm_errs = np.append(fwhm_errs, fwhm_o_max_err)
    
    # Make sure fit isn't based on only a few points
    if len(fwhms)< 10:
        print("less than 10 fits successful")
        return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan} 

    ids = fwhm_errs < 5*np.mean(sorted(fwhm_errs)[:5])
    # Fit alpha curve to get best alpha
    
    try:
        alphas = np.arange(final_alphas[ids][0],final_alphas[ids][-1],astep/20)#num=
        fit = np.polynomial.polynomial.polyfit(final_alphas[ids], fwhms[ids], w=1/fwhm_errs[ids], deg= 4)
        fit_vals = np.polynomial.polynomial.polyval(alphas,fit)
        alpha = alphas[np.nanargmin(fit_vals)]
        if display >0:
            plt.figure()
            plt.errorbar(final_alphas, fwhms, yerr = fwhm_errs, linestyle = ' ')
            plt.plot(alphas, fit_vals)
            plt.show()
            
    except:
        print("alpha fit failed")
        return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan}
    if np.isnan(fit_vals).all():
        print("alpha fit all nan")
        return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan}
    else:
        # Return fwhm of optimal alpha in kev with error
        final_fwhm, _, final_err,_, csqr, n_sig, n_sig_err = get_peak_fwhm_with_dt_corr(Energies, alpha,dt, func,
                                                                          peak,kev_width, kev=True, display=display)
        if np.isnan(final_fwhm) or np.isnan(final_err):
            print(f"final fit failed, alpha was {alpha}")
        return {'fwhm': final_fwhm,
                'fwhm_err': final_err,
                'alpha':alpha,
                'chisquare':csqr,
                'n_sig':n_sig,
                'n_sig_err':n_sig_err
                } 


def fom_all_fit(tb_in,verbosity, kwarg_dict):
    """
    fom to run over different ctc parameters
    """
    ctc_parameters = ['dt', 'QDrift']
    output_dict = {}
    for param in ctc_parameters:
        out = fom_FWHM_with_dt_corr_fit(tb_in,verbosity, kwarg_dict, param)
        output_dict[param] = out
    return output_dict

def fom_FWHM_fit(tb_in,verbosity, kwarg_dict):
    """
    fom with no ctc sweep
    """
    parameter = kwarg_dict['parameter']
    func = kwarg_dict['func']
    Energies=tb_in[parameter].nda
    Energies = Energies.astype('float64')
    peak = kwarg_dict['peak']
    kev_width = kwarg_dict['kev_width']
    #alpha = kwarg_dict['alpha']
    if np.isnan(Energies).any(): return {'fwhm':np.nan, 'fwhm_err':np.nan}

    _, final_fwhm_o_max, _,final_fwhm_o_max_err, csqr, n_sig, n_sig_err  = get_peak_fwhm_with_dt_corr(Energies, 0,0, 
                                                  func, peak=peak, kev_width=kev_width, kev=True)
    return {'fwhm_o_max':final_fwhm_o_max,
            'max_o_fwhm':final_fwhm_o_max_err,
            'chisquare':csqr,
            'n_sig':n_sig,
            'n_sig_err':n_sig_err} 

def event_selection(raw_file, dsp_config, db_dict, peaks_keV, peak_idx, kev_width):
    """
    Finds the indexes of events in the peak after cuts to run optimizer on, uses raw files
    """
    sto=lh5.Store()
    baseline = sto.read_object('/raw/baseline', raw_file,verbosity=0, n_rows=5*10**6)[0].nda
    wf_max = sto.read_object('/raw/wf_max', raw_file,verbosity=0, n_rows=5*10**6)[0].nda
    rough_energy = wf_max-baseline
    # Get events around peak using raw file values
    peak = peaks_keV[peak_idx]
    guess_keV = 1/18
    Euc_min = peaks_keV[0]/guess_keV * 0.6
    Euc_max = peaks_keV[-1]/guess_keV * 1.1
    dEuc = 1/guess_keV
    hist, bins, var = pgh.get_hist(rough_energy, range=(Euc_min, Euc_max), dx=dEuc)
    detected_peaks_locs, detected_peaks_keV, roughpars = pgc.hpge_find_E_peaks(hist, bins, var, peaks_keV)
    try:
        if peak not in detected_peaks_keV:
            raise ValueError
        detected_peak_idx = np.where(detected_peaks_keV == peak)[0]
        peak_loc = detected_peaks_locs[detected_peak_idx]
        print(detected_peaks_keV)
        print(detected_peaks_locs)
        print(f'{peak} peak found at {peak_loc}')
        rough_adc_to_kev = roughpars[0]
        e_lower_lim = peak_loc - (1.1*kev_width[0])/rough_adc_to_kev
        e_upper_lim = peak_loc + (1.1*kev_width[1])/rough_adc_to_kev
    except:
        print("Peak not found attempting to use rough parameters")
        peak_loc = ((peak- roughpars[1]) / roughpars[0])
        rough_adc_to_kev = roughpars[0]
        e_lower_lim = peak_loc - (1.5*kev_width[0])/rough_adc_to_kev
        e_upper_lim = peak_loc + (1.5*kev_width[1])/rough_adc_to_kev
    print(e_lower_lim, e_upper_lim)
    e_mask = (rough_energy>e_lower_lim)&(rough_energy<e_upper_lim)
    e_idxs = np.where(e_mask)[0]
    print(len(e_idxs))
    waveforms = sto.read_object('/raw/waveform', raw_file,verbosity=0, idx=e_idxs, n_rows=30000)[0]
    baseline = sto.read_object('/raw/baseline', raw_file,verbosity=0, idx=e_idxs, n_rows=30000)[0]
    input_data = lh5.Table(col_dict = { 'waveform' : waveforms, 'baseline':baseline } )
    print("Processing data")
    tb_data = opt.run_one_dsp(input_data, dsp_config, db_dict=db_dict)
    parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}
    cut_dict = cts.generate_cuts(tb_data, parameters)
    print(cut_dict)
    print('Loaded Cuts')
    ct_mask = cts.get_cut_indexes(tb_data, cut_dict, 'raw')
    wf_idxs = e_idxs[:30000][ct_mask]
    energy = tb_data['trapEmax'].nda[ct_mask]
    hist, bins, params, covs = fit_peak_func(energy, func_i= gauss_step, peak=peak, kev_width=kev_width)
    updated_adc_to_kev = peak/params[1]
    e_lower_lim = params[1] - (kev_width[0])/updated_adc_to_kev
    e_upper_lim = params[1] + (kev_width[1])/updated_adc_to_kev
    print(e_lower_lim, e_upper_lim)
    final_mask = (energy>e_lower_lim)&(energy<e_upper_lim)
    final_events = wf_idxs[final_mask]
    return final_events

def slice_dict(in_dict, n):
    out_dict = {}
    for par in in_dict:
        out_dict[par]=in_dict[par][:n]
    return out_dict

def load_grids(files, parameter):
    """
    Loads in optimizer grids
    """
    peak_grids = []
    for file in files:
        with open(file,"rb") as d:
            grid = pkl.load(d)
        peak_grids.append(grid[parameter])
    return peak_grids

def load_config(path, filter_name):
    """
    Loads in optimizer configs
    """
    if filter_name =='Cusp':
        opt_config = os.path.join(path, 'cusp_config.json')
    elif filter_name =='Zac':
        opt_config = os.path.join(path, 'zac_config.json')
    elif filter_name =='Trap':
        opt_config = os.path.join(path, 'etrap_config.json')
    elif filter_name =='TTrap':
        opt_config = os.path.join(path, 'ttrap_config.json')
    with open(opt_config, 'r') as o:
        opt_dict = json.load(o)
    return opt_dict

def get_ctc_grid(grids, ctc_param):
    """
    Reshapes optimizer grids to be in easier form
    """
    error_grids = []
    dt_grids = []
    alpha_grids=[]
    for grid in grids:
        shape=grid.shape
        dt_grid = np.ndarray(shape=shape)
        alpha_grid = np.ndarray(shape=shape)
        error_grid = np.ndarray(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dt_grid[i,j]= grid[i,j][ctc_param]['fwhm']
                
                error_grid[i,j]= grid[i,j][ctc_param]['fwhm_err']
                try:
                    alpha_grid[i,j]= grid[i,j][ctc_param]['alpha']
                except:
                    pass
        dt_grids.append(dt_grid)
        alpha_grids.append(alpha_grid)
        error_grids.append(error_grid)
    return dt_grids, error_grids , alpha_grids

def interpolate_energy(peak_energies, grids, error_grids, energy):
    """
    Interpolates fwhm vs energy for every grid point
    """

    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    out_grid_err = np.empty(grid_shape)
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        err_points = np.array([error_grids[i][index] for i in range(len(grids))])
        nan_mask = np.isnan(points)
        nan_mask = nan_mask | (points<0) | (0.1*points<err_points)
        try:
            if len(points[nan_mask])>2:
                raise ValueError
            elif nan_mask[-1] == True or nan_mask[-2] == True:
                raise ValueError
            param_guess  = [0.2,0.001,0.000001]
            #param_bounds = (0, [10., 1. ])#,0.1
            fit_pars, fit_covs = curve_fit(fwhm_slope, peak_energies[~nan_mask],points[~nan_mask], sigma=err_points[~nan_mask], 
                               p0=param_guess,  absolute_sigma=True) #bounds=param_bounds,
            fit_qbb = fwhm_slope(energy,*fit_pars)
            sderrs = np.sqrt(np.diag(fit_covs))
            qbb_err = fwhm_slope(energy,*(fit_pars+sderrs))-fwhm_slope(energy,*fit_pars)
            out_grid[index] = fit_qbb
            out_grid_err[index] = qbb_err
        except:
            out_grid[index] = np.nan
            out_grid_err[index] = np.nan
    return out_grid, out_grid_err

def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x + m2*(x**2))

def find_lowest_grid_point_save(grid, err_grid, opt_dict):
    """
    Finds the lowest grid point, if more than one with same value returns shortest filter.
    """
    opt_name = list(opt_dict.keys())[0]
    keys = list(opt_dict[opt_name].keys())
    param_list = []
    shape = []
    db_dict={}
    for key in keys:
        param_dict = opt_dict[opt_name][key]
        grid_axis=np.linspace(param_dict['start'],param_dict['end'],param_dict['frequency'])
        param_list.append(grid_axis)
        shape.append(len(grid_axis))
       
    total_lengths = np.zeros(shape)
    
    for index, x in np.ndenumerate(total_lengths):
        for i,param in enumerate(param_list):
            total_lengths[index]+= param[index[i]]
    min_val = np.nanmin(grid)
    lowest_ixs = np.where((grid == min_val))
    #print(f'Minimum value is : {min_val} +- {err_grid[lowest_ixs][0]}')
    try:
        fwhm_dict =  {'fwhm': min_val, 'fwhm_err': err_grid[lowest_ixs][0]}
    except:
        fwhm_dict =  {'fwhm': np.nan, 'fwhm_err': np.nan}
        return np.nan, fwhm_dict, np.nan
    #print(lowest_ixs)
    if len(lowest_ixs[0]) ==1:
        for i,key in enumerate(keys):
            #print(f'{key} : {param_list[i][lowest_ixs[i]][0]}us')
            if i ==0:
                db_dict[opt_name] = {key:f'{param_list[i][lowest_ixs[i]][0]}*us'}
            else:
                db_dict[opt_name][key] = f'{param_list[i][lowest_ixs[i]][0]}*us'
    else:
        shortest_length = np.argmin(total_lengths[lowest_ixs])
        final_idxs = [lowest_ix[shortest_length] for lowest_ix in lowest_ixs]
        for i,key in enumerate(keys):
            #print(f'{key} : {param_list[i][final_idxs[i]]}us')
            db_dict[opt_name] = {key:f'{param_list[i][lowest_ixs[i]][0]}*us'}
    return lowest_ixs, fwhm_dict, db_dict

def interpolate_grid(energies, grids, int_energy, deg):
    """
    Interpolates energy vs parameter for every grid point using polynomial.
    """
    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        nan_mask = np.isnan(points)
        nan_mask = nan_mask | (points<0)
        try:
            if len(points[~nan_mask])<3:
                raise IndexError
            fit_point = np.polynomial.polynomial.polyfit(energies[~nan_mask],points[~nan_mask], deg=deg)
            out_grid[index] = np.polynomial.polynomial.polyval(int_energy, fit_point)
        except:
            out_grid[index] = np.nan
    return out_grid

def get_best_vals(peak_grids, peak_energies, param, opt_dict, save_path=None, det=None):
    """
    Finds best filter parameters
    """
    dt_grids, error_grids, alpha_grids = get_ctc_grid(peak_grids, param)
    qbb_grid, qbb_errs = interpolate_energy(peak_energies, dt_grids, error_grids, 2039.061)
    qbb_alphas = interpolate_grid(peak_energies[2:], alpha_grids[2:], 2039.061, 1)
    ixs, fwhm_dict, db_dict = find_lowest_grid_point_save(qbb_grid, qbb_errs, opt_dict)
    out_grid = {'fwhm':qbb_grid, 'fwhm_err':qbb_errs, 'alphas':qbb_alphas}

    if isinstance(save_path,str):
        mpl.use('pdf')
        e_param = list(opt_dict.keys())[0]
        opt_dict = opt_dict[e_param]

        detector = save_path.split('/')[-1]
        save_path = os.path.join(save_path, f"{e_param}-{param}.pdf")
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        with PdfPages(save_path) as pdf:
            
            keys = list(opt_dict.keys())
            print(keys)
            x_dict = opt_dict[keys[1]]
            xs=(np.arange(0,x_dict['frequency'],1),np.linspace(x_dict['start'],x_dict['end'],x_dict['frequency']))
            y_dict = opt_dict[keys[0]]
            ys=(np.arange(0,y_dict['frequency'],1),np.linspace(y_dict['start'],y_dict['end'],y_dict['frequency']))
            for i,x in enumerate(xs[1]):
                xs[1][i] = round(x,1)
            for i,y in enumerate(ys[1]):
                ys[1][i] = round(y,1)
        
            points = [dt_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(dt_grids))]
            err_points = [error_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(error_grids))]
            alpha_points = [alpha_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(alpha_grids))]
            param_guess  = [0.2,0.001,0.000001]
            #param_bounds = (0, [10., 1. ]),0.1
            fit_pars, fit_covs = curve_fit(fwhm_slope, peak_energies,points, sigma=err_points, 
                                        p0=param_guess,  absolute_sigma=True) #bounds=param_bounds,
            energy_x = np.arange(200,2600,10)
            plt.rcParams['figure.figsize'] = (12, 18)
            plt.rcParams['font.size'] = 12
            plt.figure()
            for i, dt_grid in enumerate(dt_grids):
                plt.subplot(3,2,i+1)
                plt.imshow(dt_grid, norm=LogNorm(vmin=np.nanmin(np.abs(dt_grid)), vmax=np.nanpercentile(dt_grid,98)), cmap='viridis')
                    
                plt.xticks(xs[0],xs[1])
                plt.yticks(ys[0],ys[1])

                plt.xlabel(f'{keys[1]} (us)')
                plt.ylabel(f'{keys[0]} (us)')
                plt.title(f'{peak_energies[i]:.1f} kev')
                plt.xticks(rotation=45)
                cbar = plt.colorbar()
                cbar.set_label("FWHM (keV)")
            plt.tight_layout()
            plt.suptitle(f"{det}-{e_param}-{param}")
            pdf.savefig()
            plt.close()

            plt.figure()

            plt.imshow(qbb_grid, norm=LogNorm(vmin=np.nanmin(qbb_grid), vmax=np.nanpercentile(dt_grid,98)), cmap='viridis')
            plt.xlabel(f'{keys[1]} (us)')
            plt.ylabel(f'{keys[0]} (us)')
            plt.title(f'Qbb')
            plt.xticks(rotation=45)
            cbar = plt.colorbar()
            cbar.set_label("FWHM (keV)")
            plt.tight_layout()
            plt.suptitle(f"{det}-{e_param}-{param}")
            pdf.savefig()
            plt.close()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
            ax1.errorbar(peak_energies,points, yerr=err_points, fmt= ' ')
            ax1.plot(energy_x, fwhm_slope(energy_x,*fit_pars))
            ax1.errorbar([2039],qbb_grid[ixs[0], ixs[1]], yerr=qbb_errs[ixs[0], ixs[1]], fmt= ' ')
            ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)
            ax2.scatter(peak_energies,(points-fwhm_slope(peak_energies,*fit_pars))/err_points, lw=1, c='b')
            ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
            ax2.set_ylabel("Standardised Residuals", ha='right', y=1)
            fig.suptitle(f"{det}-{e_param}-{param}")
            pdf.savefig()
            plt.close()

            try:
                alphas = qbb_alphas[ixs[0], ixs[1]][0]
                if isinstance(save_path,str):
                    alpha_fit = np.polynomial.polynomial.polyfit(peak_energies[2:], alpha_points[2:], deg=1)
                    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
                    ax1.scatter(peak_energies[:],alpha_points[:])
                    ax1.plot(peak_energies[2:], np.polynomial.polynomial.polyval(peak_energies[2:], alpha_fit))
                    ax1.scatter([2039],qbb_alphas[ixs[0], ixs[1]])
                    ax1.set_ylabel("Charge Trapping Value", ha='right', y=1)
                    ax2.scatter(peak_energies[2:],(alpha_points[2:]-np.polynomial.polynomial.polyval(peak_energies[2:], alpha_fit))/alpha_points[2:], lw=1, c='b')
                    ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
                    ax2.set_ylabel("Residuals (%)", ha='right', y=1)
                    fig.suptitle(f"{det}-{param}")
                    pdf.savefig()
                    plt.close()
            except:
                alphas = np.nan
    else:
        try:
            alphas = qbb_alphas[ixs[0], ixs[1]][0]
        except:
            alphas = np.nan
    return alphas, fwhm_dict, db_dict, out_grid

def match_config(parameters, opt_dicts):
    """
    Matches config to parameters
    """
    out_dict = {}
    for opt_dict in opt_dicts:
        key = list(opt_dict.keys())[0]
        if key =='cusp':
            out_dict['cuspEmax'] = opt_dict
        elif key =='zac':
            out_dict['zacEmax'] = opt_dict
        elif key =='etrap':
            out_dict['trapEmax'] = opt_dict
    return out_dict

def get_filter_params(files, opt_dicts, save_path=None):
    """
    Finds best parameters for filter
    """

    full_db_dict = {}
    full_fwhm_dict = {}
    full_grids={}
    parameters =['zacEmax', 'trapEmax', 'cuspEmax']
    matched_configs = match_config(parameters, opt_dicts)
    peak_energies = np.array([])
    for f in files:
        filename = os.path.basename(f)
        peak_energy = float(filename.split('.p')[0])
        peak_energies = np.append(peak_energies, peak_energy)
    for param in parameters:
        opt_dict = matched_configs[param]
        peak_grids = load_grids(files, param)
        ctc_params= list(peak_grids[0][0,0].keys())
        ctc_dict = {}
        
        for ctc_param in ctc_params:
            if ctc_param == 'QDrift':
                alpha, fwhm, db_dict, output_grid = get_best_vals(peak_grids,peak_energies, ctc_param, opt_dict, 
                                                                    save_path=save_path)
                opt_name = list(opt_dict.keys())[0]
                db_dict[opt_name].update({'alpha':alpha})
                
            else:
                alpha,fwhm,_ , output_grid = get_best_vals(peak_grids, peak_energies, ctc_param, opt_dict,
                                                            save_path=save_path)
            try:
                full_grids[param][ctc_param] = output_grid
            except:
                full_grids[param] = {ctc_param:output_grid}
            fwhm.update({'alpha':alpha})
            ctc_dict[ctc_param] =fwhm
        full_fwhm_dict[param] = ctc_dict
        full_db_dict.update(db_dict)
    return full_db_dict, full_fwhm_dict, full_grids

def run_splitter(files):
    """
    Returns list containing lists of each run
    """

    if isinstance(files, str): files = [files]
    # Expand wildcards
    files = [f for f_wc in files for f in sorted(glob.glob(os.path.expandvars(f_wc)))]
    
    runs = []
    run_files = []
    for file in files:
        base=os.path.basename(file)
        file_name = os.path.splitext(base)[0]
        parts = file_name.split('-')
        run_no = parts[3]
        if run_no not in runs:
            runs.append(run_no)
            run_files.append([])
        for i,run in enumerate(runs):
            if run == run_no:
                run_files[i].append(file) 
    return run_files