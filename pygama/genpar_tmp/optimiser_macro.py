import numpy as np
import os,json
from pygama.analysis import histograms as pgh
import pygama.lh5 as lh5
import pygama.dsp.dsp_optimize as opt
from pygama.analysis.peak_fitting import radford_peak, gauss_step
import pygama.analysis.peak_fitting as pgf
import pygama.analysis.calibration as pgc
import pygama.genpar_tmp.cuts as cts
import pickle as pkl
from scipy.optimize import curve_fit

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

def run_optimisation_multiprocessed(file,opt_config,dsp_config, cuts, fom, db_dict=None, processes=5, verbosity=1, **fom_kwargs):
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
    if 'fom_kwargs' in fom_kwargs:
        fom_kwargs = fom_kwargs['fom_kwargs']
    if not isinstance(opt_config, list):
        opt_config = [opt_config]
    grid = []
    for i,opt_conf in enumerate(opt_config):
        grid.append(set_par_space(opt_conf)) 
    sto=lh5.Store()
    waveforms = sto.read_object('/raw/waveform', file,idx=cuts,verbosity=0)[0]
    baseline = sto.read_object('/raw/baseline', file,idx=cuts,verbosity=0)[0]
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms, 'baseline':baseline } )
    return opt.run_grid_multiprocess_parallel(tb_data,dsp_config,grid, fom, 
                                 db_dict=db_dict,processes=processes, 
                                 verbosity=verbosity, fom_kwargs=fom_kwargs)

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


def simple_guess(hist, bins, var, func_i):
    if func_i == pgf.radford_peak:
        mu, sigma, amp = pgh.get_gaussian_guess(hist,bins)
        i_0 = np.argmax(hist)
        height = hist[i_0]
        bg0 = np.sum(hist[-5:])/5
        step = np.sum(hist[:5])/5 - bg0
        htail = 1./5
        tau = 6.*sigma
        height -= (bg0 + step/2)
        amp = height / (htail*0.87/35 + (1-htail)/(sigma*np.sqrt(2*np.pi)))
        hstep = step/(2*amp)

        parguess = [mu, sigma, hstep, htail, tau, bg0, amp]

        return parguess
    elif func_i == pgf.gauss_step:
        mu, sigma, amp = pgh.get_gaussian_guess(hist,bins)
        i_0 = np.argmax(hist)
        height = hist[i_0]
        bg = np.sum(hist[-5:])/5
        step = np.sum(hist[:5])/5 - bg
        tau = 6.*sigma
        height -= (bg + step/2)
        amp = height * sigma * np.sqrt(2 * np.pi)

        return [amp, mu, sigma, bg, step]
        

def fit_peak_func(energies, peak, kev_width, func_i= pgf.gauss_step):
    bin_width = 1
    lower_bound = (np.nanmin(energies)//bin_width) * bin_width
    upper_bound = ((np.nanmax(energies)//bin_width)+1) * bin_width
    hist, bins, var = pgh.get_hist(energies, dx = bin_width, range = (lower_bound,upper_bound))  
    mu = bins[np.argmax(hist)]
    adc_to_kev = mu/peak
    # Making the window slightly smaller removes effects where as mu moves edge can be outside bin width
    lower_bound = mu - (kev_width[0] -2)* adc_to_kev 
    upper_bound = mu + (kev_width[1] -2)* adc_to_kev
    hist, bins, var = pgh.get_hist(energies, dx = bin_width, range = (lower_bound,upper_bound))
    #par_guesses = pgc.get_hpge_E_peak_par_guess(hist, bins, var, func_i)
    #if par_guesses == []: 
    par_guesses = simple_guess(hist, bins, var, func_i)
    try: 
        bounds = pgc.get_hpge_E_peak_bounds(hist, bins, var, func_i, par_guesses)
        pars_i, cov_i = pgf.fit_hist(func_i, hist, bins, var=var, guess=par_guesses, bounds=bounds)

    except: pars_i, cov_i = None, None
    return hist, bins, pars_i, cov_i

def get_peak_fwhm_with_dt_corr(Energies, alpha,dt, func, peak, kev_width, kev=False):
    correction = np.multiply(np.multiply(alpha,dt),Energies, dtype='float64')
    ct_energy = np.add(correction, Energies)
    
    if func == radford_peak:
        try:   
            hist, bins, params, covs = fit_peak_func(ct_energy, func_i= func, peak=peak, kev_width=kev_width)
            fwhm, fwhm_err = pgf.get_fwhm_func(func, params,covs)
            if kev==True:
                fwhm = fwhm * (peak/params[0])
                fwhm_err = fwhm_err * (peak/params[0])
        
            try:
                hist2, bins2, params2, covs2 = fit_peak_func(ct_energy, func_i= pgf.gauss_step, peak=peak, kev_width=kev_width)
                fwhm2, fwhm2_err = pgf.get_fwhm_func(pgf.gauss_step, params2, covs2)
                if kev==True:
                    fwhm2 = fwhm2 * (peak/params2[1])
                    fwhm2_err = fwhm2_err * (peak/params2[1])
                if np.isnan(fwhm) and np.isnan(fwhm2):
                    return np.nan, np.nan
                elif np.isnan(fwhm) and not np.isnan(fwhm2):
                    fwhm = fwhm2
                    fwhm_err = fwhm2_err
                    fit = gauss_step(bins2, *params2)
                elif np.isnan(fwhm2) and not np.isnan(fwhm):
                    fit = radford_peak(bins, *params)
                else: 
                    bin_centres = (bins[:-1]+bins[1:])/2
                    fit_rp = radford_peak(bin_centres, *params)
                    fit_gs = gauss_step(bin_centres, *params2)
                    cs_rp = chisquare(hist, f_exp=fit_rp, ddof=7)[0] 
                    cs_gs = chisquare(hist, f_exp=fit_gs, ddof=5)[0]
                    if cs_rp < cs_gs: 
                        fit = radford_peak(bins, *params)
                    else: 
                        fwhm = fwhm2
                        fwhm_err = fwhm2_err
                        fit = gauss_step(bins2, *params2)
            except: 
                fit = radford_peak(bins, *params)
            
        except: 
            return np.nan, np.nan, np.nan
    elif func == gauss_step:
        try:
            hist, bins, params, covs = fit_peak_func(ct_energy, func_i= func, peak=peak, kev_width=kev_width)
            fwhm, fwhm_err = pgf.get_fwhm_func(func, params,covs)
            if kev==True:
                fwhm = fwhm * (peak/params[1])
                fwhm_err = fwhm_err * (peak/params[1])
            fit = gauss_step(bins, *params)
            
        except:
            return np.nan, np.nan, np.nan
    max_val = np.nanmax(fit)
    if not np.isnan(fwhm/max_val) and not max_val == 0:
        return fwhm, max_val, fwhm_err
    else:
        return np.nan, np.nan, np.nan


def fom_FWHM_with_dt_corr_fit(tb_in,verbosity, kwarg_dict, ctc_parameter):
    parameter = kwarg_dict['parameter']
    func = kwarg_dict['func']
    Energies=tb_in[parameter].nda
    Energies = Energies.astype('float64')
    peak = kwarg_dict['peak']
    kev_width = kwarg_dict['kev_width']
    max_alpha = 3*10**-6
    astep= 0.625*10**-7
    if ctc_parameter == 'QDrift':
        dt = tb_in['dt_eff'].nda 
    elif ctc_parameter == 'dt':
        #ends = tb_in['tp_99'].nda
        #nan_idxs = np.where(np.isnan(ends))[0]
        #if len(nan_idxs) < 0.01*len(Energies):
        #    maxs = tb_in['wf_max'].nda
        #    ends[nan_idxs] = maxs[nan_idxs]
        dt = np.subtract(tb_in['tp_99'].nda , tb_in['tp_0_est'].nda, dtype='float64')
    elif ctc_parameter == 'rt':
        dt = np.subtract(tb_in['tp_99'].nda,tb_in['tp_01'].nda, dtype='float64')
    if np.isnan(Energies).any(): return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan}
    if np.isnan(dt).any(): 
        print(np.where(np.isnan(dt))[0])
        return {'fwhm':np.nan,'fwhm_err':np.nan, 'alpha':np.nan}
    min_alpha=0
    alphas = np.arange(min_alpha,max_alpha+astep, astep,  dtype='float64')
    fwhms = np.array([])
    final_alphas = np.array([])
    for alpha in alphas:
        fwhm, max_val,_ = get_peak_fwhm_with_dt_corr(Energies, alpha,dt, func=func,
                                                    peak=peak, kev_width=kev_width)
        if not np.isnan(fwhm/max_val):
            fwhms = np.append(fwhms,fwhm/max_val)
            final_alphas = np.append(final_alphas, alpha)  
    
    # Make sure fit isn't based on only a few points
    if len(fwhms)< 10:
        return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan} 

    # This block is to remove anomalously high values from floating point errors
    if fwhms[-1]>fwhms[0]:
        ids = np.where(fwhms>fwhms[-1])
    else:
        ids = np.where(fwhms>fwhms[0])
    fwhms = np.delete(fwhms,ids)
    final_alphas = np.delete(final_alphas,ids)
    
    # Fit alpha curve to get best alpha
    alphas = np.arange(0,max_alpha+astep,astep/10)
    fit = np.polynomial.polynomial.polyfit(final_alphas, fwhms, 4)
    fit_vals = np.polynomial.polynomial.polyval(alphas, fit)
    if np.isnan(fit_vals).all():
        return {'fwhm':np.nan, 'fwhm_err':np.nan, 'alpha':np.nan}
    else:
        # Return fwhm of optimal alpha in kev with error
        final_fwhm, final_max, final_err = get_peak_fwhm_with_dt_corr(Energies, alphas[np.nanargmin(fit_vals)],dt, 
                                                  func, peak=peak, kev_width=kev_width, kev=True)
        return {'fwhm': final_fwhm,
                'fwhm_err': final_err,
                'alpha':alphas[np.nanargmin(fit_vals)],
                } 

def fom_all_fit(tb_in,verbosity, kwarg_dict):
    ctc_parameters = ['dt', 'rt', 'QDrift']
    output_dict = {}
    for param in ctc_parameters:
        out = fom_FWHM_with_dt_corr_fit(tb_in,verbosity, kwarg_dict, param)
        output_dict[param] = out
    return output_dict

def fom_FWHM_fit(tb_in,verbosity, kwarg_dict):
    parameter = kwarg_dict['parameter']
    func = kwarg_dict['func']
    Energies=tb_in[parameter].nda
    Energies = Energies.astype('float64')
    peak = kwarg_dict['peak']
    kev_width = kwarg_dict['kev_width']
    if np.isnan(Energies).any(): return {'fwhm':np.nan, 'fwhm_err':np.nan}

    final_fwhm, final_max, final_err = get_peak_fwhm_with_dt_corr(Energies, 0,0, 
                                                  func, peak=peak, kev_width=kev_width, kev=True)
    return {'fwhm': final_fwhm,
            'fwhm_err': final_err} 

def get_peak_indices(dsp_file, peak_val, verbose=True):
    energy = lh5.load_nda(dsp_file, ['trapEmax'], 'raw')['trapEmax']
    if verbose:print('Data Loaded')

    peaks_keV = np.array([238.632,   583.191, 727.330, 860.564, 1620.5, 2614.553])
    guess_keV = 1/18
    Euc_min = peaks_keV[0]/guess_keV * 0.6
    Euc_max = peaks_keV[-1]/guess_keV * 1.1
    dEuc = 1/guess_keV

    hist, bins, var = pgh.get_hist(energy, range=(Euc_min, Euc_max), dx=dEuc)
    detected_peaks_locs, detected_peaks_keV, roughpars = pgc.hpge_find_E_peaks(hist, bins, var, peaks_keV)
    kev_widths = [(10,10), (25,40), (25,40),(25,40),(25,40), (50,50)]
    funcs = [pgf.gauss_step, pgf.radford_peak, pgf.radford_peak,pgf.radford_peak,pgf.radford_peak, pgf.radford_peak]
    peak_idx = np.where(peaks_keV == peak_val)[0][0]
    peak_loc = detected_peaks_locs[peak_idx]
    kev_width = kev_widths[peak_idx]
    rough_adc_to_kev = roughpars[0]
    func= funcs[peak_idx]
    
    e_lower_lim = peak_loc - (kev_width[0])/rough_adc_to_kev
    e_upper_lim = peak_loc + (kev_width[1])/rough_adc_to_kev
    if verbose:print(e_lower_lim, e_upper_lim)
    e_mask = (energy>e_lower_lim)&(energy<e_upper_lim)
    e_idxs = np.where(e_mask)[0]
    if verbose:print(f'Got cut events in {peak_val}') 

    parameters = {'bl_mean':4,'bl_std':4, 'pz_std':4}
    par_data = lh5.load_nda(dsp_file, parameters.keys(), 'raw')

    for key in parameters.keys():
        par_data[key] = par_data[key][e_idxs]

    cut_dict = cts.generate_cuts(slice_dict(par_data, 40000), parameters)
    if verbose:print('Loaded Cuts')
    ct_mask = cts.get_cut_indexes(par_data, cut_dict, 'raw')
    wf_idxs = e_idxs[ct_mask][:10000]
    if verbose:print(f'Got events in {peak_val}')
    return wf_idxs, func, kev_width

def slice_dict(in_dict, n):
    out_dict = {}
    for par in in_dict:
        out_dict[par]=in_dict[par][:n]
    return out_dict

def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x +(m2*(x**2)))

def load_grids(det, filter_name):
    data_path = os.path.join(initial_data_path, det, filter_name)
    peak_files = os.listdir(data_path)
    peak_files.sort(key =key)
    peak_energies = np.array([peak.split('.p')[0] for peak in peak_files], dtype='float32')
    print(peak_energies)
    peak_grids = []
    for peak in peak_files:
        full_data_path = os.path.join(data_path, peak)
        d = open(full_data_path,"rb")
        grid = pkl.load(d)
        peak_grids.append(grid)
    return peak_grids, peak_energies

def load_config(filter_name):
    if filter_name =='Cusp':
        opt_config = '/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v02/src/python/pygama/pygama/genpar_tmp/cusp_config.json'
    elif filter_name =='Zac':
        opt_config = '/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v02/src/python/pygama/pygama/genpar_tmp/zac_config.json'
    elif filter_name =='Trap':
        opt_config = '/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v02/src/python/pygama/pygama/genpar_tmp/trap_config.json'
    with open(opt_config, 'r') as o:
        opt_dict = json.load(o)
    opt_name = filter_name.lower()
    if opt_name =='trap':
        opt_name = 'etrap'
    keys = list(opt_dict[opt_name].keys())
    x_dict = opt_dict[opt_name][keys[1]]
    xs=(np.arange(0,x_dict['frequency'],1),np.linspace(x_dict['start'],x_dict['end'],x_dict['frequency']))
    y_dict = opt_dict[opt_name][keys[0]]
    ys=(np.arange(0,y_dict['frequency'],1),np.linspace(y_dict['start'],y_dict['end'],y_dict['frequency']))
    for i,x in enumerate(xs[1]):
        xs[1][i] = round(x,1)
    for i,y in enumerate(ys[1]):
        ys[1][i] = round(y,1)
    return xs, ys, keys, opt_dict

def get_ctc_grid(grids, ctc_param):
    error_grids = []
    dt_grids = []
    alpha_grids=[]
    for grid in peak_grids:
        shape=grid.shape
        dt_grid = np.ndarray(shape=shape)
        alpha_grid = np.ndarray(shape=shape)
        error_grid = np.ndarray(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dt_grid[i,j]= grid[i,j][ctc_param]['fwhm']
                alpha_grid[i,j]= grid[i,j][ctc_param]['alpha']
                error_grid[i,j]= grid[i,j][ctc_param]['fwhm_err']
        dt_grids.append(dt_grid)
        alpha_grids.append(alpha_grid)
        error_grids.append(error_grid)
    return dt_grids, error_grids , alpha_grids

def interpolate_energy(peak_energies, grids, error_grids, energy):
    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    out_grid_err = np.empty(grid_shape)
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        err_points = np.array([error_grids[i][index] for i in range(len(grids))])
        nan_mask = np.isnan(points)
        nan_mask = nan_mask | (points<0)
        try:
            if len(points[~nan_mask])<5:
                return np.nan, np.nan
            param_guess  = [0.2,0.001,0.000001]
            param_bounds = (0, [10., 1. ,0.1])
            fit_pars, fit_covs = curve_fit(fwhm_slope, peak_energies,points, sigma=err_points, 
                               p0=param_guess, bounds=param_bounds, absolute_sigma=True)
            fit_qbb = fwhm_slope(energy,*fit_pars)
            sderrs = np.sqrt(np.diag(fit_covs))
            qbb_err = fwhm_slope(energy,*(fit_pars+sderrs))-fwhm_slope(energy,*fit_pars)
            out_grid[index] = fit_qbb
            out_grid_err[index] = qbb_err
        except:
            out_grid[index] = np.nan
            out_grid_err[index] = np.nan
    return out_grid, out_grid_err

def find_lowest_grid_point(grid, err_grid, opt_dict, filter_name):
    opt_name = filter_name.lower()
    if opt_name =='trap':
        opt_name = 'etrap'
    keys = list(opt_dict[opt_name].keys())
    x_dict = opt_dict[opt_name][keys[0]]
    xs=np.linspace(x_dict['start'],x_dict['end'],x_dict['frequency'])
    y_dict = opt_dict[opt_name][keys[1]]
    ys=np.linspace(y_dict['start'],y_dict['end'],y_dict['frequency'])
    total_lengths = np.zeros((len(xs), len(ys)))
    
    for i in range(len(xs)):
        for j in range(len(ys)):
            total_lengths[i][j]= xs[i]+ys[j]
    min_val = np.nanmin(grid)
    lowest_ixs = np.where(grid == min_val)
    print('Minimum value is :', min_val, '+-', err_grid[lowest_ixs][0])
    print(lowest_ixs)
    if isinstance(lowest_ixs[0], np.int64):
        print(keys[1],':',ys[lowest_ixs[1]], 'us',  keys[0],' : ',xs[lowest_ixs[0]],'us')
    else:
        shortest_length = np.argmin(total_lengths[lowest_ixs])
        final_idxs = (lowest_ixs[0][shortest_length],lowest_ixs[1][shortest_length])
        print(keys[1],':',ys[final_idxs[1]], 'us',  keys[0],' : ',xs[final_idxs[0]],'us')
    return lowest_ixs, min_val, err_grid[lowest_ixs][0]

def interpolate_grid(energies, grids, int_energy, deg):
    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        nan_mask = np.isnan(points)
        nan_mask = nan_mask | (points<0)
        try:
            if len(points[~nan_mask])<5:
                raise IndexError
            fit_point = np.polynomial.polynomial.polyfit(energies[~nan_mask],points[~nan_mask], deg=deg)
            out_grid[index] = np.polynomial.polynomial.polyval(int_energy, fit_point)
        except:
            out_grid[index] = np.nan
    return out_grid

