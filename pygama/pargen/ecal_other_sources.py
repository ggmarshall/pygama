import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np
import os,json
import pathlib
from scipy.optimize import curve_fit
import pygama.pargen.cuts as cut
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgf
import scipy.stats
import math
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

def fwhm_slope(x, m0, m1):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x) 

def am_calibration(energy, verbose=False):
    results={}
    guess_keV  = (81/np.nanpercentile(energy,95))
    range_keV = (10,10) 
    Euc_min = 81/guess_keV * 0.6
    Euc_max = 81/guess_keV * 1.1
    dEuc = 1/guess_keV
    hist, bins, var = pgh.get_hist(energy, range=(Euc_min, Euc_max), dx=dEuc)
    if np.any(var == 0):
        if verbose:
            print(f'hpge_find_E_peaks: replacing var zeros with {var_zero}')
        var[np.where(var == 0)] = var_zero
        peaks_keV = np.asarray(peaks_keV)

    # Find all maxes with > n_sigma significance
    imaxes = cal.get_i_local_maxima(hist/np.sqrt(var), 10)
    if len(imaxes)>1:
        imaxes = [imaxes[-1]]

    # Now pattern match to peaks_keV within Etol_keV using poly_match
    detected_max_locs = pgh.get_bin_centers(bins)[imaxes]
    roughpars = 81/detected_max_locs[0]

    results['got_peaks_locs'] = detected_max_locs
    results['got_peaks_keV'] = [81]

    Euc_min, Euc_max = [(np.poly1d([roughpars,0])-i).roots for i in (81*.9, 81*1.1)]
    dEuc = 0.2/roughpars
    hist, bins, var = pgh.get_hist(energy, range=(Euc_min[0], Euc_max[0]), dx=dEuc)
    derco = np.polyder(np.poly1d([roughpars,0])).coefficients
    der = pgf.poly(81, derco)
    range_uncal = [(range_keV[0]/der, range_keV[1]/der)]
    n_bins = [sum(range_keV)/0.5/der]
    pk_pars,pk_errors, pk_covs, pk_binws, pk_ranges, pk_pvals = cal.hpge_fit_E_peaks(energy, 
                                                                detected_max_locs, range_uncal, 
                                                                n_bins=n_bins,
                                                                funcs=[pgf.extended_gauss_step_pdf], 
                                                                method="unbinned", 
                                                                gof_funcs = [pgf.gauss_step_pdf], 
                                                                allowed_p_val=0,
                                                                n_events=15000, simplex=True)
    results['pk_pars'] = pk_pars
    results['pk_errors'] = pk_errors
    results['pk_covs'] = pk_covs
    results['pk_binws'] = pk_binws
    results['pk_ranges'] = pk_ranges
    results['pk_pvals'] = pk_pvals

    fitidx = [i is not None for i in pk_pars]
    fitted_peaks_keV = results['fitted_keV'] = detected_max_locs[fitidx]
    
    mu,mu_err = pgf.get_mu_func(pgf.extended_gauss_step_pdf, pk_pars[0], errors=pk_errors[0]) 
    
    pars, cov = cal.hpge_fit_E_scale([mu], [mu_err], [81], deg=0)
    results['pk_cal_pars'] = pars
    results['pk_cal_cov'] = cov
    pars, cov = cal.hpge_fit_E_cal_func([mu], [mu_err], [81], pars, deg=0)
    
    
    uncal_fwhms,uncal_fwhm_errs = pgf.get_fwhm_func(pgf.extended_gauss_step_pdf, pk_pars[0], pk_covs[0])
    cal_fwhms = uncal_fwhms * der 
    cal_fwhms_errs = uncal_fwhm_errs*der
    results['pk_fwhms'] = np.asarray([(cal_fwhms, cal_fwhms_errs)])
    return pars, cov, results

def energy_cal_am(files, energy_params, save_path=None, 
                    plot_path=None, cut_parameters={'bl_mean':4,'bl_std':4, 'pz_std':4} ,lh5_path='raw',n_events=15000):

    if isinstance(energy_params, str): energy_params = [energy_params]

    mpl.use('pdf')
    plt.rcParams['figure.figsize'] = (20, 12)
    plt.rcParams['font.size'] = 12

    ####################
    # Start the analysis
    ####################
    print('Load and apply quality cuts...',end=' ')
    uncal_pass_bl, uncal_cut_bl = cut.load_nda_with_cuts(files,energy_params,'raw',  cut_parameters= {"bl_mean":4,"bl_std":4}, verbose=False)
    uncal_pass, uncal_cut = cut.load_nda_with_cuts(files,energy_params,'raw',  cut_parameters= cut_parameters, verbose=False)
    print("Done")
    

    Npass = len(uncal_pass[energy_params[0]])
    Ncut  = len(uncal_cut[energy_params[0]])
    Ratio = 100.*float(Ncut)/float(Npass+Ncut)
    print(f'{Npass} events pass')
    print(f'{Ncut} events cut')
    
    output_dict = {}
    for energy_param in energy_params:
        datatype, detector, measurement, run, timestamp = os.path.basename(files[0]).split('-')
        pars, cov, results = am_calibration(uncal_pass[energy_param], verbose=False)
        print("done")
        print(" ")
        if pars is None:
            print("Calibration failed")
            continue
        fitted_peaks = results['fitted_keV']
        fitted_funcs = []
        fitted_gof_funcs = []
        for peak in fitted_peaks: 
            fitted_funcs.append(pgf.extended_gauss_step_pdf)
            fitted_gof_funcs.append(pgf.gauss_step_pdf)
                    
        ecal_pass = pgf.poly(uncal_pass[energy_param], pars)
        ecal_fail = pgf.poly(uncal_cut[energy_param], pars)
        bl_pass = pgf.poly(uncal_pass_bl[energy_param], pars)
        bl_fail = pgf.poly(uncal_cut_bl[energy_param], pars)
        
        fitted_peaks = results['fitted_keV']
        pk_pars      = results['pk_pars']
        pk_covs      = results['pk_covs']
        plot_title = f'{detector}-{measurement}-{run}'
        peaks_kev = results['got_peaks_keV']
        pk_ranges = results['pk_ranges']
        p_vals = results['pk_pvals']

        fwhms        = results['pk_fwhms'][:,0]
        dfwhms       = results['pk_fwhms'][:,1]

        for i,peak in enumerate(fitted_peaks):
            print(f'FWHM of 81 keV peak is: {fwhms[i]:1.2f} +- {dfwhms[i]:1.2f} keV')
        if plot_path is not None:
            plot_save_path = os.path.join(plot_path, f'{energy_param}.pdf')
            pathlib.Path(os.path.dirname(plot_save_path)).mkdir(parents=True, exist_ok=True)

            with PdfPages(plot_save_path) as pdf:

                plt.figure()
                range_adu = 5/pars[0] #10keV window around peak in adu
                binning = np.arange(pk_ranges[0][0], pk_ranges[0][1], 0.1)
                bin_cs = (binning[1:]+binning[:-1])/2
                energies = uncal_pass[energy_param][(uncal_pass[energy_param]> pk_ranges[0][0])&
                                            (uncal_pass[energy_param]< pk_ranges[0][1])][:n_events]

                counts, bs, bars = plt.hist(energies, bins=binning, histtype='step')
                fit_vals = fitted_gof_funcs[i](bin_cs, *pk_pars[0])*np.diff(bs)
                plt.plot(bin_cs, fit_vals)
                plt.step(bin_cs, [(fval-count)/count if count != 0 else  (fval-count) for count, fval in zip(counts, fit_vals)] ) 
                plt.plot([bin_cs[10]],[0],label="Am 81", linestyle='None' )
                plt.plot([bin_cs[10]],[0],label = f'81 keV', linestyle='None')
                plt.plot([bin_cs[10]],[0],label = f'{fwhms[0]:.2f} +- {dfwhms[0]:.2f} keV', linestyle='None')
                plt.plot([bin_cs[10]],[0],label = f'p-value : {p_vals[i]:.2f}', linestyle='None')

                plt.xlabel('Energy (keV)')
                plt.ylabel('Counts')
                plt.legend(loc = 'upper right', frameon=False)
                locs,labels = plt.xticks()
                new_locs, new_labels = get_peak_labels(locs, pars)
                plt.xticks(ticks = new_locs, labels = new_labels)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                plt.rcParams['figure.figsize'] = (12, 8)
                plt.rcParams['font.size'] = 8

                plt.figure()
                bins = np.linspace(0,150,300)
                plt.hist(np.concatenate((ecal_pass,ecal_fail)), bins=bins, histtype='step', label=f'{len(ecal_pass)+len(ecal_fail)} total events')
                plt.hist(ecal_pass, bins=bins, histtype='step', label=f'{len(ecal_pass)} events passed quality cuts')
                plt.hist(ecal_fail, bins=bins, histtype='step', label=f'{len(ecal_fail)} events failed quality cuts')
                plt.yscale("log")
                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                plt.legend(loc= 'upper right')
                pdf.savefig()
                plt.close()

                plt.figure()
                n_bins = 150
                counts_pass, bins_pass, _ = pgh.get_hist(ecal_pass, bins =n_bins, range=(0,150))
                counts_fail, bins_fail, _ = pgh.get_hist(ecal_fail, bins =n_bins, range=(0,150))
                counts_bl, bins_bl, _ = pgh.get_hist(bl_pass, bins =n_bins, range=(0,150))
                sf = counts_pass/(counts_pass+counts_fail)
                bl_sf = counts_bl/(counts_pass+counts_fail)

                plt.step(pgh.get_bin_centers(bins_pass),sf, label='Total')
                plt.step(pgh.get_bin_centers(bins_pass),bl_sf, label='Baseline')
                plt.xlabel("Energy (keV)")
                plt.ylabel("Survival Fraction")
                plt.ylim([0,1])
                plt.legend()
                pdf.savefig()
                plt.close()


        output_dict[energy_param] = { "Calibration_pars":pars.tolist(),
                                    "Number_passed": Npass,'Number_cut': Ncut,"Cut Percentage": Ratio }

    if save_path is not None:
        with open(save_path,'w') as fp:
            json.dump(output_dict,fp, indent=4)
    else:
        print(output_dict)
        return output_dict



def energy_cal(files, energy_params, glines, range_keV, funcs, gof_funcs, deg=0,save_path=None, 
                plot_path=None, cut_parameters={'bl_mean':4,'bl_std':4, 'pz_std':4} ,lh5_path='raw',n_events=15000):


    """
    This is an example script for calibrating general data.
    """
    
    if isinstance(energy_params, str): energy_params = [energy_params]

    mpl.use('pdf')
    

    ####################
    # Start the analysis
    ####################
    print('Load and apply quality cuts...',end=' ')
    uncal_pass_bl, uncal_cut_bl = cut.load_nda_with_cuts(files,energy_params,'raw',  cut_parameters= {"bl_mean":4,"bl_std":4}, verbose=False)
    uncal_pass, uncal_cut = cut.load_nda_with_cuts(files,energy_params,'raw',  cut_parameters= cut_parameters, verbose=False)
    print("Done")

    Npass = len(uncal_pass[energy_params[0]])
    Ncut  = len(uncal_cut[energy_params[0]])
    Ratio = 100.*float(Ncut)/float(Npass+Ncut)
    print(f'{Npass} events pass')
    print(f'{Ncut} events cut')
    
    output_dict = {}
    for energy_param in energy_params:
        datatype, detector, measurement, run, timestamp = os.path.basename(files[0]).split('-')
        
        kev_ranges = range_keV.copy()
        guess_keV  = ((glines[-1]+20)/np.nanpercentile(uncal_pass[energy_param],99))
        print(f'Find peaks and compute calibration curve for {energy_param}', end = ' ')
        try:
            pars, cov, results = cal.hpge_E_calibration(uncal_pass[energy_param],
                                                        glines,
                                                        guess_keV,
                                                        deg=deg,
                                                        range_keV = range_keV,
                                                        funcs = funcs,
                                                        gof_funcs = gof_funcs,
                                                        n_events=n_events,
                                                        simplex=True,
                                                        verbose=False
                                                        )
            pk_pars      = results['pk_pars']
            found_peaks = results['got_peaks_locs']
            fitted_peaks = results['fitted_keV']
        except:
            fitted_peaks=[]
        
        for i, peak in enumerate(glines):
            if peak not in fitted_peaks: 
                kev_ranges[i] = (kev_ranges[i][0]-10,  kev_ranges[i][1]-10)
        for i, peak in enumerate(fitted_peaks):
            try:
                if results['pk_fwhms'][:,1][i]/results['pk_fwhms'][:,0][i] >0.05:
                    index = np.where(glines == peak)[0][0]
                    kev_ranges[i] = (kev_ranges[index][0]-5,  kev_ranges[index][1]-5)
            except:
                pass

        pars, cov, results = cal.hpge_E_calibration(uncal_pass[energy_param],
                                                    glines,
                                                    guess_keV,
                                                    deg=deg,
                                                    range_keV = kev_ranges,
                                                    funcs = funcs,
                                                    gof_funcs = gof_funcs,
                                                    n_events=n_events,
                                                    simplex=True,
                                                    verbose=False
                                                    )
        print("done")
        print(" ")
        if pars is None:
            print("Calibration failed")
            continue
        fitted_peaks = results['fitted_keV']
        fitted_funcs = []
        fitted_gof_funcs = []
        for i, peak in enumerate(glines):
            if peak in fitted_peaks: 
                fitted_funcs.append(funcs[i])
                fitted_gof_funcs.append(gof_funcs[i])

        
                    
        ecal_pass = pgf.poly(uncal_pass[energy_param], pars)
        ecal_fail = pgf.poly(uncal_cut[energy_param], pars)
        bl_pass = pgf.poly(uncal_pass_bl[energy_param], pars)
        bl_fail = pgf.poly(uncal_cut_bl[energy_param], pars)
        fitted_peaks = results['fitted_keV']
        pk_pars      = results['pk_pars']
        pk_covs      = results['pk_covs']
    
        plot_title = f'{detector}-{measurement}-{run}'
        peaks_kev = results['got_peaks_keV']
        
        pk_ranges = results['pk_ranges']
        p_vals = results['pk_pvals']
        mus = [pgf.get_mu_func(func_i, pars_i) for func_i, pars_i in zip(fitted_funcs, pk_pars)]

        fwhms        = results['pk_fwhms'][:,0]
        dfwhms       = results['pk_fwhms'][:,1]

        param_guess  = [0.2,0.001]
        param_bounds = (0, [10., 1.])
        fit_pars, fit_covs = curve_fit(fwhm_slope, fitted_peaks, fwhms, sigma=dfwhms, 
                            p0=param_guess, bounds=param_bounds, absolute_sigma=True)

        
        predicted_fwhms = fwhm_slope(fitted_peaks,*fit_pars)
        for i,peak in enumerate(fitted_peaks):
            print(f'FWHM of {peak} keV peak is: {fwhms[i]:1.2f} +- {dfwhms[i]:1.2f} keV')

        print(f'FWHM curve fit: {fit_pars}')

        if plot_path is not None:
            plot_save_path = os.path.join(plot_path, f'{energy_param}.pdf')
            pathlib.Path(os.path.dirname(plot_save_path)).mkdir(parents=True, exist_ok=True)


            with PdfPages(plot_save_path) as pdf:

                plt.rcParams['figure.figsize'] = (12, 12)
                plt.rcParams['font.size'] = 12

                plt.figure()
                range_adu = 5/pars[0] #10keV window around peak in adu
                for i, peak in enumerate(mus):
                    plt.subplot(math.ceil((len(mus))/2),2,i+1)
                    binning = np.arange(pk_ranges[i][0], pk_ranges[i][1], 1)
                    bin_cs = (binning[1:]+binning[:-1])/2
                    energies = uncal_pass[energy_param][(uncal_pass[energy_param]> pk_ranges[i][0])&
                                                (uncal_pass[energy_param]< pk_ranges[i][1])][:n_events]

                    counts, bs, bars = plt.hist(energies, bins=binning, histtype='step')
                    fit_vals = fitted_gof_funcs[i](bin_cs, *pk_pars[i])*np.diff(bs)
                    plt.plot(bin_cs, fit_vals)
                    plt.step(bin_cs, [(fval-count)/count if count != 0 else  (fval-count) for count, fval in zip(counts, fit_vals)] ) 
                    plt.plot([bin_cs[10]],[0],label=get_peak_label(fitted_peaks[i]), linestyle='None' )
                    plt.plot([bin_cs[10]],[0],label = f'{fitted_peaks[i]:.1f} keV', linestyle='None')
                    plt.plot([bin_cs[10]],[0],label = f'{fwhms[i]:.2f} +- {dfwhms[i]:.2f} keV', linestyle='None')
                    plt.plot([bin_cs[10]],[0],label = f'p-value : {p_vals[i]:.2f}', linestyle='None')

                    plt.xlabel('Energy (keV)')
                    plt.ylabel('Counts')
                    plt.legend(loc = 'upper left', frameon=False)
                    plt.xlim([peak-range_adu, peak+range_adu])
                    locs,labels = plt.xticks()
                    new_locs, new_labels = get_peak_labels(locs, pars)
                    plt.xticks(ticks = new_locs, labels = new_labels)

                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                
                plt.rcParams['figure.figsize'] = (12, 8)
                plt.rcParams['font.size'] = 8

                fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
                ax1.errorbar(fitted_peaks,fwhms,yerr=dfwhms, marker='x',lw=0, c='b')
                ax1.plot(fitted_peaks,predicted_fwhms,ls=' ')
                fwhm_slope_bins = np.linspace(np.amin(fitted_peaks),np.amax(fitted_peaks),100)
                ax1.plot(fwhm_slope_bins ,fwhm_slope(fwhm_slope_bins,*fit_pars),lw=1, c='g')
                ax1.legend(loc='upper left', frameon=False )
                ax1.set_ylim([0,2])
                ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)
                ax2.plot(fitted_peaks,pgf.poly(np.array(mus), pars)-fitted_peaks, lw=1, c='b')
                ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
                ax2.set_ylabel("Residuals (keV)", ha='right', y=1)
                fig.suptitle(plot_title)
                pdf.savefig()
                plt.close()

                plt.figure()
                bins = np.arange(0,np.nanmax(ecal_pass)+1,1)
                plt.hist(ecal_pass, bins=bins, histtype='step', label=f'{len(ecal_pass)} events passed quality cuts')
                plt.hist(ecal_fail, bins=bins, histtype='step', label=f'{len(ecal_fail)} events failed quality cuts')
                plt.yscale("log")
                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                plt.legend(loc= 'upper right')
                pdf.savefig()
                plt.close()

                plt.figure()
                n_bins = int(np.nanmax(ecal_pass)/6)
                counts_pass, bins_pass, _ = pgh.get_hist(ecal_pass, bins =n_bins, range=(0,np.nanpercentile(ecal_pass,99)))
                counts_fail, bins_fail, _ = pgh.get_hist(ecal_fail, bins =n_bins, range=(0,np.nanpercentile(ecal_pass,99)))
                counts_bl, bins_bl, _ = pgh.get_hist(bl_pass, bins =n_bins, range=(0,np.nanmax(ecal_pass)))
                sf = counts_pass/(counts_pass+counts_fail)
                bl_sf = counts_bl/(counts_pass+counts_fail)

                plt.step(pgh.get_bin_centers(bins_pass),sf, label='Total')
                plt.step(pgh.get_bin_centers(bins_pass),bl_sf, label='Baseline')
                plt.xlabel("Energy (keV)")
                plt.ylabel("Survival Fraction")
                plt.ylim([0,1])
                plt.legend(loc= 'upper right')
                pdf.savefig()
                plt.close()


        output_dict[energy_param] = { "Calibration_pars":pars.tolist(),
                                    "m0":fit_pars[0], "m1":fit_pars[1], 
                                    "Number_passed": Npass,'Number_cut': Ncut,"Cut Percentage": Ratio }

    if save_path is not None:
        with open(save_path,'w') as fp:
            json.dump(output_dict,fp, indent=4)
    else:
        print(output_dict)
        return output_dict

def get_peak_labels(labels, pars):
    out = []
    out_labels = []
    for i,label in enumerate(labels):
        if i%2 == 1:
            continue
        else:
            out.append( f'{pgf.poly(label, pars):.1f}')
            out_labels.append(label)
    return out_labels, out

def get_peak_label(peak):
    if peak == 1332: 
        return 'Co 1332'
    elif peak == 1173:
        return 'Co 1173'
    elif peak == 276.4:
        return "Ba 276"
    elif peak == 302.9:
        return "Ba 303"
    elif peak == 356:
        return "Ba 356"
    elif peak == 383.8:
        return "Ba 384"


def energy_cal_co(files, energy_params, save_path=None, plot_path=None, cut_parameters={'bl_mean':4,'bl_std':4, 'pz_std':4},lh5_path='raw',n_events=15000):
    glines    = [1173,1332] # gamma lines used for calibration
    range_keV = [(20,20), (30,30)] # side bands width
    funcs = [pgf.extended_radford_pdf,pgf.extended_radford_pdf]
    gof_funcs = [pgf.radford_pdf,pgf.radford_pdf]
    out_dict = energy_cal(files, energy_params, glines, range_keV, funcs, gof_funcs, deg=0,
                save_path=save_path, plot_path=plot_path, 
                cut_parameters=cut_parameters,lh5_path=lh5_path,n_events=n_events)
    if save_path is None:
        return out_dict
    else:
        return

def energy_cal_ba(files, energy_params, save_path=None, plot_path=None, cut_parameters={'bl_mean':4,'bl_std':4, 'pz_std':4},lh5_path='raw',n_events=15000):
    glines    = [276.4, 302.9, 356, 383.8]
    range_keV = [(20,20), (20,20), (20,20), (20,20)] # side bands width
    funcs = [pgf.extended_radford_pdf,pgf.extended_radford_pdf,pgf.extended_radford_pdf,pgf.extended_radford_pdf]
    gof_funcs = [pgf.radford_pdf,pgf.radford_pdf,pgf.radford_pdf,pgf.radford_pdf]
    out_dict = energy_cal(files, energy_params, glines, range_keV, funcs, gof_funcs, deg=0,
                save_path=save_path, plot_path=plot_path, 
                cut_parameters=cut_parameters,lh5_path=lh5_path,n_events=n_events)
    if save_path is None:
        return out_dict
    else:
        return