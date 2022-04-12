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


def energy_cal_th(files, energy_params,  save_path=None, plot_path=None, cut_parameters={'bl_mean':4,'bl_std':4, 'pz_std':4} ,lh5_path='raw',n_events=15000):


    """
    This is an example script for calibrating Th data.
    """
    
    if isinstance(energy_params, str): energy_params = [energy_params]

    mpl.use('pdf')
    plt.rcParams['figure.figsize'] = (12, 20)
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
    
    glines    = [583.191, 727.330, 860.564,1592.53,1620.50,2103.53,2614.50] # gamma lines used for calibration
    range_keV = [(20,20),(30,30), (30,30),(40,25),(25,40),(40,40),(60,60)] # side bands width
    funcs = [pgf.extended_radford_pdf,pgf.extended_radford_pdf,pgf.extended_radford_pdf,pgf.extended_radford_pdf,
         pgf.extended_radford_pdf,pgf.extended_radford_pdf,pgf.extended_radford_pdf]
    gof_funcs = [pgf.radford_pdf,pgf.radford_pdf,pgf.radford_pdf,pgf.radford_pdf,
            pgf.radford_pdf,pgf.radford_pdf,pgf.radford_pdf]
    output_dict = {}
    for energy_param in energy_params:
        datatype, detector, measurement, run, timestamp = os.path.basename(files[0]).split('-')
        
        kev_ranges = range_keV.copy()
        guess_keV  = (2620/np.nanpercentile(uncal_pass[energy_param],99))
        print(f'Find peaks and compute calibration curve for {energy_param}', end = ' ')
        pars, cov, results = cal.hpge_E_calibration(uncal_pass[energy_param],
                                                    glines,
                                                    guess_keV,
                                                    deg=1,
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
        
        for i, peak in enumerate(glines):
            if peak not in fitted_peaks: 
                kev_ranges[i] = (kev_ranges[i][0]-5,  kev_ranges[i][1]-5)
        for i, peak in enumerate(glines):
            if peak not in fitted_peaks: 
                kev_ranges[i] = (kev_ranges[i][0]-5,  kev_ranges[i][1]-5)
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
                                                    deg=1,
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
        xpb = 1
        xlo = 0
        xhi = 4000
        nb = int((xhi-xlo)/xpb)
        hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
        bins_pass = pgh.get_bin_centers(bin_edges)
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


         #####
        # Remove the Tl SEP and DEP from calibration if found
        fwhm_peaks   = np.array([], dtype=np.float32)
        indexes=[]
        for i,peak in enumerate(fitted_peaks):
            if(peak==2103.53): 
                print(f"Tl SEP found at index {i}")
                indexes.append(i)
                continue
            elif(peak==1592.53): 
                print(f"Tl DEP found at index {i}")
                indexes.append(i)
                continue
            else:
                fwhm_peaks = np.append(fwhm_peaks,peak)
        fit_fwhms  = np.delete(fwhms,[indexes])
        fit_dfwhms = np.delete(dfwhms,[indexes])
        fit_mus = np.delete(mus,[indexes])
        #####
        
        for i,peak in enumerate(fwhm_peaks):
            print(f'FWHM of {peak} keV peak is: {fit_fwhms[i]:1.2f} +- {fit_dfwhms[i]:1.2f} keV')
        param_guess  = [0.2,0.001]
        param_bounds = (0, [10., 1.])
        fit_pars, fit_covs = curve_fit(fwhm_slope, fwhm_peaks, fit_fwhms, sigma=fit_dfwhms, 
                            p0=param_guess, bounds=param_bounds, absolute_sigma=True)
        
        rng = np.random.default_rng(1)
        pars_b = rng.multivariate_normal(fit_pars, fit_covs, size=1000)
        fits = np.array([fwhm_slope(fwhm_peaks, *pars) for pars in pars_b])
        qbb_vals = np.array([fwhm_slope(2039.0, *pars) for pars in pars_b])
        qbb_err = np.nanstd(qbb_vals)

        print(f'FWHM curve fit: {fit_pars}')
        predicted_fwhms = fwhm_slope(fwhm_peaks,*fit_pars)
        print(f'FWHM fit values: {predicted_fwhms}')
        fit_qbb = fwhm_slope(2039.0,*fit_pars)
        print(f'FWHM energy resolution at Qbb: {fit_qbb:1.2f} +- {qbb_err:1.2f} keV' )

        if plot_path is not None:
            plot_save_path = os.path.join(plot_path, f'{energy_param}.pdf')
            pathlib.Path(os.path.dirname(plot_save_path)).mkdir(parents=True, exist_ok=True)


            with PdfPages(plot_save_path) as pdf:

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
                
                qbb_line_hx = [2039.0,2039.0]
                qbb_line_hy = [np.amin(fit_fwhms),fit_qbb]
                qbb_line_vx = [np.amin(fwhm_peaks),2039.0]
                qbb_line_vy = [fit_qbb,fit_qbb]

                fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
                ax1.errorbar(fwhm_peaks,fit_fwhms,yerr=fit_dfwhms, marker='x',lw=0, c='b')
                ax1.plot(fwhm_peaks,predicted_fwhms,ls=' ')
                fwhm_slope_bins = np.arange(np.amin(fwhm_peaks),np.amax(fwhm_peaks),10)
                ax1.plot(fwhm_slope_bins ,fwhm_slope(fwhm_slope_bins,*fit_pars),lw=1, c='g')
                ax1.plot(qbb_line_hx,qbb_line_hy,lw=1, c='r')
                ax1.plot(qbb_line_vx,qbb_line_vy,lw=1, c='r')
                ax1.plot(np.nan, np.nan, '-', color='none', label=f'Qbb fwhm: {fit_qbb:1.2f} +- {qbb_err:1.2f} keV')
                ax1.legend(loc='upper left', frameon=False )
                ax1.set_ylim([1,3])
                ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)
                ax2.plot(fwhm_peaks,pgf.poly(fit_mus, pars)-fwhm_peaks, lw=1, c='b')
                ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
                ax2.set_ylabel("Residuals (keV)", ha='right', y=1)
                fig.suptitle(plot_title)
                pdf.savefig()
                plt.close()

                plt.figure()
                bins = np.linspace(0,3000,6000)
                plt.hist(ecal_pass, bins=bins, histtype='step', label=f'{len(ecal_pass)} events passed quality cuts')
                plt.hist(ecal_fail, bins=bins, histtype='step', label=f'{len(ecal_fail)} events failed quality cuts')
                plt.yscale("log")
                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                plt.legend(loc= 'upper right')
                pdf.savefig()
                plt.close()

                plt.figure()
                n_bins = 500
                counts_pass, bins_pass, _ = pgh.get_hist(ecal_pass, bins =n_bins, range=(0,3000))
                counts_fail, bins_fail, _ = pgh.get_hist(ecal_fail, bins =n_bins, range=(0,3000))
                counts_bl, bins_bl, _ = pgh.get_hist(bl_pass, bins =n_bins, range=(0,3000))
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

        
        if fitted_peaks[-1] == 2614.50:
            fep_fwhm = round(fwhms[-1],2)
            fep_dwhm = round(dfwhms[-1],2)
        else:
            fep_fwhm = np.nan
            fep_dwhm = np.nan
        output_dict[energy_param] = {'Qbb_fwhm': round(fit_qbb,2), 'Qbb_fwhm_err': round(qbb_err,2),
                                    '2.6_fwhm': fep_fwhm, '2.6_fwhm_err': fep_dwhm, 
                                    "m0":fit_pars[0], "m1":fit_pars[1], 
                                    "Calibration_pars":pars.tolist(),
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
    if peak == 583.191: 
        return 'Tl 583'
    elif peak == 727.33:
        return 'Bi 727'
    elif peak == 860.564:
        return 'Tl 860'
    elif peak == 1592.53:
        return 'Tl DEP'
    elif peak == 1620.5:
        return 'Bi FEP'
    elif peak == 2103.53:
        return 'Tl SEP'
    elif peak == 2614.5:
        return 'Tl FEP'