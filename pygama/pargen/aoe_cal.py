import pygama.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.colors import LogNorm
import pathlib
from iminuit import cost,Minuit, util
from scipy.special import erf, erfc
from scipy.stats import norm, poisson
from scipy.integrate import simps
from iminuit.util import propagate

import os,json
import math
from scipy.optimize import curve_fit
import pygama.pargen.cuts as cts
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as pgc
import pygama.analysis.peak_fitting as pgf
import scipy.optimize as opt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from scipy.integrate import quad
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from iminuit.util import propagate
import argparse
import pathlib
import numba as nb
from math import erfc
import sys
import matplotlib as mpl

kwd = {"parallel": False, "fastmath": True}
limit = np.log(sys.float_info.max)/10

def PDF_AoE(x, lambda_s, lambda_b, mu,sigma,tau, 
            lower_range=np.inf , upper_range=np.inf, components=False):
    
    if components == False:
        try:
            pdf = (lambda_b * pgf.gauss_tail_norm(x,mu,sigma,tau, lower_range , upper_range)+\
                     lambda_s *  pgf.gauss_norm(x, mu,sigma))
            return lambda_s+lambda_b, pdf
        except:
            return np.nan, np.full_like(x, np.nan)
    else:
        sig = lambda_s *  pgf.gauss_norm(x, mu,sigma)
        bkg = lambda_b * pgf.gauss_tail_norm(x,mu,sigma,tau, lower_range , upper_range)
        return lambda_s+lambda_b, sig,bkg 

@nb.njit(**kwd)
def exp_pdf(x,p1,p2):
    return np.exp(p1+x*p2)

def norm_exp_pdf(x,z,p1,p2):
    norm = 1/p2 *(exp_pdf(np.nanmax(x), p1,p2) - exp_pdf(np.nanmin(x), p1,p2))
    return z*exp_pdf(x,p1,p2)/norm

def unbinned_aoe_fit(aoe, display=0, verbose=False):
    aoe_len = len(aoe)
    hist, bins,var = pgh.get_hist(aoe,bins=500)
    bin_centers = (bins[:-1]+bins[1:])/2
    
    pars, cov = pgf.gauss_mode_max(hist, bins, var)
    mu = bin_centers[np.argmax(hist)]
    amp = np.amax(hist)
    _,sigma,_ = pgh.get_gaussian_guess(hist, bins)
    #m_guess = np.mean(hist[bin_centers<(mu-5*sigma)])#/len(bin_centers<(mu-5*sigma))
    ls_guess = 2*np.sum(hist[(bin_centers>mu)&(bin_centers<(mu+2.5*sigma))])#-np.sum(hist[(bin_centers>(mu-15*sigma))&(bin_centers<(mu-5*sigma))])
    #sigma=0.0035
    def gauss(x,z,mu,sigma):
        return z * norm.pdf(x,mu,sigma)
    c1_min= mu-2*sigma #0.495
    c1_max= mu+5*sigma #0.52
    c1 = cost.UnbinnedNLL(aoe[(aoe<c1_max)&(aoe>c1_min)], gauss) #+5*sigma
    m1 = Minuit(c1, ls_guess, mu,sigma)
    m1.limits = [(0, len(aoe[(aoe<c1_max)&(aoe>c1_min)])),(mu*0.8, mu*1.2),(0.8*sigma,sigma*1.2)]
    m1.migrad()
    ls_guess =m1.values[0]
    mu = m1.values[1]
    sigma = m1.values[2]
    #plt.figure()
    #xs = np.arange(0.2,1,.0001)

    #counts, bins, bars = plt.hist(aoe[(aoe<c1_max)&(aoe>c1_min)], bins=100, histtype='step')

    #dx = np.diff(bins)
    #plt.plot(xs , m1.values[0]*norm.pdf(xs,  m1.values[1], m1.values[2])*dx[0])
    #plt.show()
    
    #print(ls_guess,mu,sigma)
    fmin=  mu-15*sigma #0.45
    fmax = mu+5*sigma #0.52
    c2_max = mu-4*sigma
    #print(fmin,fmax)
    c2 = cost.UnbinnedNLL(aoe[(aoe<c2_max)&(aoe>fmin)], norm_exp_pdf)
    m2 = Minuit(c2, len(aoe[(aoe<c2_max)&(aoe>fmin)]), -20,10)
    m2.limits=[(0,len(aoe[(aoe<c2_max)&(aoe>fmin)])),(-100,0),(0,20)]
    m2.migrad()
    f = m2.values[2]
    #print(m2.values)
    #plt.figure()
    #xs = np.arange(fmin,c2_max,.01)

    #counts, bins, bars = plt.hist(aoe[(aoe<c2_max)&(aoe>fmin)], bins=100, histtype='step')

    #dx = np.diff(bins)
    #plt.plot(xs , norm_fexpo(xs,  *m2.values)*dx[0])
    #plt.show()
    #pars = [mu,sigma,amp,m_guess,0.01,mu-1.5*sigma,0,0.01]
    pars = [mu,sigma,amp,0,f] #[mu,sigma,amp,0,f,10]
    
    
    
    #fas_pars,_ = fit_aoe_spectrum(hist,bins,var) 
    #if verbose:print(fas_pars)
    bg_guess = len(aoe[(aoe<fmax)&(aoe>fmin)])-ls_guess
    #x0 = [ls_guess,bg_guess,pars[0],pars[1],pars[4],pars[5]]
    x0 = [ls_guess,bg_guess,pars[0],pars[1],pars[4], fmin, fmax,0]
    if verbose:print(x0)
        
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe<fmax)&(aoe>fmin)], PDF_AoE)

    m = Minuit(c, *x0)
    m.fixed[5:] = True
    #m.limits = bounds
    m.simplex().migrad()
    m.hesse()
    if verbose:print(m.values)
    if display>1:
        plt.figure()
        xs = np.linspace(fmin,fmax,1000)
        counts, bins, bars = plt.hist(aoe[(aoe<fmax)&(aoe>fmin)], bins=400, histtype='step')
        dx = np.diff(bins)
        plt.plot(xs, PDF_AoE(xs,*m.values)[1]* dx[0])
        #plt.yscale('log')
        n_events, sig, bkg = PDF_AoE(xs,*m.values[:-1], True)
        plt.plot(xs , sig* dx[0])
        plt.plot(xs , bkg*dx[0])
        plt.show()
        
        plt.figure()
        bin_centers= (bins[1:]+bins[:-1])/2
        res = (PDF_AoE(bin_centers,*m.values)[1]* dx[0]) - counts
        plt.plot(bin_centers, [re/count if count != 0 else re for re,count in zip(res,counts)])
        plt.show()
        return m.values, m.errors#, bin_centers, (PDF_AoE(bin_centers,*m.values)[1]* dx[0]) - counts
    else: return m.values, m.errors


def AoEcorrection(e,aoe,eres, pdf_path=None, display=0, plot_all=False):
    


    comptBands_width = 20;
    comptBands = np.array([940,960,980,1000,1020,1040,1130,1150,1170,1190,1210,1250,1270,1290,
                         1310,1330,1370,1390,1420,1520,1540,1650,1700,1780,1810,1850,1870,1890,1910,1930,1950,
                         1970,1990,2010,2030,2050,2150,2170,2190,2210,2230,2250,2270,2290])
                        #2310,2330,2350])
    comptBands = comptBands[::-1]
    peaks = np.array([1080,1094,1459,1512, 1552, 1592,1620, 1650, 1670,1830,2105]) 
    compt_aoe = np.zeros(len(comptBands))
    aoe_sigmas = np.zeros(len(comptBands))
    compt_aoe_err = np.zeros(len(comptBands))
    aoe_sigmas_err = np.zeros(len(comptBands))
    ratio = np.zeros(len(comptBands))
    ratio_err = np.zeros(len(comptBands))
    
    copper = cm = plt.get_cmap('copper') 
    cNorm  = mcolors.Normalize(vmin=0, vmax=len(comptBands))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=copper)
    if display>0 or pdf_path is not None:
        plt.figure()
    for i, band in enumerate(comptBands):
        aoe_tmp = aoe[(e>band) & (e<band+comptBands_width) & (aoe>0)][:10000]
        pars,errs = unbinned_aoe_fit(aoe_tmp, display=display)
        compt_aoe[i] = pars[2]
        aoe_sigmas[i] = pars[3]
        compt_aoe_err[i] = errs[2]
        aoe_sigmas_err[i] = errs[3]
        ratio[i] = pars[0]/pars[1]
        ratio_err[i] = ratio[i]*np.sqrt((errs[0]/pars[0])**2 + (errs[1]/pars[1])**2)
        xs = np.arange(pars[2]-4*pars[3], pars[2]+3*pars[3], pars[3]/10)
        if np.isnan(errs[2])|np.isnan(errs[3])|(errs[2]==0)|(errs[3]==0): pass
        else:
            if display>0 or pdf_path is not None:
                colorVal = scalarMap.to_rgba(i)
                plt.plot(xs,PDF_AoE(xs, *pars)[1], color = colorVal)
    
    if display>0 or pdf_path is not None:
        plt.xlabel('A/E')
        plt.ylabel("Expected Counts")
        plt.title("Compton Band Fits")
        cbar = plt.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('copper_r')), 
                            orientation='horizontal', label='Compton Band Energy', ticks=[0,16,32,len(comptBands)])#cax=ax,
        cbar.ax.set_xticklabels([comptBands[::-1][0],comptBands[::-1][16],comptBands[::-1][32],comptBands[::-1][-1]])
        if pdf_path is not None:
            pdf_path.savefig()
            plt.close()
        elif display>0:
            plt.show()
    ids = np.isnan(compt_aoe_err)|np.isnan(aoe_sigmas_err)|(aoe_sigmas_err==0)|(compt_aoe_err==0)
    #print(compt_aoe_err, aoe_sigmas_err)

    if display>0 or pdf_path is not None:
        plt.figure()
        plt.errorbar(comptBands[~ids], ratio[~ids], xerr=10,yerr = ratio_err[~ids], linestyle=' ')
        plt.xlabel("Energy (keV)")
        plt.ylabel("N_sig/N_bkg")
        if pdf_path is not None:
            pdf_path.savefig()
            plt.close()
        elif display>0:
            plt.show()

    def pol1(x,a,b):
        return a * x + b

    pars, cov = opt.curve_fit(pol1,comptBands[~ids],compt_aoe[~ids],sigma = compt_aoe_err[~ids], absolute_sigma=True)
    errs = np.sqrt(np.diag(cov))

    sig_pars, sig_cov = opt.curve_fit(pol1,comptBands[~ids],aoe_sigmas[~ids],sigma = aoe_sigmas_err[~ids], absolute_sigma=True)
    sig_errs = np.sqrt(np.diag(sig_cov))

    def sigma_fit(x, a,b,c):
        return np.sqrt(a+(b/x)**c)
    
    p0 = [0.001,10, 3]
    c = cost.LeastSquares(comptBands[~ids],aoe_sigmas[~ids], aoe_sigmas_err[~ids], sigma_fit)

    c.loss = "soft_l1"
    m = Minuit(c, *p0)
    m.migrad()
    m.hesse()

    sig_pars2 = m.values
    sig_errs2 = m.errors

    model = pol1(comptBands,*pars)
    sig_model = pol1(comptBands,*sig_pars)
    sig_model2 = sigma_fit(comptBands,*sig_pars2)
    
    sigma = np.sqrt(eres[0]+1592*eres[1])/2.355
    n_sigma = 4
    peak = 1592
    emin           = peak - n_sigma*sigma
    emax           = peak + n_sigma*sigma
    dep_pars, dep_err = unbinned_aoe_fit(aoe[(e>emin) & (e<emax) & (aoe>0)][:10000])

    if display>0 or pdf_path is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        ax1.errorbar(comptBands[~ids]+10,compt_aoe[~ids],
                        yerr=compt_aoe_err[~ids], 
                        xerr=10,
                        label='data', linestyle= ' ')
        ax1.plot(comptBands[~ids]+10,model[~ids],label='linear model')
        ax1.errorbar(1592, dep_pars[2], xerr = n_sigma*sigma, yerr = dep_err[2], 
                        label='DEP', color='green', linestyle= ' ')
            
        ax1.legend(title='A/E mu energy dependence', frameon=False)
            
        ax1.set_ylabel("raw A/E (a.u.)", ha='right', y=1)
        #ax1.ylim([np.amax(model), np.amin(model)])
        ax2.scatter(comptBands[~ids]+10,100*(compt_aoe[~ids]-model[~ids])/compt_aoe_err[~ids], lw=1, c='b')
        ax2.scatter(1592,100*(dep_pars[2]-pol1(1592,*pars))/dep_err[2], lw=1, c='g')
        ax2.set_ylabel("Residuals %", ha='right', y=1)
        ax2.set_xlabel("Energy (keV)", ha='right', x=1)
        #plt.savefig('./plots/aoe_energy_dependence.pdf', bbox_inches='tight', transparent=True)
        if pdf_path is not None:
            pdf_path.savefig()
            plt.close()
        elif display>0:
            plt.show()

    if display>0 or pdf_path is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        ax1.errorbar(comptBands[~ids]+10, aoe_sigmas[~ids],
                    yerr= aoe_sigmas_err[~ids],
                    xerr = 10, label='data', linestyle= ' ')
        #ax1.plot(comptBands[~ids],sig_model[~ids],label='linear model')
        ax1.plot(comptBands[~ids],sig_model2[~ids],label=f'sqrt model: sqrt({sig_pars2[0]:1.4f}+({sig_pars2[1]:1.1f}/E)^{sig_pars2[2]:1.1f})')
        ax1.errorbar(1592, dep_pars[3], xerr = n_sigma*sigma,yerr = dep_err[3], label='DEP', color='green')
        ax1.set_ylabel("A/E stdev (a.u.)", ha='right', y=1)
        ax1.legend(title='A/E stdev energy dependence', frameon=False)
        ax2.scatter(comptBands[~ids]+10,100*(aoe_sigmas[~ids]-sig_model2[~ids])/aoe_sigmas_err[~ids], lw=1, c='b')
        ax2.scatter(1592,100*(dep_pars[3]-sigma_fit(1592,*sig_pars2))/dep_err[3], lw=1, c='g')
        ax2.set_ylabel("Residuals", ha='right', y=1)
        ax2.set_xlabel("Energy (keV)", ha='right', x=1)
        if pdf_path is not None:
            pdf_path.savefig()
            plt.close()
        elif display>0:
            plt.show()

    return pars, sig_pars2

def plot_compt_bands_overlayed(aoe,energy,eranges, aoe_range=None):
    for erange in eranges:
        hist, bins,var = pgh.get_hist(aoe[(energy>erange-10) &(energy<erange+10)],bins=500)
        bin_cs = (bins[1:]+bins[:-1])/2
        mu = bin_cs[np.argmax(hist)]
        if aoe_range is None:
            aoe_range = [mu*0.97, mu*1.02]
        idxs = (energy>erange-10) &(energy<erange+10)&(aoe>aoe_range[0])&(aoe<aoe_range[1])
        plt.hist(aoe[idxs], bins=50, histtype='step', label=f'{erange-10}-{erange+10}')

def plot_dt_dep(aoe, energy,dt, erange,title):
    
    hist, bins,var = pgh.get_hist(aoe[(energy>erange[0]) &(energy<erange[1])],bins=500)
    bin_cs = (bins[1:]+bins[:-1])/2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu*0.9, mu*1.1]
    
    idxs = (energy>erange[0]) &(energy<erange[1])&(aoe>aoe_range[0])&(aoe<aoe_range[1])&(dt<2000)
    
    plt.hist2d(aoe[idxs], dt[idxs], bins=[200,100], norm=LogNorm())
    plt.ylabel('Drift Time (ns)')
    plt.xlabel('A/E')
    plt.title(title)

def load_aoe(files, cal_dict, energy_param, cal_energy_param, cut_parameters = {"bl_mean":4, "bl_std":4,"pz_std":4}):
    
    params = [energy_param, cal_energy_param, 'dt_eff','A_max', 'tp_0_est', 'tp_99']
    cal_pars = cal_dict[cal_energy_param]['Calibration_pars']
    eres_pars = [cal_dict[cal_energy_param]['m0'], cal_dict[cal_energy_param]['m1']]
    print(len(files),'files found')
    print('Load and apply quality cuts...',end=' ')
    
    uncal_pass, uncal_cut = cts.load_nda_with_cuts(files,params,'raw', cut_parameters= cut_parameters, verbose=False)
    print("done")
    
    Npass = len(uncal_pass[cal_energy_param])
    Ncut  = len(uncal_cut[cal_energy_param])
    Ratio = 100.*float(Ncut)/float(Npass+Ncut)
    print('  ',Npass,'events pass')
    print('  ',Ncut, 'events cut')
    print('   ratio: %2.1f %%' % Ratio)
    ecal_pass = pgf.poly(uncal_pass[cal_energy_param], cal_pars)
    curr = uncal_pass['A_max']
    aoe = np.divide(curr,pgf.poly(uncal_pass[energy_param], cal_pars))
    full_dt = uncal_pass['tp_99']-uncal_pass['tp_0_est']
    return aoe, ecal_pass, uncal_pass['dt_eff'], full_dt, eres_pars

def unbinned_energy_fit(energy, peak):
    energy_len = len(energy)
    hist, bins,var = pgh.get_hist(energy,dx=0.1, range= (np.amin(energy), np.amax(energy)))
    x0 = pgc.get_hpge_E_peak_par_guess(hist, bins, var, pgf.extended_radford_pdf)
    fixed,mask = pgc.get_hpge_E_fixed(pgf.extended_radford_pdf)
    bounds = pgc.get_hpge_E_bounds(pgf.extended_radford_pdf)

    pars, errs, cov = pgf.fit_unbinned(pgf.extended_radford_pdf, energy, guess=x0,
             Extended=True, cost_func = 'LL',simplex=True, fixed=fixed, bounds=bounds)

    return pars,errs

def get_peak_label(peak):
    if peak == 2039: 
        return 'CC @'
    elif peak == 1592.5:
        return 'Tl DEP @'
    elif peak == 1620.5:
        return 'Bi FEP @'
    elif peak == 2103.53:
        return 'Tl SEP @'
    elif peak == 2614.5:
        return 'Tl FEP @'

def get_aoe_cut_fit(energy,aoe,peak,ranges,dep_acc, display=1):
    min_range, max_range = ranges
    
    peak_energy = energy[(energy>peak-min_range)&(energy<peak+max_range)][:20000]
    peak_aoe = aoe[(energy>peak-min_range)&(energy<peak+max_range)][:20000]
    cut_vals = np.arange(-5,-0.5, 0.1)
    pars, errors = unbinned_energy_fit(peak_energy, peak)
    pc_n = pars[0]
    pc_err = errors[0]
    sfs = []
    sf_errs=[]
    for cut_val in cut_vals:
        idxs = peak_aoe>cut_val
        cut_pars,ct_errs = unbinned_energy_fit(peak_energy[idxs],  peak)
        ct_n = cut_pars[0]
        ct_err = ct_errs[0]
        sf = (ct_n/pc_n)*100
        sfs.append(sf)
        err = sf*np.sqrt((pc_err/pc_n)**2 + (ct_err/ct_n)**2)
        sf_errs.append(err)
    #return cut_vals, sfs, sf_errs
    ids =  (sf_errs<(1.5*np.nanpercentile(sf_errs,50)))&(~np.isnan(sf_errs))
    fit = np.polynomial.polynomial.polyfit( cut_vals[ids], np.array(sfs)[ids],w=1/np.array(sf_errs)[ids], deg=4)  

    xs = np.arange(-5,-0.5,0.01)
    p = np.polynomial.polynomial.polyval(xs, fit)
    cut_val = xs[np.argmin(np.abs(p-(100*dep_acc)))]

    return cut_val

def get_sf(energy,aoe,peak,fit_width,aoe_cut_val, display=1):
    #fwhm = np.sqrt(eres[0]+peak*eres[1])
    min_range = peak-fit_width[0]
    max_range = peak+fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy>min_range)&(energy<max_range)][:20000]
        peak_aoe = aoe[(energy>min_range)&(energy<max_range)][:20000]
    else:
        peak_energy = energy[(energy>min_range)&(energy<max_range)][:50000]
        peak_aoe = aoe[(energy>min_range)&(energy<max_range)][:50000]
    pars, errors = unbinned_energy_fit(peak_energy, peak)
    pc_n = pars[0]
    pc_err = errors[0]
    sfs = []
    sf_errs=[]
    
    cut_vals = np.arange(-5,5,0.2)
    #cut_vals = np.append(cut_vals, aoe_cut_val)
    final_cut_vals = []
    for cut_val in cut_vals:
        idxs = peak_aoe>cut_val
        try:
            cut_pars,ct_errs = unbinned_energy_fit(peak_energy[idxs],  peak)
        except:
            pass
        ct_n = cut_pars[0]
        ct_err = ct_errs[0]
        sf = (ct_n/pc_n)*100
        sfs.append(sf)
        err = sf*np.sqrt((pc_err/pc_n)**2 + (ct_err/ct_n)**2)
        sf_errs.append(err)
        final_cut_vals.append(cut_val)
    ids =  (sf_errs<(5*np.nanpercentile(sf_errs,50)))&(~np.isnan(sf_errs))&(np.array(sfs)<100)
    idxs = peak_aoe>aoe_cut_val
    cut_pars,ct_errs = unbinned_energy_fit(peak_energy[idxs],  peak)
    ct_n = cut_pars[0]
    ct_err = ct_errs[0]
    sf = (ct_n/pc_n)*100
    sf_err = sf*np.sqrt((pc_err/pc_n)**2 + (ct_err/ct_n)**2)
    return np.array(final_cut_vals)[ids], np.array(sfs)[ids],np.array(sf_errs)[ids], sf, sf_err

def compton_sf(energy,aoe,cut, peak,eres,display=1):
    fwhm = np.sqrt(eres[0]+peak*eres[1])

    emin           = peak - 2*fwhm
    emax           = peak + 2*fwhm
    sfs = []
    aoe = aoe[(energy>emin) & (energy<emax)]
    cut_vals = np.arange(-5,5,0.1)
    for cut_val in cut_vals:
        sfs.append(100*len(aoe[(aoe>cut_val)])/len(aoe))
    sf = 100*len(aoe[(aoe>cut)])/len(aoe)
    return sf, cut_vals,sfs

def get_sf_no_sweep(energy,aoe,peak,fit_width,aoe_low_cut_val, aoe_high_cut_val=None, display=1):
    min_range = peak-fit_width[0]
    max_range = peak+fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy>min_range)&(energy<max_range)][:20000]
        peak_aoe = aoe[(energy>min_range)&(energy<max_range)][:20000]
    else:
        peak_energy = energy[(energy>min_range)&(energy<max_range)][:50000]
        peak_aoe = aoe[(energy>min_range)&(energy<max_range)][:50000]
    pars, errors = unbinned_energy_fit(peak_energy, peak)
    pc_n = pars[0]
    pc_err = errors[0]
    if aoe_high_cut_val is None:
        idxs = peak_aoe>aoe_cut_val
    else:
        idxs = (peak_aoe>aoe_low_cut_val) & (peak_aoe<aoe_high_cut_val)
    cut_pars,ct_errs = unbinned_energy_fit(peak_energy[idxs],  peak)
    ct_n = cut_pars[0]
    ct_err = ct_errs[0]
    sf = (ct_n/pc_n)*100
    sf_err = sf*np.sqrt((pc_err/pc_n)**2 + (ct_err/ct_n)**2)
    return sf, sf_err

def compton_sf_no_sweep(energy,aoe, peak,eres, aoe_low_cut_val, aoe_high_cut_val=None, display=1):
    fwhm = np.sqrt(eres[0]+peak*eres[1])

    emin           = peak - 2*fwhm
    emax           = peak + 2*fwhm
    sfs = []
    aoe = aoe[(energy>emin) & (energy<emax)]
    cut_vals = np.arange(-5,5,0.1)
    if aoe_high_cut_val is None:
        sf = 100*len(aoe[(aoe>cut)])/len(aoe)
    else:
        sf = 100*len(aoe[(aoe>aoe_low_cut_val)&(aoe<aoe_high_cut_val)])/len(aoe)
    return sf

def get_classifier(aoe, energy, mu_pars, sigma_pars):
    classifier = aoe/(mu_pars[0]*energy + mu_pars[1])
    classifier = (classifier - 1)/ np.sqrt(sigma_pars[0] + (sigma_pars[1]/energy)**sigma_pars[2])
    return classifier



def cal_aoe(files, cal_dict, energy_param, cal_energy_param, dt_corr=False, cut_parameters = {"bl_mean":4, "bl_std":4,"pz_std":4}, plot_savepath=None, data_savepath=None):


    aoe, energy, dt, full_dt, eres_pars = load_aoe(files, cal_dict, energy_param, cal_energy_param, cut_parameters = cut_parameters)
    
    if data_savepath is not None:
        pathlib.Path(os.path.dirname(data_savepath)).mkdir(parents=True, exist_ok=True)
    
    
    print("Starting A/E correction")
    mu_pars,sigma_pars = AoEcorrection(energy,aoe,eres_pars)
    print("Finished A/E correction")
    
    classifier = get_classifier(aoe, energy, mu_pars, sigma_pars)

    cut = get_aoe_cut_fit(energy,classifier,1592,(40,20), 0.9, display=0)
    print("  Compute low side survival fractions: ")

    peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
    sf = np.zeros(len(peaks_of_interest))
    sferr = np.zeros(len(peaks_of_interest))
    fit_widths = [(40,25),(25,40), (0,0),(25,40), (50,50)]
    full_sfs =[]
    full_sf_errs =[]
    full_cut_vals =[]
    

    for i,peak in enumerate(peaks_of_interest):
        if peak == 2039:
            sf[i], cut_vals,sfs = compton_sf(energy,classifier,cut, peak,eres_pars)
            sferr[i] =0
            
            full_cut_vals.append(cut_vals)
            full_sfs.append(sfs)
            full_sf_errs.append(None)
        else:
            cut_vals, sfs,sf_errs, sf[i], sferr[i] = get_sf(energy,classifier, peak,fit_widths[i],cut)
            full_cut_vals.append(cut_vals)
            full_sfs.append(sfs)
            full_sf_errs.append(sf_errs)
            

        print(f'{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %')

    sf_2side = np.zeros(len(peaks_of_interest))
    sferr_2side = np.zeros(len(peaks_of_interest))
    print("Calculating 2 sided cut sfs")
    for i,peak in enumerate(peaks_of_interest):
        if peak == 2039:
            sf_2side[i] = compton_sf_no_sweep(energy,classifier, peak,eres_pars, cut,aoe_high_cut_val=4)#upper_cut=upper_cut
            sferr_2side[i] =0
        else:
            sf_2side[i], sferr_2side[i] = get_sf_no_sweep(energy,classifier, peak,fit_widths[i],cut,aoe_high_cut_val=4)

        print(f'{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %')
    print("Done")

    if plot_savepath is not None:
        mpl.use('pdf')
        pathlib.Path(os.path.dirname(plot_savepath)).mkdir(parents=True, exist_ok=True)
        with PdfPages(plot_savepath) as pdf:
            
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 16

            plt.figure()
            plt.subplot(3,2,1)
            plot_dt_dep(aoe, energy, dt, [1582,1602], f'Tl DEP')
            plt.subplot(3,2,2)
            plot_dt_dep(aoe, energy, dt, [1510,1630], f'Bi FEP')
            plt.subplot(3,2,3)
            plot_dt_dep(aoe, energy,dt,[2030,2050], 'Qbb')
            plt.subplot(3,2,4)
            plot_dt_dep(aoe, energy, dt, [2080,2120], f'Tl SEP')
            plt.subplot(3,2,5)
            plot_dt_dep(aoe, energy, dt, [2584,2638], f'Tl FEP')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            plt.figure()
            plot_compt_bands_overlayed(aoe,energy,[950,1250,1460,1660,1860,2060])
            plt.ylabel('Counts')
            plt.xlabel('Raw A/E')
            plt.title(f'Compton Bands before Correction')
            plt.legend(loc='upper left')
            pdf.savefig()
            plt.close()
            
            print("Starting A/E correction")
            mu_pars,sigma_pars = AoEcorrection(energy,aoe,eres_pars, pdf)
            print("Finished A/E correction")


            plt.figure()
            plot_compt_bands_overlayed(classifier,energy,[950,1250,1460,1660,1860,2060], [-5,5])
            plt.ylabel('Counts')
            plt.xlabel('Corrected A/E')
            plt.title(f'Compton Bands after Correction')
            plt.legend(loc='upper left')
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.vlines(cut,0,100, label=f'Cut Value: {cut:1.2f}', color='black')
            
            for i, peak in enumerate(peaks_of_interest):
                if peak == 2039:
                    plt.plot(full_cut_vals[i], full_sfs[i], label = f'{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %')
                else:
                    plt.errorbar(full_cut_vals[i], full_sfs[i], yerr=full_sf_errs[i], 
                        label = f'{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %')
            
                
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1,2,3,0,4,5]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right') 
            plt.xlabel('Cut Value')
            plt.ylabel('Survival Fraction %')
            pdf.savefig()
            plt.close()

            fig,ax = plt.subplots()
            bins = np.linspace(0,4000,4000)
            ax.hist(energy, bins=bins, histtype='step', label='Before PSD')
            ax.hist(energy[classifier>cut], bins=bins, histtype='step', label="Low side PSD cut")
            ax.hist(energy[(classifier>cut)&(classifier<4)], bins=bins, histtype='step', label="Double sided PSD cut")
            ax.hist(energy[(classifier<cut)|(classifier>4)], bins=bins, histtype='step', label="Rejected by PSD cut")

            axins = ax.inset_axes([0.25, 0.1, 0.3, 0.3])
            bins = np.linspace(1580,1640,200)
            axins.hist(energy, bins=bins, histtype='step')
            axins.hist(energy[classifier>cut], bins=bins, histtype='step')
            axins.hist(energy[(classifier>cut)&(classifier<4)], bins=bins, histtype='step')
            axins.hist(energy[(classifier<cut)|(classifier>4)], bins=bins, histtype='step')
            axins.set_xlabel("Energy (keV)")
            axins.set_ylabel("Counts")
            ax.set_xlim([0,4000])
            ax.set_yscale('log')
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend()
            pdf.savefig()
            plt.close()

            plt.figure()
            bins = np.linspace(0,3000,3000)
            counts_pass, bins_pass, _ = pgh.get_hist(energy[(classifier>cut)&(classifier<4)], bins =bins)
            counts, bins, _ = pgh.get_hist(energy, bins =bins)
            sf = counts_pass/(counts)

            plt.step(pgh.get_bin_centers(bins_pass),sf)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Survival Fraction")
            plt.ylim([0,1])
            pdf.savefig()
            plt.close()

    def convert_sfs_to_dict(peaks_of_interest, sfs, sf_errs):
        out_dict = {}
        for i,peak in enumerate(peaks_of_interest):
            out_dict[str(peak)] = {'sf':f"{sfs[i]:2f}", 'sf_err':f"{sf_errs[i]:2f}"}
        return out_dict

    out_dict = {'A/E_Energy_param': 'cuspEmax', 
        'Cal_energy_param': 'cuspEmax_ctc' ,
        'dt_param':'dt_eff',
        'rt_correction':False,
        'Mean_pars':list(mu_pars),
        'Sigma_pars':list(sigma_pars),
        'Low_cut': cut,
        'High_cut':4,
        'Low_side_sfs':convert_sfs_to_dict(peaks_of_interest, sf, sferr),
        '2_side_sfs':convert_sfs_to_dict(peaks_of_interest, sf_2side, sferr_2side)
        }
    
    if data_savepath is not None:
        with open(data_savepath, 'w') as w:
            json.dump(out_dict,w, indent=4)
    else:
        print(out_dict)
        return out_dict

