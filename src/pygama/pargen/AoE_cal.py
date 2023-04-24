"""
This module provides functions for correcting the a/e energy dependence, determining the cut level and calculating survival fractions.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
from datetime import datetime

import matplotlib as mpl

mpl.use("agg")
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.stats import chi2

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.ecal_th as thc
import pygama.pargen.energy_cal as pgc
import pygama.pargen.cuts as cts
from pygama.math.peak_fitting import nb_erfc

log = logging.getLogger(__name__)

def tag_pulser(files, lh5_path):
    pulser_df = lh5.load_dfs(files, ["timestamp", "trapTmax"], lh5_path)
    pulser_props = cts.find_pulser_properties(pulser_df, energy="trapTmax")
    if len(pulser_props) > 0:
        final_mask = None
        for entry in pulser_props:
            e_cut = (pulser_df.trapTmax.values < entry[0] + entry[1]) & (
                pulser_df.trapTmax.values > entry[0] - entry[1]
            )
            if final_mask is None:
                final_mask = e_cut
            else:
                final_mask = final_mask | e_cut
        ids = ~(final_mask)
        log.debug(f"pulser found: {pulser_props}")
    else:
        ids = np.ones(len(pulser_df), dtype=bool)
        log.debug(f"no pulser found")
    return ids

def load_aoe(
    files: list,
    lh5_path: str,
    cal_dict: dict,
    params:["A_max", "tp_0_est", "tp_99", "dt_eff", 
            'cuspEmax', "cuspEmax_ctc_cal", "is_valid_cal"],
    energy_param: str
) -> tuple(np.array, np.array, np.array, np.array):

    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    #switch this to dataframes, include timestamp

    sto = lh5.LH5Store()
    
    if isinstance(files, dict):
        if len(list(files))>1:
            df = []
            all_files = []
            for tstamp, tfiles in files.items():
                table = sto.read_object(lh5_path, tfiles)[0]
                if tstamp in cal_dict:
                    file_df = table.eval(cal_dict[tstamp]).get_dataframe()
                else:
                    file_df = table.eval(cal_dict).get_dataframe()
                file_df["timestamp"] = np.full(len(file_df),tstamp,dtype=object)
                ids = tag_pulser(tfiles, lh5_path)
                file_df["is_not_pulser"] = ids
                params.append("is_not_pulser")
                params.append("timestamp")
                df.append(file_df)
                all_files += tfiles
                
            df = pd.concat(df)
            
        else:
            for tstamp, tfiles in files.items():
                table = sto.read_object(lh5_path, tfiles)[0]
                if tstamp in cal_dict:
                    df = table.eval(cal_dict[tstamp]).get_dataframe()
                else:
                    df = table.eval(cal_dict).get_dataframe()
                df["timestamp"] = np.full(len(df),tstamp,dtype=object)
                params.append("timestamp")
                ids = tag_pulser(tfiles, lh5_path)
                df["is_not_pulser"] = ids
                params.append("is_not_pulser")
                all_files = tfiles
                
    elif isinstance(files, list):
        table = sto.read_object(lh5_path, files)[0]
        df = table.eval(cal_dict).get_dataframe()
        ids = tag_pulser(tfiles, lh5_path)
        df["is_not_pulser"] = ids
        params.append("is_not_pulser")
        all_files = files
    
    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True,axis=1)

    param_dict = {}
    for param in params:
        # add cuts in here
        if param not in df:
            df[param] = lh5.load_nda(all_files, [param], lh5_path)[param]
            
    df["AoE_uncorr"] = np.divide(df["A_max"], df[energy_param])
    return df


def PDF_AoE(
    x: np.array,
    lambda_s: float,
    lambda_b: float,
    mu: float,
    sigma: float,
    tau: float,
    lower_range: float = np.inf,
    upper_range: float = np.inf,
    components: bool = False,
) -> tuple(float, np.array):
    """
    PDF for A/E consists of a gaussian signal with gaussian tail background
    """
    try:
        sig = lambda_s * pgf.gauss_norm(x, mu, sigma)
        bkg = lambda_b * pgf.gauss_tail_norm(
            x, mu, sigma, tau, lower_range, upper_range
        )
    except:
        sig = np.full_like(x, np.nan)
        bkg = np.full_like(x, np.nan)

    if components == False:
        pdf = sig + bkg
        return lambda_s + lambda_b, pdf
    else:
        return lambda_s + lambda_b, sig, bkg


def unbinned_aoe_fit(
    aoe: np.array, display: int = 0, verbose: bool = False
) -> tuple(np.array, np.array):

    """
    Fitting function for A/E, first fits just a gaussian before using the full pdf to fit
    if fails will return NaN values
    """

    hist, bins, var = pgh.get_hist(aoe, bins=500)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    pars, cov = pgf.gauss_mode_max(hist, bins)
    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
    ls_guess = 2 * np.sum(hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))])
    c1_min = mu - 2 * sigma
    c1_max = mu + 5 * sigma

    # Initial fit just using Gaussian
    c1 = cost.UnbinnedNLL(aoe[(aoe < c1_max) & (aoe > c1_min)], pgf.gauss_pdf)
    m1 = Minuit(c1, mu, sigma, ls_guess)
    m1.limits = [
        (mu * 0.8, mu * 1.2),
        (0.8 * sigma, sigma * 1.2),
        (0, len(aoe[(aoe < c1_max) & (aoe > c1_min)])),
    ]
    m1.migrad()
    mu, sigma, ls_guess = m1.values
    if verbose:
        print(m1)

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = mu - 15 * sigma
    fmax = mu + 5 * sigma

    bg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - ls_guess
    x0 = [ls_guess, bg_guess, mu, sigma, 0.2, fmin, fmax, 0]
    if verbose:
        print(x0)

    # Full fit using gaussian signal with gaussian tail background
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax) & (aoe > fmin)], PDF_AoE)
    m = Minuit(c, *x0)
    m.fixed[5:] = True
    m.simplex().migrad()
    m.hesse()
    if verbose:
        print(m)

    if np.isnan(m.errors).all():
        try:
            m.simplex.migrad()
            m.minos()
            if np.isnan(m.errors).all():
                raise RuntimeError
        except:
            return np.full_like(x0, np.nan), np.full_like(x0, np.nan)

    if display > 1:
        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, bars = plt.hist(
            aoe[(aoe < fmax) & (aoe > fmin)], bins=400, histtype="step", label="Data"
        )
        dx = np.diff(bins)
        plt.plot(xs, PDF_AoE(xs, *m.values)[1] * dx[0], label="Full fit")
        n_events, sig, bkg = PDF_AoE(xs, *m.values[:-1], True)
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(xs, pgf.gauss_pdf(xs, *m1.values) * dx[0], label="Initial Gaussian")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        bin_centers = (bins[1:] + bins[:-1]) / 2
        res = (PDF_AoE(bin_centers, *m.values)[1] * dx[0]) - counts
        plt.plot(
            bin_centers,
            [re / count if count != 0 else re for re, count in zip(res, counts)],
            label="Normalised Residuals",
        )
        plt.legend(loc="upper left")
        plt.show()
        return m.values, m.errors

    else:
        return m.values, m.errors

def fit_time_means(tstamps,means, errs):
    out_dict = {}
    current_tstamps = []
    current_means = []
    current_errs = []
    rolling_mean = means[np.where((np.abs(np.diff(means))<(5*np.array(errs)[1:]))&\
                                  (~np.isnan(np.abs(np.diff(means))<(5*np.array(errs)[1:]))))[0][0]]
    for i,tstamp in enumerate(tstamps):
        if np.abs(means[i]-rolling_mean) >  5* errs[i] or np.isnan(means[i]) or np.isnan(errs[i]):
            if i+1 == len(means):
                out_dict[tstamp] = np.nan
            else:
                if np.abs(means[i+1]-means[i]) <  5* errs[i+1] and not (np.isnan(means[i]) or np.isnan(means[i+1]) or np.isnan(errs[i]) or np.isnan(errs[i+1])):
                    for ts in current_tstamps:
                        out_dict[ts] = rolling_mean
                    rolling_mean= means[i]
                    current_means = [means[i]]
                    current_tstamps=[tstamp]
                    current_errs = [errs[i]]
                else:
                    out_dict[tstamp] = np.nan
        else:
            current_tstamps.append(tstamp)
            current_means.append(means[i])
            current_errs.append(errs[i])
            rolling_mean = np.average(current_means, weights=1/ np.array(current_errs))
    for tstamp in current_tstamps:
        out_dict[tstamp] = rolling_mean
    return out_dict

def aoe_timecorr(df, energy_param, plot_dict={} , display=0):
    if "timestamp" in df:
        tstamps = sorted(np.unique(df["timestamp"]))
        if len(tstamps)>1:
            means = []
            errors = []
            reses=[]
            res_errs = []
            final_tstamps=[]
            for tstamp, time_df in df.groupby("timestamp", sort=True):
                pars, errs = unbinned_aoe_fit(
                    time_df.query(f"is_valid_cal& is_not_pulser & cuspEmax_ctc_cal>1000 & cuspEmax_ctc_cal<1300")["AoE_uncorr"])
                final_tstamps.append(tstamp)
                means.append(pars[2])
                errors.append(errs[2])
                reses.append(pars[3]/pars[2])
                res_errs.append(reses[-1]*np.sqrt(errs[3]/pars[3] + errs[2]/pars[2]))
            mean_dict = fit_time_means(tstamps,means, errors)
            
            df["AoE_timecorr"] = df["AoE_uncorr"] /np.array([mean_dict[tstamp] for tstamp in df["timestamp"]])
            out_dict = {tstamp:
            {"AoE_Timecorr": {
                "expression": f"(A_max/{energy_param})/a",
                "parameters": {"a": mean_dict[tstamp]},
            }} for tstamp in mean_dict
            }
            if display>0:
                fig1, ax = plt.subplots(1,1)
                ax.errorbar([datetime.strptime(tstamp, '%Y%m%dT%H%M%SZ') for tstamp in tstamps],
                            means, yerr = errors, linestyle=" ")
                ax.step([datetime.strptime(tstamp, '%Y%m%dT%H%M%SZ') for tstamp in list(mean_dict)], 
                        [mean_dict[tstamp] for tstamp in mean_dict], where="post")
                ax.set_xlabel("time")
                ax.set_ylabel("A/E mean")
                myFmt = mdates.DateFormatter('%b %d')
                ax.xaxis.set_major_formatter(myFmt)
                plot_dict["aoe_time"] = fig1
                if display > 1:
                    plt.show()
                else:
                    plt.close()
                fig2, ax = plt.subplots(1,1)
                ax.errorbar([datetime.strptime(tstamp, '%Y%m%dT%H%M%SZ') for tstamp in tstamps],
                            reses, yerr = res_errs, linestyle=" ")
                ax.set_xlabel("time")
                ax.set_ylabel("A/E res")
                myFmt = mdates.DateFormatter('%b %d')
                ax.xaxis.set_major_formatter(myFmt)
                plot_dict["aoe_res"] = fig2
                if display > 1:
                    plt.show()
                else:
                    plt.close()
                return df, out_dict, {"times":tstamps, "mean":means,
                 "mean_errs":errors, "res": reses, 
                 "res_errs":res_errs},  plot_dict
            else:
                return df, out_dict, {"times":tstamps, "mean":means, 
                "mean_errs":errors , "res": reses, "res_errs":res_errs}
    pars, errs = unbinned_aoe_fit(
        data.query("is_valid_cal& is_not_pulser & cuspEmax_ctc_cal>1000 & cuspEmax_ctc_cal<1300")["AoE_uncorr"])
    df["AoE_timecorr"]=df["AoE_uncorr"]/pars[2]
    out_dict = {
    "AoE_Timecorr": {
        "expression": f"(A_max/{energy_param})/a",
        "parameters": {"a": pars[2]},
    }
    }
    if display>0:
        return df, out_dict, {"times":[np.nan], "mean":[pars[2]], "res": [pars[2]/pars[3]]}, plot_dict
    else:
        return df, out_dict, {"times":[np.nan], "mean":[pars[2]], "res": [pars[2]/pars[3]]}

def pol1(x: np.array, a: float, b: float) -> np.array:
    """Basic Polynomial for fitting A/E centroid against energy"""
    return a * x + b


def sigma_fit(x: np.array, a: float, b: float, c: float) -> np.array:
    """Function definition for fitting A/E sigma against energy"""
    return np.sqrt(a + (b / x) ** c)


def AoEcorrection(
    energy: np.array, aoe: np.array, eres: list, plot_dict: dict = {}, display: int = 0,
    comptBands_width = 20
) -> tuple(np.array, np.array):

    """
    Calculates the corrections needed for the energy dependence of the A/E.
    Does this by fitting the compton continuum in slices and then applies fits to the centroid and variance.
    """

    comptBands =  np.arange(900,2350,comptBands_width)
    peaks = np.array([1080,1094,1459,1512, 1552, 1592,1620, 1650, 1670,1830,2105])
    allowed =np.array([],dtype=bool)
    for i, band in enumerate(comptBands):
        allow=True
        for peak in peaks:
            if (peak-5)>band and (peak-5)<(band+ comptBands_width):
                allow = False
            elif (peak+5>band) and (peak+5)<(band + comptBands_width):
                allow = False
        allowed = np.append(allowed,allow)
    comptBands = comptBands[allowed]

    results_dict = {}
    comptBands = comptBands[::-1]  # Flip so color gets darker when plotting
    compt_aoe = np.zeros(len(comptBands))
    aoe_sigmas = np.zeros(len(comptBands))
    compt_aoe_err = np.zeros(len(comptBands))
    aoe_sigmas_err = np.zeros(len(comptBands))
    ratio = np.zeros(len(comptBands))
    ratio_err = np.zeros(len(comptBands))

    copper = cm = plt.get_cmap("copper")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(comptBands))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=copper)

    if display > 0:
        fits_fig = plt.figure()

    # Fit each compton band
    for i, band in enumerate(comptBands):
        aoe_tmp = aoe[
            (energy > band) & (energy < band + comptBands_width) & (aoe > 0)
        ]  # [:20000]
        try:
            pars, errs = unbinned_aoe_fit(aoe_tmp, display=display)
            compt_aoe[i] = pars[2]
            aoe_sigmas[i] = pars[3]
            compt_aoe_err[i] = errs[2]
            aoe_sigmas_err[i] = errs[3]
            ratio[i] = pars[0] / pars[1]
            ratio_err[i] = ratio[i] * np.sqrt(
                (errs[0] / pars[0]) ** 2 + (errs[1] / pars[1]) ** 2
            )
        except:
            compt_aoe[i] = np.nan
            aoe_sigmas[i] = np.nan
            compt_aoe_err[i] = np.nan
            aoe_sigmas_err[i] = np.nan
            ratio[i] = np.nan
            ratio_err[i] = np.nan

        if display > 0:
            if np.isnan(errs[2]) | np.isnan(errs[3]) | (errs[2] == 0) | (errs[3] == 0):
                pass
            else:
                xs = np.arange(
                    pars[2] - 4 * pars[3], pars[2] + 3 * pars[3], pars[3] / 10
                )
                colorVal = scalarMap.to_rgba(i)
                plt.plot(xs, PDF_AoE(xs, *pars)[1], color=colorVal)

    if display > 0:
        plt.xlabel("A/E")
        plt.ylabel("Expected Counts")
        plt.title("Compton Band Fits")
        cbar = plt.colorbar(
            cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("copper_r")),
            orientation="horizontal",
            label="Compton Band Energy",
            ticks=[0, 16, 32, len(comptBands)],
        )  # cax=ax,
        cbar.ax.set_xticklabels(
            [
                comptBands[::-1][0],
                comptBands[::-1][16],
                comptBands[::-1][32],
                comptBands[::-1][-1],
            ]
        )
        plot_dict["band_fits"] = fits_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

    ids = (
        np.isnan(compt_aoe_err)
        | np.isnan(aoe_sigmas_err)
        | (aoe_sigmas_err == 0)
        | (compt_aoe_err == 0)
    )
    results_dict["n_of_valid_fits"] = len(np.where(~ids)[0])
    # Fit mus against energy
    p0_mu = [-1e-06, 5e-01]
    c_mu = cost.LeastSquares(
        comptBands[~ids], compt_aoe[~ids], compt_aoe_err[~ids], pol1
    )
    c_mu.loss = "soft_l1"
    m_mu = Minuit(c_mu, *p0_mu)
    m_mu.simplex()
    m_mu.migrad()
    m_mu.hesse()

    pars = m_mu.values
    errs = m_mu.errors

    csqr_mu = np.sum(
        ((compt_aoe[~ids] - pol1(comptBands[~ids], *pars)) ** 2) / compt_aoe_err[~ids]
    )
    dof_mu = len(compt_aoe[~ids]) - len(pars)
    results_dict["p_val_mu"] = chi2.sf(csqr_mu, dof_mu)
    results_dict["csqr_mu"] = (csqr_mu, dof_mu)

    # Fit sigma against energy
    p0_sig = [np.nanpercentile(aoe_sigmas[~ids], 50) ** 2, 2, 2]
    c_sig = cost.LeastSquares(
        comptBands[~ids], aoe_sigmas[~ids], aoe_sigmas_err[~ids], sigma_fit
    )
    c_sig.loss = "soft_l1"
    m_sig = Minuit(c_sig, *p0_sig)
    m_sig.simplex()
    m_sig.migrad()
    m_sig.hesse()

    sig_pars = m_sig.values
    sig_errs = m_sig.errors

    csqr_sig = np.sum(
        ((aoe_sigmas[~ids] - sigma_fit(comptBands[~ids], *sig_pars)) ** 2)
        / aoe_sigmas_err[~ids]
    )
    dof_sig = len(aoe_sigmas[~ids]) - len(sig_pars)
    results_dict["p_val_sig"] = chi2.sf(csqr_sig, dof_sig)
    results_dict["csqr_sig"] = (csqr_sig, dof_sig)

    model = pol1(comptBands, *pars)
    sig_model = sigma_fit(comptBands, *sig_pars)

    # Get DEP fit
    sigma = np.sqrt(eres[0] + 1592 * eres[1]) / 2.355
    n_sigma = 4
    peak = 1592
    emin = peak - n_sigma * sigma
    emax = peak + n_sigma * sigma
    try:
        dep_pars, dep_err = unbinned_aoe_fit(
            aoe[(energy > emin) & (energy < emax) & (aoe > 0)][:10000]
        )
    except:
        dep_pars = np.full(6,np.nan)
        dep_err = np.full(6,np.nan)

    if display > 0:
        mean_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            compt_aoe[~ids],
            yerr=compt_aoe_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        ax1.plot(comptBands[~ids] + 10, model[~ids], label="linear model")
        ax1.errorbar(
            1592,
            dep_pars[2],
            xerr=n_sigma * sigma,
            yerr=dep_err[2],
            label="DEP",
            color="green",
            linestyle=" ",
        )

        ax1.legend(title="A/E mu energy dependence", frameon=False)

        ax1.set_ylabel("raw A/E (a.u.)", ha="right", y=1)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (compt_aoe[~ids] - model[~ids]) / compt_aoe_err[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592, 100 * (dep_pars[2] - pol1(1592, *pars)) / dep_err[2], lw=1, c="g"
        )
        ax2.set_ylabel("Residuals %", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["mean_fit"] = mean_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

    if display > 0:
        sig_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            aoe_sigmas[~ids],
            yerr=aoe_sigmas_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        ax1.plot(
            comptBands[~ids],
            sig_model[~ids],
            label=f"sqrt model: sqrt({sig_pars[0]:1.4f}+({sig_pars[1]:1.1f}/E)^{sig_pars[2]:1.1f})",
        )  # {sig_pars[2]:1.1f}
        ax1.errorbar(
            1592,
            dep_pars[3],
            xerr=n_sigma * sigma,
            yerr=dep_err[3],
            label="DEP",
            color="green",
        )
        ax1.set_ylabel("A/E stdev (a.u.)", ha="right", y=1)
        ax1.legend(title="A/E stdev energy dependence", frameon=False)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (aoe_sigmas[~ids] - sig_model[~ids]) / aoe_sigmas_err[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100 * (dep_pars[3] - sigma_fit(1592, *sig_pars)) / dep_err[3],
            lw=1,
            c="g",
        )
        ax2.set_ylabel("Residuals", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["sigma_fit"] = sig_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return pars, sig_pars, results_dict, plot_dict
    else:
        return pars, sig_pars, results_dict


def plot_compt_bands_overlayed(
    aoe: np.array, energy: np.array, eranges: list[tuple], aoe_range: list[float] = None
) -> None:

    """
    Function to plot various compton bands to check energy dependence and corrections
    """

    for erange in eranges:
        hist, bins, var = pgh.get_hist(
            aoe[(energy > erange - 10) & (energy < erange + 10) & (~np.isnan(aoe))],
            bins=500,
        )
        bin_cs = (bins[1:] + bins[:-1]) / 2
        mu = bin_cs[np.argmax(hist)]
        if aoe_range is None:
            aoe_range = [mu * 0.97, mu * 1.02]
        idxs = (
            (energy > erange - 10)
            & (energy < erange + 10)
            & (aoe > aoe_range[0])
            & (aoe < aoe_range[1])
        )
        plt.hist(aoe[idxs], bins=50, histtype="step", label=f"{erange-10}-{erange+10}")


def plot_dt_dep(
    aoe: np.array, energy: np.array, dt: np.array, erange: list[tuple], title: str
) -> None:

    """
    Function to produce 2d histograms of A/E against drift time to check dependencies
    """

    hist, bins, var = pgh.get_hist(
        aoe[(energy > erange[0]) & (energy < erange[1]) & (~np.isnan(aoe))], bins=500
    )
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > erange[0])
        & (energy < erange[1])
        & (aoe > aoe_range[0])
        & (aoe < aoe_range[1])
        & (dt < 2000)
    )

    plt.hist2d(aoe[idxs], dt[idxs], bins=[200, 100], norm=LogNorm())
    plt.ylabel("Drift Time (ns)")
    plt.xlabel("A/E")
    plt.title(title)


def energy_guess(hist, bins, var, func_i, peak, eres_pars, fit_range):
    """
    Simple guess for peak fitting
    """
    if func_i == pgf.extended_radford_pdf:
        bin_cs = (bins[1:] + bins[:-1]) / 2
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
        i_0 = np.nanargmax(hist)
        mu = peak
        height = hist[i_0]
        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])-((n_bins_range*2)*(bg0-step/2))
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0
        parguess = [
            nsig_guess,
            mu,
            sigma,
            htail,
            tau,
            nbkg_guess,
            hstep,
            fit_range[0],
            fit_range[1],
            0,
        ]  #
        return parguess

    elif func_i == pgf.extended_gauss_step_pdf:
        mu = peak
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0
        return [nsig_guess, mu, sigma, nbkg_guess, hstep, fit_range[0], fit_range[1], 0]


def unbinned_energy_fit(
    energy: np.array,
    peak: float,
    eres_pars: list = None,
    simplex=False,
    guess=None,
    display=0,
    verbose: bool = False,
) -> tuple(np.array, np.array):
    """
    Fitting function for energy peaks used to calculate survival fractions
    """
    hist, bins, var = pgh.get_hist(
        energy, dx=0.5, range=(np.nanmin(energy), np.nanmax(energy))
    )
    sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
    if guess is None:
        x0 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_gauss_step_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_gauss_step_pdf)
        m = Minuit(c, *x0)
        m.limits = [
        (0, None),
        (peak-1, peak+1),
        (None, None),
         (0, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]
        m.fixed[-3:] = True
        m.simplex().migrad()
        m.hesse()
        x0 = m.values[:3]
        x0 += [1 / 5, 0.2 * m.values[2]]
        x0 += m.values[3:]
        if verbose:
            print(m)
        bounds = [
        (0, None),
        (peak-1, peak+1),
        (None, None),
        (0, None),
        (None, None),
        (0, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]
    else:
        x0 = guess
        x1 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_radford_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        x0[0]=x1[0]
        x0[5]=x1[5]
        bounds = [
        (0, None),
        (guess[1]-0.5, guess[1]+0.5),
        sorted((0.8*guess[2], 1.2*guess[2])),
        sorted((0.8*guess[3], 1.2*guess[3])),
        sorted((0.8*guess[4], 1.2*guess[4])),
        (0, None),
        sorted((0.8*guess[6], 1.2*guess[6])),
        (None, None),
        (None, None),
        (None, None),
    ]
    if len(x0) == 0:
        return [np.nan], [np.nan]

    fixed=[1,2,3,4,6,7,8,9]
    if verbose:
        print(x0)
    c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_radford_pdf)
    m = Minuit(c, *x0)
    m.limits = bounds
    for fix in fixed:
        m.fixed[fix] = True
    if simplex == True:
        m.simplex().migrad()
    else:
        m.migrad()

    m.hesse()
    if verbose:
        print(m)
    if display>1:
        plt.figure()
        bcs = (bins[1:]+bins[:-1])/2
        plt.step(bcs,hist,where="mid")
        plt.plot(bcs, pgf.radford_pdf(bcs, *x0)*np.diff(bcs)[0])
        plt.plot(bcs, pgf.radford_pdf(bcs, *m.values)*np.diff(bcs)[0])
        plt.show()
    

    if not np.isnan(m.errors[:-3]).all():
        return m.values, m.errors
    else:
        try:
            m.simplex().migrad()
            m.minos()
            if not np.isnan(m.errors[:-3]).all():
                return m.values, m.errors
        except:
             return np.full_like(x0, np.nan), np.full_like(x0, np.nan)


def get_peak_label(peak: float) -> str:
    if peak == 2039:
        return "CC @"
    elif peak == 1592.5:
        return "Tl DEP @"
    elif peak == 1620.5:
        return "Bi FEP @"
    elif peak == 2103.53:
        return "Tl SEP @"
    elif peak == 2614.5:
        return "Tl FEP @"


def get_survival_fraction(
    energy,
    aoe,
    cut_val,
    peak,
    eres_pars,
    high_cut=None,
    guess_pars_cut=None,
    guess_pars_surv=None,
    display=0
):
    nan_idxs = np.isnan(aoe)
    if high_cut is not None:
        idxs = (aoe > cut_val) & (aoe < high_cut)
    else:
        idxs = (aoe > cut_val)
    
    if guess_pars_cut is None or guess_pars_surv is None:
        pars, errs = unbinned_energy_fit(
        energy, peak, eres_pars, simplex=True
        )
        guess_pars_cut = pars
        guess_pars_surv = pars
    
        
    cut_pars, ct_errs = unbinned_energy_fit(
        energy[(~nan_idxs)&(~idxs)], peak, eres_pars, guess=guess_pars_cut, simplex=False, display=display, verbose=False
    )
    surv_pars, surv_errs = unbinned_energy_fit(
        energy[(~nan_idxs)&(idxs)], peak, eres_pars, guess=guess_pars_surv, simplex=False,display=display
    )

    ct_n = cut_pars[0]
    ct_err = ct_errs[0]
    surv_n = surv_pars[0]
    surv_err = surv_errs[0]

    pc_n = ct_n + surv_n
    pc_err = np.sqrt(surv_err**2 + ct_err**2)

    sf = (surv_n / pc_n) * 100
    err = sf * np.sqrt((pc_err / pc_n) ** 2 + (surv_err / surv_n) ** 2)
    return sf, err, cut_pars, surv_pars


def get_aoe_cut_fit(
    energy: np.array,
    aoe: np.array,
    peak: float,
    ranges: tuple(int, int),
    dep_acc: float,
    eres_pars: list,
    display: int = 1,
) -> float:

    """
    Determines A/E cut by sweeping through values and for each one fitting the DEP to determine how many events survive.
    Then interpolates to get cut value at desired DEP survival fraction (typically 90%)
    """

    min_range, max_range = ranges

    peak_energy = energy[(energy > peak - min_range) & (energy < peak + max_range)&(~np.isnan(aoe))]
    peak_aoe = aoe[(energy > peak - min_range) & (energy < peak + max_range)&(~np.isnan(aoe))]

    cut_vals = np.arange(-8, 0, 0.2)
    sfs = []
    sf_errs = []
    for cut_val in cut_vals:
        sf, err, cut_pars, surv_pars = get_survival_fraction(
            peak_energy,
            peak_aoe,
            cut_val,
            peak,
            eres_pars,
            guess_pars_cut=None,
            guess_pars_surv=None,
        )
        sfs.append(sf)
        sf_errs.append(err)

    # return cut_vals, sfs, sf_errs
    ids = (
        (sf_errs < (1.5 * np.nanpercentile(sf_errs, 85)))
        & (~np.isnan(sf_errs))
    )
    def fit_func(x,a,b,c,d):
        return (a+b*x)*nb_erfc(c*x+d)

    c = cost.LeastSquares(cut_vals[ids], np.array(sfs)[ids], np.array(sf_errs)[ids], fit_func)
    c.loss = "soft_l1"
    m1 = Minuit(c, np.nanmax(sfs)/2,0,1,1.5)
    m1.simplex().migrad()
    xs = np.arange(np.nanmin(cut_vals[ids]), np.nanmax(cut_vals[ids]), 0.01)
    p = fit_func(xs, *m1.values)
    cut_val = round(xs[np.argmin(np.abs(p - (100 * 0.9)))],3)

    if display > 0:
        plt.figure()
        plt.errorbar(
            cut_vals[ids],
            np.array(sfs)[ids],
            yerr=np.array(sf_errs)[ids],
            linestyle=" ",
        )

        plt.plot(xs, p)
        plt.hline((100 * dep_acc),[xs[0], cut_val])
        plt.vline(cut_val,[50, (100 * dep_acc)])
        plt.ylim([50,100])
        plt.show()

    return cut_val


def get_sf(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    aoe_cut_val: float,
    eres_pars: list,
    display: int = 0,
) -> tuple(np.array, np.array, np.array, float, float):

    """
    Calculates survival fraction for gamma lines using fitting method as in cut determination
    """

    # fwhm = np.sqrt(eres[0]+peak*eres[1])
    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))]
    else:
        peak_energy = energy[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))][:50000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))][:50000]
    pars, errors = unbinned_energy_fit(peak_energy, peak, eres_pars, simplex=False)
    pc_n = pars[0]
    pc_err = errors[0]
    sfs = []
    sf_errs = []

    cut_vals = np.arange(-5, 5, 0.2)
    # cut_vals = np.append(cut_vals, aoe_cut_val)
    final_cut_vals = []
    for cut_val in cut_vals:
        try:

            sf, err, cut_pars, surv_pars = get_survival_fraction(
                peak_energy, peak_aoe, cut_val, peak, eres_pars
            )
            if np.isnan(cut_pars).all() == False and np.isnan(surv_pars).all() == False:
                guess_pars_cut = cut_pars
                guess_pars_surv = surv_pars
        except:
            sf = np.nan
            err = np.nan
        sfs.append(sf)
        sf_errs.append(err)
        final_cut_vals.append(cut_val)
    ids = (
        (sf_errs < (5 * np.nanpercentile(sf_errs, 50)))
        & (~np.isnan(sf_errs))
        & (np.array(sfs) < 100)
    )
    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy, peak_aoe, aoe_cut_val, peak, eres_pars
    )

    if display > 0:
        plt.figure()
        plt.errorbar(cut_vals, sfs, sf_errs)
        plt.show()

    return (
        np.array(final_cut_vals)[ids],
        np.array(sfs)[ids],
        np.array(sf_errs)[ids],
        sf,
        sf_err,
    )


def compton_sf(
    energy: np.array,
    aoe: np.array,
    cut: float,
    peak: float,
    eres: list[float, float],
    display: int = 1,
) -> tuple(float, np.array, list):
    """
    Determines survival fraction for compton continuum by basic counting
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    sf_errs=[]
    aoe = aoe[(energy > emin) & (energy < emax)&(~np.isnan(aoe))]
    cut_vals = np.arange(-5, 5, 0.1)
    for cut_val in cut_vals:
        sfs.append(100 * len(aoe[(aoe > cut_val)]) / len(aoe))
        sf_errs.append(sfs[-1] * np.sqrt((1/len(aoe)) + 1/len(aoe[(aoe > cut_val)])))
    sf = 100 * len(aoe[(aoe > cut)]) / len(aoe)
    sf_err = sf * np.sqrt(1/len(aoe) + 1/len(aoe[(aoe > cut)]))
    return cut_vals, sfs, sf_errs, sf, sf_err


def get_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    eres_pars: list,
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    display: int = 1,
) -> tuple(float, float):

    """
    Calculates survival fraction for gamma line without sweeping through values
    """

    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))]
    else:
        peak_energy = energy[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))][:50000]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range)&(~np.isnan(aoe))][:50000]

    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy,
        peak_aoe,
        aoe_low_cut_val,
        peak,
        eres_pars,
        high_cut=aoe_high_cut_val,
    )
    return sf, sf_err


def compton_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    eres: list[float, float],
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    display: int = 1,
) -> float:

    """
    Calculates survival fraction for compton contiuum without sweeping through values
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    aoe = aoe[(energy > emin) & (energy < emax)&(~np.isnan(aoe))]
    if aoe_high_cut_val is None:
        sf = 100 * len(aoe[(aoe > aoe_low_cut_val)]) / len(aoe)
        sf_err = sf * np.sqrt(1/len(aoe) + 1/len(aoe[(aoe > aoe_low_cut_val)]))
    else:
        sf = (
            100
            * len(aoe[(aoe > aoe_low_cut_val) & (aoe < aoe_high_cut_val)])
            / len(aoe)
        )
        sf_err = sf * np.sqrt(1/len(aoe) + 1/len(aoe[(aoe > aoe_low_cut_val) & (aoe < aoe_high_cut_val)]))
    return sf, sf_err


def get_classifier(
    df,
    cal_energy_param,
    mu_pars: list[float, float],
    sigma_pars: list[float, float],
) -> np.array:
    """
    Applies correction to A/E energy dependence
    """

    classifier = df["AoE_timecorr"] / (mu_pars[0] * df[cal_energy_param] + mu_pars[1])
    df["AoE_classifier"] = (classifier - 1) / sigma_fit(df[cal_energy_param], *sigma_pars)
    return df


def get_dt_guess(hist: np.array, bins: np.array, var: np.array) -> list:
    """
    Guess for fitting dt spectrum
    """

    mu, sigma, amp = pgh.get_gaussian_guess(hist, bins)
    i_0 = np.argmax(hist)
    bg = np.mean(hist[-10:])
    step = bg - np.mean(hist[:10])
    hstep = step / (bg + np.mean(hist[:10]))
    dx = np.diff(bins)[0]
    n_bins_range = int((3 * sigma) // dx)
    nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
    nbkg_guess = np.sum(hist) - nsig_guess
    return [
        nsig_guess * np.diff(bins)[0],
        mu,
        sigma,
        nbkg_guess * np.diff(bins)[0],
        hstep,
        np.inf,
        np.inf,
        0,
    ]


def apply_dtcorr(aoe: np.array, dt: np.array, alpha: float) -> np.array:
    """Aligns dt regions"""
    return aoe * (1 + alpha * dt)


def drift_time_correction(
    aoe: np.array,
    energy: np.array,
    dt: np.array,
    display: int = 0,
    plot_dict: dict = {},
) -> tuple(np.array, float):
    """
    Calculates the correction needed to align the two drift time regions for ICPC detectors
    """
    hist, bins, var = pgh.get_hist(aoe[(energy > 1582) & (energy < 1602)], bins=500)
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > 1582) & (energy < 1602) & (aoe > aoe_range[0]) & (aoe < aoe_range[1])
    )

    mask = (
        (idxs)
        & (dt < np.nanpercentile(dt[idxs], 55))
        & (dt > np.nanpercentile(dt[idxs], 1))
    )

    hist, bins, var = pgh.get_hist(
        dt[mask], dx=10, range=(np.nanmin(dt[mask]), np.nanmax(dt[mask]))
    )

    gpars = get_dt_guess(hist, bins, var)
    dt_pars, dt_errs, dt_cov = pgf.fit_binned(
        pgf.gauss_step_pdf,
        hist,
        bins,
        guess=gpars,
        fixed=[-3, -2, -1],
        cost_func="Least Squares",
    )

    aoe_mask = (
        (idxs) & (dt > dt_pars[1] - 2 * dt_pars[2]) & (dt < dt_pars[1] + 2 * dt_pars[2])
    )
    aoe_tmp = aoe[aoe_mask]
    aoe_pars, aoe_errs = unbinned_aoe_fit(aoe_tmp, display=display)

    mask2 = (
        (idxs)
        & (dt > np.nanpercentile(dt[idxs], 50))
        & (dt < np.nanpercentile(dt[idxs], 99))
    )
    hist2, bins2, var2 = pgh.get_hist(
        dt[mask2], dx=10, range=(np.nanmin(dt[mask2]), np.nanmax(dt[mask2]))
    )
    gpars2 = get_dt_guess(hist2, bins2, var2)

    dt_pars2, dt_errs2, dt_cov2 = pgf.fit_binned(
        pgf.gauss_step_pdf,
        hist2,
        bins2,
        guess=gpars2,
        fixed=[-3, -2, -1],
        cost_func="Least Squares",
    )

    aoe_mask2 = (
        (idxs)
        & (dt > dt_pars2[1] - 2 * dt_pars2[2])
        & (dt < dt_pars2[1] + 2 * dt_pars2[2])
    )
    aoe_tmp2 = aoe[aoe_mask2]
    aoe_pars2, aoe_errs2 = unbinned_aoe_fit(aoe_tmp2, display=display)

    alpha = (aoe_pars[2] - aoe_pars2[2]) / (
        dt_pars2[1] * aoe_pars2[2] - dt_pars[1] * aoe_pars[2]
    )
    aoe_corrected = apply_dtcorr(aoe, dt, alpha)

    if display > 0:
        dt_fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.step(pgh.get_bin_centers(bins), hist, label="Data")
        plt.plot(
            pgh.get_bin_centers(bins),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins), *gpars),
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins), *dt_pars),
            label="Fit",
        )
        plt.xlabel("Drift Time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

        plt.subplot(3, 2, 2)
        plt.step(pgh.get_bin_centers(bins2), hist2, label="Data")
        plt.plot(
            pgh.get_bin_centers(bins2),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins2), *gpars2),
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins2),
            pgf.gauss_step_pdf(pgh.get_bin_centers(bins2), *dt_pars2),
            label="Fit",
        )
        plt.xlabel("Drift Time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

        plt.subplot(3, 2, 3)
        xs = np.linspace(aoe_pars[-3], aoe_pars[-2], 1000)
        counts, aoe_bins, bars = plt.hist(
            aoe[(aoe < aoe_pars[-2]) & (aoe > aoe_pars[-3]) & aoe_mask],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins)
        plt.plot(xs, PDF_AoE(xs, *aoe_pars)[1] * dx[0], label="Full fit")
        # plt.yscale('log')
        n_events, sig, bkg = PDF_AoE(xs, *aoe_pars[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(3, 2, 4)
        xs = np.linspace(aoe_pars2[-3], aoe_pars2[-2], 1000)
        counts, aoe_bins2, bars = plt.hist(
            aoe[(aoe < aoe_pars2[-2]) & (aoe > aoe_pars2[-3]) & aoe_mask2],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins2)
        plt.plot(xs, PDF_AoE(xs, *aoe_pars2)[1] * dx[0], label="Full fit")
        # plt.yscale('log')
        n_events, sig, bkg = PDF_AoE(xs, *aoe_pars2[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(3, 2, 5)
        counts, bins, bars = plt.hist(
            aoe[idxs], bins=200, histtype="step", label="Uncorrected"
        )
        plt.hist(aoe_corrected[idxs], bins=bins, histtype="step", label="Corrected")
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlim([aoe_pars[-3], aoe_pars[-2] * 1.01])
        plot_dict["dt_corr"] = dt_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return aoe_corrected, alpha, plot_dict
    else:
        return aoe_corrected, alpha


def cal_aoe(
    files: list,
    lh5_path,
    cal_dict: dict,
    energy_param: str,
    cal_energy_param: str,
    eres_pars: list,
    cut_field: str = "is_valid_cal",
    dt_corr: bool = False,
    aoe_high_cut: int = 4,
    display: int = 0,
) -> tuple(dict, dict):
    """
    Main function for running the a/e correction and cut determination.
    """

    df = load_aoe(
        files, lh5_path, cal_dict, 
        ["A_max", "tp_0_est", "tp_99", "dt_eff", 
        energy_param, cal_energy_param, cut_field],
        energy_param=energy_param
    )
    try:
        df, timecorr_dict, res_dict = aoe_timecorr(df, energy_param)
    except:
        res_dict ={}
        timecorr_dict = {
                "AoE_Timecorr": {
                    "expression": f"(A_max/{energy_param})/a",
                    "parameters": {"a": np.nan},
                }
                }
        

    if re.match("(\d{8})T(\d{6})Z", list(cal_dict)[0]):
        for tstamp in cal_dict:
            if tstamp in timecorr_dict:
                cal_dict[tstamp].update(timecorr_dict[tstamp])
            else:
                cal_dict[tstamp].update(timecorr_dict)
    else:
        cal_dict.update(timecorr_dict)


    # if dt_corr == True:
    #     aoe, alpha = drift_time_correction(aoe_uncorr, energy, dt)
    # else:
    #     aoe = aoe_uncorr
    try:
        log.info("Starting A/E correction")
        mu_pars, sigma_pars, results_dict = AoEcorrection(
            df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
            df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
            eres_pars)
        log.info("Finished A/E correction")
        df = get_classifier(df, cal_energy_param, mu_pars, sigma_pars)
    except:
        log.error("A/E calibration failed")
        mu_pars = np.full(2, np.nan)
        sigma_pars = np.full(2, np.nan)
        results_dict = {}


    
    try:
        cut = get_aoe_cut_fit(df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                            df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                            1592, (40, 20), 0.9, eres_pars, display=0)
        log.info(f"Cut found at {cut}")
    except:
        log.error("A/E cut determination failed")
        cut = np.nan

    aoe_cal_dict = {
            "AoE_Corrected": {
                "expression": f"(((AoE_Timecorr)/(a*{cal_energy_param} +b))-1)",
                "parameters": {"a": mu_pars[0], "b": mu_pars[1]},
            },
            "AoE_Classifier": {
                "expression": f"AoE_Corrected/(sqrt(c+(d/{cal_energy_param})**2)/(a*{cal_energy_param} +b))",
                "parameters": {
                    "a": mu_pars[0],
                    "b": mu_pars[1],
                    "c": sigma_pars[0],
                    "d": sigma_pars[1],
                },
            },
            "AoE_Low_Cut": {
                "expression": "AoE_Classifier>a",
                "parameters": {"a": cut},
            },
            "AoE_Double_Sided_Cut": {
                "expression": "(b>AoE_Classifier)&(AoE_Classifier>a)",
                "parameters": {"a": cut, "b": aoe_high_cut},
            }

        }
    
    if re.match("(\d{8})T(\d{6})Z", list(cal_dict)[0]):
        for tstamp in cal_dict:
            cal_dict[tstamp].update(aoe_cal_dict)
    else:
        cal_dict.update(aoe_cal_dict)

    try:
        log.info("  Compute low side survival fractions: ")

        peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        sf = np.zeros(len(peaks_of_interest))
        sferr = np.zeros(len(peaks_of_interest))
        fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]
        full_sfs = []
        full_sf_errs = []
        full_cut_vals = []

        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                cut_vals, sfs, sf_errs, sf[i], sferr[i] = compton_sf(
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                    df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                    cut, peak, eres_pars
                )

                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(sf_errs)
            else:
                cut_vals, sfs, sf_errs, sf[i], sferr[i] = get_sf(
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                    df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                    peak, fit_widths[i], cut, eres_pars
                )
                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(sf_errs)

            log.info(f"{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %")

        sf_2side = np.zeros(len(peaks_of_interest))
        sferr_2side = np.zeros(len(peaks_of_interest))
        log.info("Calculating 2 sided cut sfs")
        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                sf_2side[i], sferr_2side[i] = compton_sf_no_sweep(
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                    df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                    peak,
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                )
            else:
                sf_2side[i], sferr_2side[i] = get_sf_no_sweep(
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                    df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                    peak,
                    fit_widths[i],
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                )

            log.info(f"{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %")

        def convert_sfs_to_dict(peaks_of_interest, sfs, sf_errs):
            out_dict = {}
            for i, peak in enumerate(peaks_of_interest):
                out_dict[str(peak)] = {
                    "sf": f"{sfs[i]:2f}",
                    "sf_err": f"{sf_errs[i]:2f}",
                }
            return out_dict

        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": energy_param,
            "Cal_energy_param": cal_energy_param,
            "dt_param": "dt_eff",
            "rt_correction": dt_corr,
            "1000-1300keV": res_dict,
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
            "Low_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf, sferr),
            "2_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf_2side, sferr_2side),
        }
        log.info("Done")
        log.info(f"Results are {out_dict}")

        if display > 0:
            plot_dict = {}

            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 16

            fig1 = plt.figure()
            plt.subplot(3, 2, 1)
            plot_dt_dep(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                        df.query("is_valid_cal& is_not_pulser")["dt_eff"],  
                        [1582, 1602], f"Tl DEP")
            plt.subplot(3, 2, 2)
            plot_dt_dep(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                        df.query("is_valid_cal& is_not_pulser")["dt_eff"],  
                        [1510, 1630], f"Bi FEP")
            plt.subplot(3, 2, 3)
            plot_dt_dep(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                        df.query("is_valid_cal& is_not_pulser")["dt_eff"],   
                        [2030, 2050], "Qbb")
            plt.subplot(3, 2, 4)
            plot_dt_dep(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                        df.query("is_valid_cal& is_not_pulser")["dt_eff"],   
                        [2080, 2120], f"Tl SEP")
            plt.subplot(3, 2, 5)
            plot_dt_dep(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                        df.query("is_valid_cal& is_not_pulser")["dt_eff"],   
                        [2584, 2638], f"Tl FEP")
            plt.tight_layout()
            plot_dict["dt_deps"] = fig1
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig2 = plt.figure()
            plot_compt_bands_overlayed(df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"], 
                                        df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                                        [950, 1250, 1460, 1660, 1860, 2060])
            plt.ylabel("Counts")
            plt.xlabel("Raw A/E")
            plt.title(f"Compton Bands before Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_nocorr"] = fig2
            if display > 1:
                plt.show()
            else:
                plt.close()

            # if dt_corr == True:
            #     _, plot_dict = drift_time_correction(
            #         aoe_uncorr, energy, dt, display=display, plot_dict=plot_dict
            #     )

            _,_ , _,plot_dict= aoe_timecorr(df, energy_param, plot_dict=plot_dict, display=display)

            _,_,_, plot_dict = AoEcorrection(
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param], 
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                eres_pars, plot_dict=plot_dict, display=display
            )

            fig3 = plt.figure()
            plot_compt_bands_overlayed(df.query("is_valid_cal& is_not_pulser")["AoE_classifier"], 
                                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],  
                                    [950, 1250, 1460, 1660, 1860, 2060], [-5, 5]
            )
            plt.ylabel("Counts")
            plt.xlabel("Corrected A/E")
            plt.title(f"Compton Bands after Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_corr"] = fig3
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig4 = plt.figure()
            plt.vlines(cut, 0, 100, label=f"Cut Value: {cut:1.2f}", color="black")

            for i, peak in enumerate(peaks_of_interest):
                plt.errorbar(
                    full_cut_vals[i],
                    full_sfs[i],
                    yerr=full_sf_errs[i],
                    label=f"{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %",
                )

            handles, labels = plt.gca().get_legend_handles_labels()
            #order = [1, 2, 3, 0, 4, 5]
            plt.legend(
                # [handles[idx] for idx in order],
                # [labels[idx] for idx in order],
                loc="upper right",
            )
            plt.xlabel("Cut Value")
            plt.ylabel("Survival Fraction %")
            plot_dict["surv_fracs"] = fig4
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig5, ax = plt.subplots()
            bins = np.linspace(1000, 3000, 2000)
            ax.hist(df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param], bins=bins, histtype="step", label="Before PSD")
            ax.hist(
                df.query(f"is_valid_cal& is_not_pulser & AoE_classifier > {cut}")[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Low side PSD cut",
            )
            ax.hist(
                df.query(f"is_valid_cal& is_not_pulser & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}")[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Double sided PSD cut",
            )
            ax.hist(
                df.query(f"is_valid_cal& is_not_pulser & (AoE_classifier < {cut} | AoE_classifier > {aoe_high_cut})")[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Rejected by PSD cut",
            )

            axins = ax.inset_axes([0.25, 0.07, 0.4, 0.3])
            bins = np.linspace(1580, 1640, 200)
            axins.hist(df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param], 
                        bins=bins, histtype="step")
            axins.hist(df.query(f"is_valid_cal& is_not_pulser & AoE_classifier > {cut}")[cal_energy_param],
                            bins=bins, histtype="step")
            axins.hist(
                df.query(f"is_valid_cal& is_not_pulser & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}")[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                df.query(f"is_valid_cal& is_not_pulser & (AoE_classifier < {cut} | AoE_classifier > {aoe_high_cut})")[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            ax.set_xlim([1000, 3000])
            ax.set_yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper left")
            plot_dict["PSD_spectrum"] = fig5
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig6 = plt.figure()
            bins = np.linspace(1000, 3000, 1000)
            counts_pass, bins_pass, _ = pgh.get_hist(
                df.query(f"is_valid_cal& is_not_pulser & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}")[cal_energy_param], bins=bins
            )
            counts, bins, _ = pgh.get_hist(df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param], bins=bins)
            survival_fracs = counts_pass / (counts)

            plt.step(pgh.get_bin_centers(bins_pass), survival_fracs)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Survival Fraction")
            plt.ylim([0, 1])
            plot_dict["psd_sf"] = fig6
            if display > 1:
                plt.show()
            else:
                plt.close()

            return cal_dict, out_dict, plot_dict
        else:
            return cal_dict, out_dict

    except:
        log.error("survival fraction determination failed")
        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": energy_param,
            "Cal_energy_param": cal_energy_param,
            "dt_param": "dt_eff",
            "rt_correction": False,
            "1000-1300keV_mean": res_dict,
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
        }
        if display > 0:
            plot_dict = {}
            return cal_dict, out_dict, plot_dict
        else:
            return cal_dict, out_dict
