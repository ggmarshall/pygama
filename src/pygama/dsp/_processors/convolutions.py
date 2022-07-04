import numpy as np
from numba import guvectorize

from pygama.dsp.errors import DSPFatal


def cusp_filter(length, sigma, flat, decay):
    """
    Apply a CUSP filter to the waveform.  Note that it is composed of a
    factory function that is called using the init_args argument and that
    the function the waveforms are passed to using args.

    Parameters
    ----------
    length : int
        The length of the filter to be convolved
    sigma : float
        The curvature of the rising and falling part of the kernel
    flat : int
        The length of the flat section
    decay : int
        The decay constant of the exponential to be convolved

    Examples
    --------
    .. code-block :: json

        "wf_cusp": {
            "function": "cusp_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "wf_cusp(101,f)"],
            "unit": "ADC",
            "prereqs": ["wf_bl"],
            "init_args": ["len(wf_bl)-100", "40*us", "3*us", "45*us"]
        }
    """
    if length <= 0:
        raise DSPFatal('The length of the filter must be positive')

    if sigma < 0:
        raise DSPFatal('The curvature parameter must be positive')

    if flat < 0:
        raise DSPFatal('The length of the flat section must be positive')

    if decay < 0:
        raise DSPFatal('The decay constant must be positive')

    lt       = int((length - flat) / 2)
    flat_int = int(flat)
    cusp     = np.zeros(length)
    for ind in range(0, lt, 1):
        cusp[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
    for ind in range(lt, lt + flat_int + 1, 1):
        cusp[ind] = 1
    for ind in range(lt + flat_int + 1, length, 1):
        cusp[ind] = float(np.sinh((length - ind) / sigma) / np.sinh(lt / sigma))

    den   = [1, -np.exp(-1 / decay)]
    cuspd = np.convolve(cusp, den, 'same')

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def cusp_out(w_in, w_out):
        """
        Parameters
        ----------
        w_in : array-like
            The input waveform
        w_out : array-like
            The filtered waveform
        """
        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(cuspd) > len(w_in):
            raise DSPFatal('The filter is longer than the input waveform')

        w_out[:] = np.convolve(w_in, cuspd, 'valid')

    return cusp_out

def zac_filter(length, sigma, flat, decay):
    """
    Apply a ZAC (Zero Area CUSP) filter to the waveform.  Note that it is
    composed of a factory function that is called using the init_args
    argument and that the function the waveforms are passed to using args.

    Parameters
    ----------
    length : int
        The length of the filter to be convolved
    sigma : float
        The curvature of the rising and falling part of the kernel
    flat : int
        The length of the flat section
    decay : int
        The decay constant of the exponential to be convolved

    Examples
    --------
    .. code-block :: json

        "wf_zac": {
            "function": "zac_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_bl", "wf_zac(101,f)"],
            "unit": "ADC",
            "prereqs": ["wf_bl"],
            "init_args": ["len(wf_bl)-100", "40*us", "3*us", "45*us"],
        }
    """
    if length <= 0:
        raise DSPFatal('The length of the filter must be positive')

    if sigma < 0:
        raise DSPFatal('The curvature parameter must be positive')

    if flat < 0:
        raise DSPFatal('The length of the flat section must be positive')

    if decay < 0:
        raise DSPFatal('The decay constant must be positive')

    lt       = int((length - flat) / 2)
    flat_int = int(flat)

    # calculate cusp filter and negative parables
    cusp = np.zeros(length)
    par  = np.zeros(length)
    for ind in range(0, lt, 1):
        cusp[ind] = float(np.sinh(ind / sigma) / np.sinh(lt / sigma))
        par [ind] = np.power(ind - lt / 2, 2) - np.power(lt / 2, 2)
    for ind in range(lt, lt + flat_int + 1, 1):
        cusp[ind] = 1
    for ind in range(lt + flat_int + 1, length, 1):
        cusp[ind] = float(np.sinh((length - ind) / sigma) / np.sinh(lt / sigma))
        par [ind] = np.power(length - ind - lt / 2, 2) - np.power(lt / 2, 2)

    # calculate area of cusp and parables
    areapar, areacusp = 0, 0
    for i in range(0, length, 1):
        areapar  += par [i]
        areacusp += cusp[i]

    # normalize parables area
    par = -par / areapar * areacusp

    # create zac filter
    zac = cusp + par

    # deconvolve zac filter
    den  = [1, -np.exp(-1 / decay)]
    zacd = np.convolve(zac, den, 'same')

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def zac_out(w_in, w_out):
        """
        Parameters
        ----------
        w_in : array-like
            The input waveform
        w_out : array-like
            The filtered waveform
        """
        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(zacd) > len(w_in):
            raise DSPFatal('The filter is longer than the input waveform')

        w_out[:] = np.convolve(w_in, zacd, 'valid')

    return zac_out

def t0_filter(rise, fall):

    """
    Apply a modified, asymmetric trapezoidal filter to the waveform.  Note
    that it is composed of a factory function that is called using the init_args
    argument and that the function the waveforms are passed to using args.

    Parameters
    ----------
    rise : int
        The length of the rise section.  This is the linearly increasing
        section of the filter that performs a weighted average.
    fall : int
        The length of the fall section.  This is the simple averaging part
        of the filter.

    Examples
    --------
    .. code-block :: json

        "wf_t0_filter": {
            "function": "t0_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "wf_t0_filter(3748,f)"],
            "unit": "ADC",
            "prereqs": ["wf_pz"],
            "init_args": ["128*ns", "2*us"]
        }
    """
    if rise < 0:
        raise DSPFatal('The length of the rise section must be positive')

    if fall < 0:
        raise DSPFatal('The length of the fall section must be positive')

    t0_kern = np.zeros(int(rise)+int(fall))
    for i in range(int(rise)):
        t0_kern[i] = 2*(int(rise)-i)/ (rise**2)
    for i in range(int(rise), len(t0_kern),1):
        t0_kern[i] =  - 1 / fall

    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def t0_filter_out(w_in, w_out):
        """
        Parameters
        ----------
        w_in : array-like
            The input waveform
        w_out : array-like
            The filtered waveform
        """
        w_out[:] = np.nan

        if np.isnan(w_in).any():
            return

        if len(t0_kern) > len(w_in):
            raise DSPFatal('The filter is longer than the input waveform')

        w_out[:] = np.convolve(w_in, t0_kern)[:len(w_in)]

    return t0_filter_out
