


@guvectorize(["void(float32[:], float32, float32, float32[:])",
              "void(float64[:], float32, float32, float64[:])"],
              "(n),(),()->(n)", forceobj=True)

def t0_filter_rec(w_in,rise,fall, w_out):

    """
    This processor applies modified assymetric trap filter to the waveform for use in t0 estimation. 

    Parameters
    ----------

    w_in : array-like
            waveform to be convolved with t0 filter
    
    rise : float
            Specifies length of rise section. This is the linearly increasing section of the filter that performs a weighted average.
    
    fall : float
            Specifies length of fall section. This is the simple averaging part of the filter.

    w_out : array-like
            waveform convolved with t0 filter
    """

    w_out[:] = np.nan


    if (np.isnan(w_in).any()):
        return

    if (not rise >= 0):
        raise DSPFatal('rise out of range')
    if (not fall >= 0):
        raise DSPFatal('fall out of range')

    if (not (rise+fall) in range(len(w_in))):
        raise DSPFatal('Filter longer than input waveform')

    rsum = w_in[0]
    irise = int(rise)
    ifall = int(fall)
    w_out[0] = 2*w_in[0]/rise
    for i in range(1,irise):
        w_out[i] = w_out[i-1] - 2*rsum/(rise**2) + 2*w_in[i]/rise
        rsum += w_in[i]
    for i in range(irise, irise +ifall):
        w_out[i] = w_out[i-1] - 2*rsum/(rise**2) + 2*w_in[i]/rise - w_in[i-irise]/fall
        rsum += (w_in[i]-w_in[i-irise])
    for i in range(irise +ifall, len(w_in)):
        w_out[i] = w_out[i-1] - 2*rsum/(rise**2) + 2*w_in[i]/rise - (w_in[i-irise]- w_in[i-irise-ifall])/fall
        rsum += (w_in[i]-w_in[i-irise])