import numpy as np
from numba import guvectorize

@guvectorize(["void(float32, float32, float32, float32[:])",
              "void(float64, float64, float64, float64[:])"],
             "(),(),()->()", nopython=True, cache=True)



def charge_trapping_correction(e_in, t_in, alpha, e_out):
    """
    Calculates the charge trapping corrected energy using a drift time parameter and the constant alpha.

    e_in : float
            input energy
    t_in : float
            input drift time
    alpha: float
            charge trapping correction constant
    e_out: float
            charge trapping corrected energy
    
    """

    e_out[0] = np.nan
    if (np.isnan(alpha)):
        return
    e_out[0] = (e_in * (t_in * alpha)) + e_in