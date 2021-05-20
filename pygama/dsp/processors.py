"""
Contains a list of dsp processors used by the legend experiment, implemented
using numba's guvectorize to implement numpy's ufunc interface. In other words,
all of the functions are void functions whose outputs are given as parameters.
The ufunc interface provides additional information about the function
signatures that enables broadcasting the arrays and SIMD processing. Thanks to
the ufunc interface, they can also be called to return a numpy array, but if
 this is done, memory will be allocated for this array, slowing things down.
"""

# I think there's a way to do this recursively, but I'll figure it out later...
from ._processors.old_mean_stdev import mean_stdev
from ._processors.old_trap_filter import old_trap_filter
from ._processors.old_current import old_avg_current
from ._processors.old_asym_trap_filter import old_asymTrapFilter
from ._processors.old_fixed_time_pickoff import old_fixed_time_pickoff
from ._processors.old_trap_norm import old_trap_norm
from ._processors.old_trap_pickoff import old_trap_pickoff
from ._processors.old_time_point_frac import time_point_frac
from ._processors.old_time_point_thresh import old_time_point_thresh
from ._processors.old_linear_fit import linear_fit
from ._processors.old_zac_filter import old_zac_filter
from ._processors.param_lookup import param_lookup
from ._processors.old_cusp_filter import old_cusp_filter
from ._processors.fftw import dft, inv_dft, psd
from ._processors.linear_slope_fit import linear_slope_fit
from ._processors.log_check import log_check
from ._processors.old_min_max import old_min_max
from ._processors.presum import presum
from ._processors.old_find_tp100 import old_find_tp100
from ._processors.old_t0_filter import old_t0_filter_inv
from ._processors.bl_subtract import bl_subtract
from ._processors.convolutions import cusp_filter, zac_filter, t0_filter
from ._processors.old_t0_filter import old_t0_filter_inv
from ._processors.old_bl_subtract import old_bl_subtract
from ._processors.old_log_check import old_log_check
from ._processors.old_pole_zero import old_pole_zero, old_double_pole_zero
from ._processors.pole_zero import pole_zero, double_pole_zero
from ._processors.moving_windows import moving_window_left, moving_window_right, moving_window_multi, avg_current
from ._processors.old_moving_window import old_moving_window_left, old_moving_window_right, old_moving_window_multi
from ._processors.fixed_time_pickoff import fixed_time_pickoff
from ._processors.trap_filters import trap_filter, trap_norm, asym_trap_filter, trap_pickoff
from ._processors.min_max import min_max
from ._processors.time_point_thresh import time_point_thresh
from ._processors.old_linear_slope_fit import old_linear_slope_fit
from ._processors.presum import presum
from ._processors.windower import windower
from ._processors.upsampler import upsampler
