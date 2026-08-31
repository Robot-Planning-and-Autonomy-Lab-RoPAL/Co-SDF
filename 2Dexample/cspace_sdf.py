#!/usr/bin/env python3
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

def _drop_periodic_duplicates(theta, grid):
	
	t1, t2 = theta; g = grid
	if np.isclose((t1[-1]-t1[0]) % (2*np.pi), 0.0) and len(t1) > 1:
		t1 = t1[:-1]; g = g[:-1, :]
	if np.isclose((t2[-1]-t2[0]) % (2*np.pi), 0.0) and len(t2) > 1:
		t2 = t2[:-1]; g = g[:, :-1]
	return (t1, t2), g

def cobs_sdf_on_torus(theta, c_space):

	(t1, t2), obs0 = _drop_periodic_duplicates(theta, c_space)
	obs = (obs0 > 0)
	dth1 = float(t1[1]-t1[0]); dth2 = float(t2[1]-t2[0])
	H, W  = obs.shape
	
	obs_t   = np.tile(obs, (3, 3))
	d_out_t = ndi.distance_transform_edt(~obs_t, sampling=(dth1, dth2))
	d_in_t  = ndi.distance_transform_edt( obs_t, sampling=(dth1, dth2))
	sdf_t   = d_out_t - d_in_t
	
	sdf = sdf_t[H:2*H, W:2*W]
	
	return (t1, t2), sdf

def wrap_samples_to_period(samples, theta):

	t1, t2 = theta
	out = samples.copy()
	out[:,0] = (out[:,0]-t1[0]) % (t1[-1]-t1[0]) + t1[0]
	out[:,1] = (out[:,1]-t2[0]) % (t2[-1]-t2[0]) + t2[0]
	return out

def cspace_sdf_at_samples(theta, sdf_grid, samples_wrapped):

	t1, t2 = theta
	interp = RegularGridInterpolator((t1, t2), sdf_grid,
									bounds_error=False, fill_value=None)
	return interp(samples_wrapped)
