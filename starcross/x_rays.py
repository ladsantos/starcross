#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module estimates the X-rays and extreme-ultraviolet luminosity of a star.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import astropy.units as u


__all__ = ["x_rays_luminosity"]


# Calculates the X-rays luminosity
def x_rays_luminosity(b_v_color, age, age_uncertainty, n_sample=10000):
    """

    Parameters
    ----------
    b_v_color
    age
    age_uncertainty
    n_sample

    Returns
    -------

    """
    # First let's convert the unit of age to year and create a random sample
    age = age.to(u.yr).value
    age_sigma = age_uncertainty.to(u.yr).value
    ages = np.random.normal(loc=age, scale=age_sigma, size=n_sample)

    # Hard-coding the several cases of Jackson+2012
    params = {'case': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
              'b_v_0': [0.290, 0.450, 0.565, 0.675, 0.790, 0.935, 1.275],
              'b_v_1': [0.450, 0.565, 0.675, 0.790, 0.935, 1.275, 1.410],
              'saturation_log_age': [7.87, 8.35, 7.84, 8.03, 7.90, 8.28, 8.21],
              'sat_log_age_unc': [0.10, 0.05, 0.06, 0.06, 0.05, 0.07, 0.04],
              'saturation_lum_x': [-4.28, -4.24, -3.67, -3.71, -3.36, -3.35,
                                   -3.14],
              'sat_lum_x_unc': [0.05, 0.02, 0.01, 0.05, 0.02, 0.01, 0.02],
              'alpha': [1.22, 1.24, 1.13, 1.28, 1.40, 1.09, 1.18],
              'alpha_unc': [0.30, 0.19, 0.13, 0.17, 0.11, 0.28, 0.31]
              }

    # First we identify the case
    cases = params['case']
    ind = None
    for i in range(len(cases)):
        if params['b_v_0'][i] < b_v_color < params['b_v_1'][i]:
            ind = i
            break
        else:
            pass

    # The saturated case
    if np.log10(age) <= params['saturation_log_age'][ind]:
        lumx_lumbol_mu = 10 ** params['saturation_lum_x'][ind]
        lumx_lumbol_sigma = 10 ** params['sat_lum_x_unc'][ind]
        lumx_lumbol_sample = np.random.normal(loc=lumx_lumbol_mu,
                                              scale=lumx_lumbol_sigma,
                                              size=n_sample)
    else:
        alpha = params['alpha'][ind]
        alpha_sigma = params['alpha_unc'][ind]
        alpha_sample = np.random.normal(loc=alpha, scale=alpha_sigma,
                                        size=n_sample)
        sat_age = 10 ** params['saturation_log_age'][ind]
        sat_age_sigma = 10 ** params['sat_log_age_unc'][ind]
        sat_age_sample = np.random.normal(loc=sat_age, scale=sat_age_sigma,
                                          size=n_sample)
        sat_lumx_lumbol = 10 ** params['saturation_lum_x'][ind]
        sat_lumx_lumbol_sigma = 10 ** params['sat_lum_x_unc'][ind]
        sat_lumx_lumbol_sample = np.random.normal(loc=sat_lumx_lumbol,
                                                  scale=sat_lumx_lumbol_sigma,
                                                  size=n_sample)
        lumx_lumbol_sample = sat_lumx_lumbol_sample * (ages / sat_age_sample) \
            ** (-alpha_sample)

    # Finally calculate lumx_lumbol
    percentiles = np.percentile(lumx_lumbol_sample, [16, 50, 84])
    q = np.diff(percentiles)
    lumx_lumbol = percentiles[1]
    lumx_lumbol_sigma_up = q[1]
    lumx_lumbol_sigma_low = q[0]

    return lumx_lumbol, lumx_lumbol_sigma_up, lumx_lumbol_sigma_low
