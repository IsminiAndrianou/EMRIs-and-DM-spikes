# imports
import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import mpmath
import glob
from scipy.integrate import quad, DOP853

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import *
from few.utils.constants import *
from few.utils.baseclasses import TrajectoryBase, SchwarzschildEccentric
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum

from modified_HaloFeedback import *
from datetime import datetime
import re

# additional imports for the waveforms
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap,
                               get_mismatch,
                               get_fundamental_frequencies,
                               get_separatrix,
                               get_mu_at_t,
                               get_p_at_t,
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
from few.utils import *
from few.utils.fdutils import *

# additional imports
from few.utils.baseclasses import TrajectoryBase
from few.utils.baseclasses import SchwarzschildEccentric

from few.fastinterp import *

from mpmath import *
from scipy.integrate import DOP853
import scipy

import time
import warnings
from abc import ABC, abstractmethod
from time import time as timeit

import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
from scipy.special import ellipeinc, ellipkinc, ellipe, ellipk, betainc
from scipy.special import gamma as Gamma
from scipy.special import beta as Beta

# for common interface with C/mathematica
def Power(x, n):
    return x**n

def Sqrt(x):
    return np.sqrt(x)


### GW energy and angular momentum losses (FEW units [dim/less])
def Edot_GW(p,e):
    # Azimuthal frequency
    Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

    # Post-Newtonian calculations
    yPN = pow(Omega_phi,2./3.)

    EdotPN = (96 + 292*Power(e,2) + 37*Power(e,4))/(15.*Power(1 - Power(e,2),3.5)) * pow(yPN, 5)
    LdotPN = (4*(8 + 7*Power(e,2)))/(5.*Power(-1 + Power(e,2),2)) * pow(yPN, 7./2.)
    return EdotPN, LdotPN


# unit conversions for plot labels
GeV_to_kg = 1.78266192e-27 # multiply with to convert GeV to kg --C
cm_3_to_pc_3 = 1/((100*PC_SI)**3) # multiply with to convert cm^3 to pc^3

GeV_cm3_to_Msun_pc3 = (GeV_to_kg / MSUN_SI) / cm_3_to_pc_3 # multiply with to convert GeV/cm^3 to Msun/pc^3
g_cm3_to_Msun_pc3 = MSUN_SI * 1e3 * cm_3_to_pc_3 # divide by to convert g/cm^3 to Msun/pc^3

def sci_notation(num, decimal_digits=0):
    """Format number into LaTeX-style scientific notation."""
    from math import floor, log10
    if num == 0:
        return "0"
    exponent = int(floor(log10(abs(num))))
    coeff = round(num / 10**exponent, decimal_digits)
    if coeff == 1:
        return fr"10^{{{exponent}}}"
    else:
        return fr"{coeff}\times 10^{{{exponent}}}"

### Some useful functions for spike parameters in HaloFeedback units

def get_r_sp_Eda(M, gamma=1, r_s = 2e4, D = 8.2e3, rho_D_input = 0.4, NFW=False):
        """ Get the spike radius in [pc] as defined by Eda et al. in arXiv: 1408.3534, 1301.5971, $r_{sp}^{Eda}$, in [pc]
        Parameters:
            - M: MBH mass, in [Msun]
            - gamma: cuspy halo power-law index
            - r_s: cuspy halo normalization radius in [pc]
            - D: distance from the sun to the galactic center in [pc]
            - rho_D_input: local DM density in [GeV/cm^3]

        """
        rho_D = rho_D_input * GeV_cm3_to_Msun_pc3 # local DM density in [Msun/pc^3]
        rho_s = rho_D * (D/r_s)**gamma # in [Msun/pc^3]
        if NFW:
                rho_s = rho_D * (D/r_s) * (1+D/r_s)**2 # in [Msun/pc^3]
        r_h = np.sqrt((M)/(np.pi * rho_s * r_s)) # radius determining the sphere of influence of the MBH in [pc]
        r_sp_Eda = 0.2 * r_h # spike radius from Eda in [pc]
        return r_sp_Eda

def get_rho_sp(M, gamma=1, r_s = 2e4, D = 8.2e3, rho_D_input = 0.4, NFW=False):
        """Get the normalizing spike density, $\rho_{sp}$, in [Msun/pc^3].
        Based on the methodology of Sec. 4.1, see also DM_profiles_plot.ipynb 
    
        Parameters:
                - M: MBH mass, in [Msun]
                - gamma: cuspy halo power-law index
                - r_s: cuspy halo normalization radius in [pc]
                - D: distance from the sun to the galactic center in [pc]
                - rho_D_input: local DM density in [GeV/cm^3]

        """
        rho_D = rho_D_input * GeV_cm3_to_Msun_pc3 # local DM density in [Msun/pc^3]
        rho_s = rho_D * (D/r_s)**gamma # in [Msun/pc^3]
        if NFW:
                rho_s = rho_D * (D/r_s) * (1+D/r_s)**2 # in [Msun/pc^3]
        r_h = np.sqrt((M)/(np.pi * rho_s * r_s)) # radius determining the sphere of influence of the MBH in [pc]
        r_sp_Eda = 0.2 * r_h # spike radius from Eda in [pc]
        rho_sp = rho_s * (r_s/r_sp_Eda)**gamma # spike normalization density in [Msun/pc^3]
        return rho_sp

def get_r_sp(M, gamma_sp, rho_sp):
        """Get the normalizing spike radius, $r_{sp}$ in [pc].
        Based on the methodology of Sec. 4.1, see also DM_profiles_plot.ipynb

        Parameters:
                - M: MBH mass, in [Msun]
                - gamma_sp: cuspy halo power-law index
                - rho_sp: spike normalization density in [Msun/pc^3]
        """
        r_sp = (((3-gamma_sp)* 0.2**(3-gamma_sp) * M)/(2 * np.pi * rho_sp))**(1/3) # pc
        return r_sp


def get_rho_sp_from_rho_6(M, gamma_sp, rho_6):
        """ Spike normalization, $\rho_{sp}$, in [Msun/pc^3]
        Parameters:
            - M: MBH mass, in [Msun]
            - gamma_sp: spike index
            - rho_6: alternative spike normalization density in [Msun/pc^3]
        """
        r_6 = 1e-6 # in [pc]
        rho_sp = ((rho_6 * r_6**gamma_sp)**(1/(1-gamma_sp/3))) / ((((3-gamma_sp) * 0.2**(3-gamma_sp) * M)/(2 * np.pi))**((gamma_sp/3)/(1-gamma_sp/3))) # rho_sp from rho_6 in [Msun/pc^3]
        return rho_sp

def get_rho_init(M, rho_sp, p, gamma_sp):
        """ Initial spike density at a given radius $\rho(r)$, in [Msun/pc^3]
        Parameters:
                - M: MBH mass in [Msun]
                - rho_sp: spike normalization density in [Msun/pc^3]
                - p: semi-latus rectum in FEW units [R_S/2]
                - gamma_sp: spike index
        """
        p_pc = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))/PC_SI
        r_sp = get_r_sp(M, gamma_sp, rho_sp)
        #rho_init = rho_sp * (r_sp/p_pc)**gamma_sp
        rho_init = rho_sp * (r_sp/p_pc)**gamma_sp
        return rho_init

def get_rho_6_from_rho_sp(M, rho_sp, gamma_sp):
        """Spike normalization, $\rho_6$, in [Msun/pc^3]
        Parameters:
            - M: MBH mass, in [Msun]
            - gamma_sp: spike index
            - rho_sp: spike normalization density in [Msun/pc^3]
        """
        r_6 = 1e-6 # in [pc]
        rho_6 = rho_sp**(1-(gamma_sp/3)) * (((3-gamma_sp) * 0.2**(3-gamma_sp) * M) / (2*np.pi))**(gamma_sp/3) * r_6**(-gamma_sp)
        return rho_6

def get_rho_eff(M, mu, rho_sp, p, gamma_sp):
    """ Effective density profile at a given radius $\rho(r)$, in [Msun/pc^3]. 
        Only valid for r > r_b, where r_b is the break point corresponding to the break frequency. 
        Based on Coogan's (pydd) paper 2108.04154
        Parameters:
                - M: MBH mass in [Msun]
                - mu: CO mass in [Msun]
                - rho_sp: spike normalization density in [Msun/pc^3]
                - p: semi-latus rectum in [pc]
                - gamma_sp: spike index [dim/less]
        """
    p_m = p * PC_SI
    gamma_e = 5/2
    r_sp = get_r_sp(M, gamma_sp, rho_sp)

    coeff = (64 * G_SI**gamma_e * ((M + mu) * MSUN_SI)**(3/2) * M**2 * MSUN_SI) / (5 * mu * C_SI**5 * p_m**gamma_e * np.log(Sqrt(M/mu)))

    rho_eff = coeff * rho_sp * (r_sp/p)**(gamma_sp)
    return rho_eff

def get_break_frequency(M, mu, gamma_sp):
    """Break frequency as given by the analytical fit from 2108.01454, in [Hz]
    Parameters:
            - M: MBH mass in [Msun]
            - mu: CO mass in [Msun]
            - gamma_sp: spike index [dim/less]
    """
    beta = 0.8163 # [Hz]
    alpha1 = 1.4412
    alpha2 = 0.4511
    zeta = -0.4971
    gamma_r = 1.4396
    break_freq = beta * (M/1E3)**(-alpha1) * mu**alpha2 * (1 + zeta * np.log(gamma_sp/gamma_r))
    return (break_freq)

def get_break_point(M, mu, gamma_sp):
    """Break point corresponding to break frequency fit from 2108.01454, in [pc]
            - M: MBH mass in [Msun]
            - mu: CO mass in [Msun]
            - gamma_sp: spike index [dim/less]
    """
    break_freq = get_break_frequency(M, mu, gamma_sp)
    break_p = ((G_SI * (M + mu) * MSUN_SI) / (np.pi**2 * break_freq**2))**(1/3)
    return (break_p / PC_SI)

def get_frequency(M, mu, p):
    """ Frequency corresponding to binary separation [Hz]. Only for circular orbits.
            - M: MBH mass in [Msun]
            - mu: CO mass in [Msun]
            - p: semi-latus rectum (i.e. separation for circular orbits) in FEW units [R_S/2]
    """
    p_m = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))
    f = (1/np.pi) * Sqrt(G_SI * (M + mu) * MSUN_SI / (p_m**3)) # in [Hz]
    return f

def get_separation(M, mu, f):
    """ Separation corresponding to binary orbital frequency [pc]. Only for circular orbits.
            - M: MBH mass [Msun]
            - mu: CO mass [Msun]
            - f: orbital frequency in [Hz]
    """
    p_m = ((G_SI * (M + mu) * MSUN_SI) / (np.pi**2 * f**2))**(1/3)
    return p_m/PC_SI

######## the energy losses need to be checked - error: dyn above stat
######## also check how they are used compared to the GW ones and then check evolution
### Dynamical Friction Energy & Angular Momentum Losses

# The separation of the relativistic or not DF losses seems unecessary, 
# but significantly reduces the computational cost and runtime of the modified few trajectories.

def dEdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, relativistic=False, eff_dens=None):
    """Rate of change of energy to DF in FEW units [dim/less] below the break frequency.
    Use to find the orbit-averaged DF energy losses, i.e. from one set of (p,e) to the next.

    Parameters:
        - p: semi-latus rectum [pc]
        - e: eccentricity [dim/less]
        - gamma_sp: DM spike index [dim/less]
        - rho_sp: DM spike normalizing density [Msun/pc^3]
        - relativistic: bool, if True use the relativistic correction
        - eff_dens: optional, use the effective DM density
    """
    pc_to_km = 3.08567758149137e13
    G_N = 4.3021937e-3 # (km/s)^2 pc M_sun^-1
    c = 2.99792458e5  # km/s
    p_pc = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))/PC_SI 
    epsilon = mu/M 
    
    r_sp = get_r_sp(M, gamma_sp, rho_sp)
    Coulomb_log = np.log(Sqrt(M/mu))

    if eff_dens is not None:
        dEdt = (1 / pc_to_km) * (2 * (1 - e**2)**(3/2) * G_N**(3/2) * mu**2 * eff_dens * Coulomb_log * p_pc**(1/2)) / Sqrt(M)
    else:
        dEdt = (1 / pc_to_km) * (2 * (1 - e**2)**(3/2) * G_N**(3/2) * mu**2 * rho_sp * r_sp**(gamma_sp) * Coulomb_log) / (Sqrt(M) * p_pc**(gamma_sp - 1/2))

    def integrand_E(theta):
        if eff_dens is not None:
            return 1 / ((1 + e * np.cos(theta))**2 * np.sqrt(1 + 2 * e * np.cos(theta) + e**2))
        return (1 + e * np.cos(theta))**(gamma_sp-2) / np.sqrt(1 + 2 * e * np.cos(theta) + e**2)
    
    result_E, error_E = quad(integrand_E, 0.0, 2*np.pi)

    E_dot_DF = (dEdt * result_E * 10**6 * MSUN_SI * G_SI / Power(C_SI, 5)) / epsilon**2 # confirmed transform from HF to FEW units

    if relativistic:
        def integrand_E(theta):
            xi = (1 + ((G_N * M)*(1 + 2 * e * np.cos(theta) + e**2) / (c**2 * p)) 
                - ((G_N * M)**2 * (1 + 2 * e * np.cos(theta) + e**2)**2 / (c**4 * p**2))
                - ((G_N * M)**3 * (1 + 2 * e * np.cos(theta) + e**2)**3 / (c**6 * p**3)))
            if eff_dens is not None:
                return (1 / ((1 + e * np.cos(theta))**2 * np.sqrt(1 + 2 * e * np.cos(theta) + e**2))) * xi
            return ((1 + e * np.cos(theta))**(gamma_sp-2) / np.sqrt(1 + 2 * e * np.cos(theta) + e**2)) * xi
        
        result_E_rel, error_E_rel = quad(integrand_E, 0.0, 2*np.pi)
        E_dot_DF_rel = (dEdt * result_E_rel * 10**6 * MSUN_SI * G_SI / Power(C_SI, 5)) / epsilon**2 # confirmed transform from HF [(km/s)^2 s^-1 M_sun] to FEW units
        return (E_dot_DF_rel)

    return E_dot_DF


def dEdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, relativistic=True, eff_dens=None):
    """Rate of change of energy to DF in FEW units [dim/less] below the break frequency.
    Use to find the orbit-averaged DF energy losses, i.e. from one set of (p,e) to the next.

    Parameters:
        - p: semi-latus rectum [pc]
        - e: eccentricity [dim/less]
        - gamma_sp: DM spike index [dim/less]
        - rho_sp: DM spike normalizing density [Msun/pc^3]
        - relativistic: bool, if True use the relativistic correction
        - eff_dens: optional, use the effective DM density
    """
    pc_to_km = 3.08567758149137e13
    G_N = 4.3021937e-3 # (km/s)^2 pc M_sun^-1
    c = 2.99792458e5  # km/s
    p_pc = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))/PC_SI 
    epsilon = mu/M 
    
    r_sp = get_r_sp(M, gamma_sp, rho_sp)
    Coulomb_log = np.log(Sqrt(M/mu))

    if eff_dens is not None:
        dEdt = (1 / pc_to_km) * (2 * (1 - e**2)**(3/2) * G_N**(3/2) * mu**2 * eff_dens * Coulomb_log * p_pc**(1/2)) / Sqrt(M)
    else:
        dEdt = (1 / pc_to_km) * (2 * (1 - e**2)**(3/2) * G_N**(3/2) * mu**2 * rho_sp * r_sp**(gamma_sp) * Coulomb_log) / (Sqrt(M) * p_pc**(gamma_sp - 1/2))

    def integrand_E(theta):
        if eff_dens is not None:
            return 1 / ((1 + e * np.cos(theta))**2 * np.sqrt(1 + 2 * e * np.cos(theta) + e**2))
        return (1 + e * np.cos(theta))**(gamma_sp-2) / np.sqrt(1 + 2 * e * np.cos(theta) + e**2)
    
    result_E, error_E = quad(integrand_E, 0.0, 2*np.pi)

    E_dot_DF = (dEdt * result_E * 10**6 * MSUN_SI * G_SI / Power(C_SI, 5)) / epsilon**2 # confirmed transform from HF to FEW units

    if relativistic:
        def integrand_E(theta):
            xi = (1 + ((G_N * M)*(1 + 2 * e * np.cos(theta) + e**2) / (c**2 * p)) 
                - ((G_N * M)**2 * (1 + 2 * e * np.cos(theta) + e**2)**2 / (c**4 * p**2))
                - ((G_N * M)**3 * (1 + 2 * e * np.cos(theta) + e**2)**3 / (c**6 * p**3)))
            if eff_dens is not None:
                return (1 / ((1 + e * np.cos(theta))**2 * np.sqrt(1 + 2 * e * np.cos(theta) + e**2))) * xi
            return ((1 + e * np.cos(theta))**(gamma_sp-2) / np.sqrt(1 + 2 * e * np.cos(theta) + e**2)) * xi
        
        result_E_rel, error_E_rel = quad(integrand_E, 0.0, 2*np.pi)
        E_dot_DF_rel = (dEdt * result_E_rel * 10**6 * MSUN_SI * G_SI / Power(C_SI, 5)) / epsilon**2 # confirmed transform from HF [(km/s)^2 s^-1 M_sun] to FEW units
        return (E_dot_DF_rel)

    return E_dot_DF


def dLdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, relativistic=False, eff_dens=None):
    """Rate of change of angular momentum in FEW units [dim/less].
        Use to find the angular momentum losses from one set of (p,e, gamma) to the next below the break frequency.
    
        Parameters:
            - p: semi-latus rectum [pc]
            - e: eccentricity [dim/les]
            - gamma_sp: DM spike index [dim/les]
            - rho_sp: DM spike normalization factor [M_sun/pc^3]
            - relativistic: bool, if True use the relativistic correction
            - eff_dens: optional, use the effective DM density
    """
    pc_to_km = 3.08567758149137e13
    G_N = 4.3021937e-3 # (km/s)^2 pc M_sun^-1
    p_pc = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))/PC_SI
    epsilon = mu/M 

    r_sp = get_r_sp(M, gamma_sp, rho_sp)
    Coulomb_log = np.log(Sqrt(M/mu))

    if eff_dens is not None:
        dLdt = (2 * (1 - e**2)**(3/2) *G_N * mu**2 * eff_dens * Coulomb_log * p_pc**2) / M
    else:
        dLdt = (2 * (1 - e**2)**(3/2) * G_N * mu**2 * rho_sp * r_sp**(gamma_sp) * Coulomb_log)/(M * p_pc**(gamma_sp -2))  
   
    def integrand_L(theta):
        if eff_dens is not None:
            return 1 / ((1 + e * np.cos(theta))**2 * (1 + 2 * e * np.cos(theta) + e**2)**(3/2))
        return (1 + e * np.cos(theta))**(gamma_sp-2) / (1 + 2 * e * np.cos(theta) + e**2)**(3/2)
    
    result_L, error_L = quad(integrand_L, 0.0, 2 * np.pi)

    L_dot_DF = (dLdt * result_L * 10**6 /(M * C_SI**2)) / epsilon**2 # confirmed transform from HF [(km/s)^2 M_sun] to FEW units

    if relativistic:
        def integrand_L(theta):
            xi = (1 + ((G_N * M)*(1 + 2 * e * np.cos(theta) + e**2) / (c**2 * p))
                - ((G_N * M)**2 * (1 + 2 * e * np.cos(theta) + e**2)**2 / (c**4 * p**2)) 
                - ((G_N * M)**3 * (1 + 2 * e * np.cos(theta) + e**2)**3 / (c**6 * p**3)))
            if eff_dens is not None:
                return (1 / ((1 + e * np.cos(theta))**2 * (1 + 2 * e * np.cos(theta) + e**2)**(3/2))) * xi
            return (1 + e * np.cos(theta))**(gamma_sp-2) / (1 + 2 * e * np.cos(theta) + e**2)**(3/2) * xi
        result_L_rel, error_L_rel = quad(integrand_L, 0.0, 2*np.pi)
        L_dot_DF_rel = (dLdt * result_L_rel * 10**6 / (M * C_SI**2)) / epsilon**2 # confirmed transform from HF [(km/s)^2 M_sun] to FEW units
        return (L_dot_DF_rel)

    return (L_dot_DF)


def dLdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, relativistic=True, eff_dens=None):
    """Rate of change of angular momentum in FEW units [dim/less].
        Use to find the angular momentum losses from one set of (p,e, gamma) to the next below the break frequency.
    
        Parameters:
            - p: semi-latus rectum [pc]
            - e: eccentricity [dim/les]
            - gamma_sp: DM spike index [dim/les]
            - rho_sp: DM spike normalization factor [M_sun/pc^3]
            - relativistic: bool, if True use the relativistic correction
            - eff_dens: optional, use the effective DM density
    """
    pc_to_km = 3.08567758149137e13
    G_N = 4.3021937e-3 # (km/s)^2 pc M_sun^-1
    p_pc = (p*G_SI*MSUN_SI*M/Power(C_SI, 2))/PC_SI
    epsilon = mu/M 

    r_sp = get_r_sp(M, gamma_sp, rho_sp)
    Coulomb_log = np.log(Sqrt(M/mu))

    if eff_dens is not None:
        dLdt = (2 * (1 - e**2)**(3/2) *G_N * mu**2 * eff_dens * Coulomb_log * p_pc**2) / M
    else:
        dLdt = (2 * (1 - e**2)**(3/2) * G_N * mu**2 * rho_sp * r_sp**(gamma_sp) * Coulomb_log)/(M * p_pc**(gamma_sp -2))  
   
    def integrand_L(theta):
        if eff_dens is not None:
            return 1 / ((1 + e * np.cos(theta))**2 * (1 + 2 * e * np.cos(theta) + e**2)**(3/2))
        return (1 + e * np.cos(theta))**(gamma_sp-2) / (1 + 2 * e * np.cos(theta) + e**2)**(3/2)
    
    result_L, error_L = quad(integrand_L, 0.0, 2 * np.pi)

    L_dot_DF = (dLdt * result_L * 10**6 /(M * C_SI**2)) / epsilon**2 # confirmed transform from HF [(km/s)^2 M_sun] to FEW units

    if relativistic:
        def integrand_L(theta):
            xi = (1 + ((G_N * M)*(1 + 2 * e * np.cos(theta) + e**2) / (c**2 * p))
                - ((G_N * M)**2 * (1 + 2 * e * np.cos(theta) + e**2)**2 / (c**4 * p**2)) 
                - ((G_N * M)**3 * (1 + 2 * e * np.cos(theta) + e**2)**3 / (c**6 * p**3)))
            if eff_dens is not None:
                return (1 / ((1 + e * np.cos(theta))**2 * (1 + 2 * e * np.cos(theta) + e**2)**(3/2))) * xi
            return (1 + e * np.cos(theta))**(gamma_sp-2) / (1 + 2 * e * np.cos(theta) + e**2)**(3/2) * xi
        result_L_rel, error_L_rel = quad(integrand_L, 0.0, 2*np.pi)
        L_dot_DF_rel = (dLdt * result_L_rel * 10**6 / (M * C_SI**2)) / epsilon**2 # confirmed transform from HF [(km/s)^2 M_sun] to FEW units
        return (L_dot_DF_rel)

    return (L_dot_DF)


### FEW trajectories for each case:

### 0.1 Vacuum case
class DF_vac:
    def __init__(self, epsilon, M, mu):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu

    def __call__(self, t, y):
        
        # mass ratio
        epsilon = self.epsilon
        M = self.mu
        mu = self.mu

        # extract the four evolving parameters 
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e>= 1.0 or p < 6.0 or (p - 6 - 2 * e) < 0.05:
            return [0.0, 0.0, 0.0, 0.0]
        if e < 1e-5:
            e = 1e-5
        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # flux
        Edot = -epsilon*(EdotPN)
        Ldot = -epsilon*(LdotPN)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        return dydt


# this is the actual class that implements the trajectory.
class vac_Trajectory(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        
        rhs = DF_vac(epsilon, M, mu)
        
        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False


        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
vacuum_traj = vac_Trajectory()

# 0.2 Vacuum case output

class vacuum_output:
    def __init__(self, epsilon, M, mu, T, p0, e0):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.T = T
        self.p0 = p0
        self.e0 = e0

        self.df = pd.DataFrame(columns=['t', 'p', 'e', 'Phi_phi', 'EdotGW', 'LdotGW'])
        filename = f"vac_traj_M_{M}_mu_{mu}_T_{T}_p0_{p0}_e0_{e0}.csv"
        self.output_dir = "few_trajectories_output/csv_files/vac_traj"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_path = os.path.join(self.output_dir, filename)

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        
        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
#             print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # flux
        Edot = -epsilon*(EdotPN)
        Ldot = -epsilon*(LdotPN)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "t": t,
            "p": p,
            "e": e,
            "Phi_phi": Phi_phi,
            "EdotGW": EdotPN,
            "LdotGW": LdotPN
        }])

        # Concatenate new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return dydt
    
    def save_to_csv(self):
        """ Save the collected data to a CSV file """
        self.df.to_csv(self.file_path, index=False)
        print(f"Results saved to {self.file_path}")


# this is the actual class that implements the trajectory. 
class vac_Trajectory_output(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T_few = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu

        rhs = vacuum_output(epsilon, M, mu, T, p0, e0)
    
        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T_few,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T_few and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False

        rhs.save_to_csv()

        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
vac_traj_output = vac_Trajectory_output()

# to load the output data of an EMRI trajectory with a static spike.
def load_vac_traj_output(M, mu, T, p0, e0):
    pattern = f"few_trajectories_output/csv_files/vac_traj/vac_traj_M_{M}_mu_{mu}_T_{T}_p0_{p0}_e0_{e0}.csv"
    file_paths = glob.glob(pattern)
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found for M={M}, mu={mu}, T={T}, p0={p0}, e0={e0}")
    elif len(file_paths) > 1:
        raise ValueError(f"Multiple files found for M={M}, mu={mu}, T={T}, p0={p0}, e0={e0}. Expected only one.")

    # Return the only file path in the list
    return file_paths[0]

# Function to store data in a dictionary
def store_traj_data_vac(load_func, M, mu, T, p0, e0, data_store, traj_type):
    """
    Loads the data from the specified load_traj_output function and stores it in a dictionary
    with a key based on the parameters and trajectory type.
    load_traj_output options:
        - load_vac_traj_output
    traj_type options:
        - vac_traj
    """
    try:
        # Get file path using the provided load function
        file_path = load_func(M, mu, T, p0, e0)
        
        # Read the data
        data = pd.read_csv(file_path)

        # Create a unique key based on parameters
        key = f"{traj_type}_M_{M}_mu_{mu}_T_{T}_p0_{p0}_e0_{e0}"

        # Store the data in the dictionary
        data_store[key] = {"data": data, "M": M, "mu": mu, "traj_type": traj_type, "T": T}

    except FileNotFoundError:
        print(f"File not found for parameters: M={M}, mu={mu}, T={T}, p0={p0}, e0={e0}")
    except ValueError as e:
        print(str(e))
        

### 1.1 Static DM Spike - non relativistic
## This trajectory should be used only for static DM distributions.

# this class is instantiated and then run like the derivative function in the integrator (ex. dydt)
class DF:
    def __init__(self, epsilon, M, mu, gamma_sp, rho_sp):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp


    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        
        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
#             print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp)
        LdotDF = dLdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF )
        Ldot = -epsilon*(LdotPN + LdotDF )
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        return dydt


# this is the actual class that implements the trajectory.
class DFTrajectory(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        
        rhs = DF(epsilon, M, mu, gamma_sp, rho_sp)
        #print(rhs)

        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False


        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
static_traj = DFTrajectory()

### 1.2 Static DM Spike - non relativistic - output [DO NOT USE FOR WAVEFORMS]
# Keeps the trajectory output in a .csv file - use for plots (reduce runtime)

class DF_output:
    def __init__(self, epsilon, M, mu, gamma_sp, rho_sp, T, p0, e0):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.T = T
        self.p0 = p0
        self.e0 = e0

        self.df = pd.DataFrame(columns=['t', 'p', 'e', 'Phi_phi', 'EdotGW', 'EdotDF', 'LdotGW', 'LdotDF'])
        filename = f"static_traj_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
        self.output_dir = "few_trajectories_output/csv_files/stat_traj"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_path = os.path.join(self.output_dir, filename)

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        
        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp)
        LdotDF = dLdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF )
        Ldot = -epsilon*(LdotPN + LdotDF )
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "t": t,
            "p": p,
            "e": e,
            "Phi_phi": Phi_phi,
            "EdotGW": EdotPN,
            "EdotDF": EdotDF,
            "LdotGW": LdotPN,
            "LdotDF": LdotDF,
        }])

        # Concatenate new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return dydt
    
    def save_to_csv(self):
        """ Save the collected data to a CSV file """
        self.df.to_csv(self.file_path, index=False)
        print(f"Results saved to {self.file_path}")


# this is the actual class that implements the trajectory. 
class DFTrajectory_output(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T_few = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]

        rhs = DF_output(epsilon, M, mu, gamma_sp, rho_sp, T, p0, e0)
    
        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T_few,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T_few and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False

        rhs.save_to_csv()

        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
static_traj_output = DFTrajectory_output()

# to load the output data of an EMRI trajectory with a static spike.
def load_stat_traj_output(M, mu, rho_sp, gamma_sp, T, p0, e0):
    pattern = f"few_trajectories_output/csv_files/stat_traj/static_traj_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
    file_paths = glob.glob(pattern)
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}")
    elif len(file_paths) > 1:
        raise ValueError(f"Multiple files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}. Expected only one.")

    # Return the only file path in the list
    return file_paths[0]


### 1.3 Static DM Spike - relativistic
## This trajectory should be used only for static DM distributions - relativistic DF losses.

# this class is instantiated and then run like the derivative function in the integrator (ex. dydt)
class DF_rel:
    def __init__(self, epsilon, M, mu, gamma_sp, rho_sp):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp


    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        
        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp)
        LdotDF = dLdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF )
        Ldot = -epsilon*(LdotPN + LdotDF )
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        return dydt


# this is the actual class that implements the trajectory. 
class DFTrajectory_rel(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        
        rhs = DF_rel(epsilon, M, mu, gamma_sp, rho_sp)
        
        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False


        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
static_traj_rel = DFTrajectory_rel()

### 1.4 Static DM Spike - relativistic - output [DO NOT USE FOR WAVEFORMS]
# Keeps the trajectory output in a .csv file - use for plots (reduce runtime)

class DF_rel_output:
    def __init__(self, epsilon, M, mu, gamma_sp, rho_sp, T, p0, e0):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.T = T
        self.p0 = p0
        self.e0 = e0

        self.df = pd.DataFrame(columns=['t', 'p', 'e', 'Phi_phi', 'EdotGW', 'EdotDF', 'LdotGW', 'LdotDF'])
        filename = f"static_traj_rel_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
        self.output_dir = "few_trajectories_output/csv_files/stat_traj"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_path = os.path.join(self.output_dir, filename)

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        
        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN,LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp)
        LdotDF = dLdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF )
        Ldot = -epsilon*(LdotPN + LdotDF )
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "t": t,
            "p": p,
            "e": e,
            "Phi_phi": Phi_phi,
            "EdotGW": EdotPN,
            "EdotDF": EdotDF,
            "LdotGW": LdotPN,
            "LdotDF": LdotDF,
        }])

        # Concatenate new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return dydt
    
    def save_to_csv(self):
        """ Save the collected data to a CSV file """
        self.df.to_csv(self.file_path, index=False)
        print(f"Results saved to {self.file_path}")


# this is the actual class that implements the trajectory.
class DFTrajectory_rel_output(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T_few = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]

        rhs = DF_rel_output(epsilon, M, mu, gamma_sp, rho_sp, T, p0, e0)
    
        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T_few,  rtol=1e-10, atol=1e-10, first_step=dt/Msec)

        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T_few and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y
            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)
           
            if (p - 6 -2*e) < 0.05:
                run = False

        rhs.save_to_csv()

        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
static_traj_rel_output = DFTrajectory_rel_output()

# to load the output data of an EMRI trajectory with a static spike - relativistic DF losses.
def load_stat_traj_rel_output(M, mu, rho_sp, gamma_sp, T, p0, e0):
    pattern = f"few_trajectories_output/csv_files/stat_traj/static_traj_rel_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
    file_paths = glob.glob(pattern)
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}")
    elif len(file_paths) > 1:
        raise ValueError(f"Multiple files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}. Expected only one.")

    # Return the only file path in the list
    return file_paths[0]

# something is wrong here - 100% pinpointed by vacuum signals
# To evolve the spike between each two steps of the few trajectory.
# use for all dynamic trajectories
def spike_evolution_eff(p0, e0, t0, p, e, t, M, mu, rho_sp, gamma_sp, f_eps_init, DMS=None):
    """This function evolves the spike density between two consecutive steps of the EMRI trajectory, i.e. between (p0, e0, t0) and (p, e, t), based on the HaloFeedback code.
        Parameters:
        - p0: initial semi-latus rectum in FEW units [R_S/2]
        - e0: initial eccentricity [dim/less]
        - t0: initial time [?]
        - p: final semi-latus rectum in FEW units [R_S/2]
        - e: final eccentricity [dim/less]
        - t: final time [?]
        - M: mass of the central MBH [Msun]
        - mu: mass of the CO [Msun]
        - rho_sp: spike normalization density [Msun/pc^3]
        - gamma_sp: spike index [dim/less]
        - f_eps_init: 1D list of dimension N_GRID, initial distribution function of DM particles
        - DMS=None to initialize the Halofeedback - do not change
    
    """
    # use any function of Final_HaloFeedback_d.py as DMS.

    # unit conversions
    pc_to_km = 3.08567758149137e13

    SPEED_CUT = True

    if DMS is None:
        DMS = PowerLawSpike(f_eps_init, M=M, mu=mu, gamma=gamma_sp, rho_sp=rho_sp, Lambda=-1)
    else:
        pass

    r0_apo = DMS.r_apo(p0, e0) # initial apocenter in pc
    r0_peri = DMS.r_peri(p0, e0) # initial pericenter in pc
    v0_ecc_apo = DMS.v_ecc_apo(p0, e0) # velocity of the CO at the initial apocenter in km/s
    v0_ecc_peri = DMS.v_ecc_peri(p0, e0) # velocity of the CO at the initial pericenter in km/s

    r_apo = DMS.r_apo(p, e) # final apocenter in pc
    r_peri = DMS.r_peri(p, e) # final pericenter in pc
    v_ecc_apo = DMS.v_ecc_apo(p, e) # velocity of the CO at final apocenter in km/s
    v_ecc_peri = DMS.v_ecc_peri(p, e) # velocity of the CO at final pericenter in km/s
    
    T_orb0 = ((r0_apo * pc_to_km / v0_ecc_apo) + (r0_peri * pc_to_km / v0_ecc_peri))/2 # orbital period [s] at the begining of the FEW step
    T_orb = ((r_apo * pc_to_km / v_ecc_apo) + (r_peri * pc_to_km / v_ecc_peri))/2 # orbital period [s] at the end of the FEW step

    T_orb_mean = (T_orb0 + T_orb)/2 # mean orbital period [s] within one FEW step
    
    N_orb = int((t-t0)/T_orb_mean) # number of orbits within one FEW step
    
    # N_step determines how many steps for spike evolution within one FEW step
    # needs confirmation/commentation on Ch.5.1.2 dynamic traj
    if N_orb >= 10000:
        orbits_per_step = 10000
    elif N_orb >= 5000 and N_orb < 10000:
        orbits_per_step = 5000
    elif N_orb >= 1000 and N_orb < 5000:
        orbits_per_step = 1000
    elif N_orb >= 500 and N_orb < 1000:
        orbits_per_step = 500
    elif N_orb >= 100 and N_orb < 500:
        orbits_per_step = 100
    else:
        orbits_per_step = 10

    N_step = int(N_orb / orbits_per_step)

    if N_step <= 4:
        N_step = 4
    
    N_r = N_step
    r_list = np.linspace(r0_apo, r_peri, N_r-2) # radial steps between r0_apo and r_peri (within one FEW step)
    r_list = np.sort(np.append(r_list, (r0_peri, r_apo))) # add the r0_peri and r_apo values
    r_list = r_list[::-1] # keep the order descending, r in [pc]
    
    r_apo_ind = np.where(r_list == r_apo)[0][0]

    t_list = np.linspace(t0, t, N_r) # time steps corresponding to r steps within one FEW step 

    v_list = np.asarray([((DMS.v_ecc_apo(r, e) + DMS.v_ecc_peri(r,e)) /2) for r in r_list])

    f_eps_list = np.empty((len(DMS.f_eps), 0))
    rho_list = []
    dEdt_circ_list = []

    for i in range(1, N_r):
        r0 = r_list[i-1]
        r = r_list[i]
        t0 = t_list[i-1]
        v0 = v_list[i-1]
        v = v_list[i]
        dt = t_list[i] - t_list[i-1]
        dr = r_list[i-1] - r_list[i]

        dfdt1 = DMS.delta_f(r0, v0, dt)
        DMS.f_eps += dfdt1
        dfdt2 = DMS.delta_f(r, v, dt)
        DMS.f_eps += 0.5 * (dfdt2 - dfdt1)

        #f_eps_list = np.append(f_eps_list, DMS.f_eps)
        f_eps_list = np.column_stack((f_eps_list, DMS.f_eps))
        
        rho = DMS.rho(r) # problem
        rho_list = np.append(rho_list, rho)
        dEdt_circ = DMS.dEdt_DF(r)
        dEdt_circ_list = np.append(dEdt_circ_list, dEdt_circ)
        
    rho_apo_evol = rho_list[r_apo_ind]
    dEdt_DF_circ = dEdt_circ_list[r_apo_ind]
    dEdt_DF_circ = dEdt_DF_circ # * 10**6 * MSUN_SI * G_SI / Power(C_SI,5)) / (mu/M)**2 # in FEW units
    print("f_eps_fin",f_eps_list[:, -1])
    # we need to get the rho_apo_evol at r_apo to use for the DF losses, the rho_list from r_apo to r_peri, 
    # to use as the initial density for the next step, the f_eps for r_grid, 
    # to use as the initial distribution function for the next step and the circular energy losses for comparisons.
    return f_eps_list[:, -1], rho_apo_evol, rho_list, dEdt_DF_circ, DMS


### 2.1 Dynamic DM spike - effective DM density
## This trajectory includes the evolution of the DM spike simultaneously with the EMRI.

# this class is instantiated and then run like the derivative function in the integrator (ex. dydt)
class DF_evol_eff:
    def __init__(self, epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.f_eps_init = f_eps_init
        self.eff_dens = eff_dens

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        eff_dens = self.eff_dens

        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y
        f_eps = self.f_eps_init

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN, LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, eff_dens)
        LdotDF = dLdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, eff_dens)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF)
        Ldot = -epsilon*(LdotPN + LdotDF)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        return dydt


# this is the actual class that implements the trajectory.
class DF_evol_eff_Trajectory(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        p_init = p0
        e_init = e0
        t_init = 0

        f_eps_init = args[2]
        eff_dens_init = args[3]
        print("eff_dens_init", eff_dens_init)

        rhs = DF_evol_eff(epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens_init)
        
        

        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T, rtol=1e-12, atol=1e-12, first_step=dt/Msec)


        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y

            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)

            t = integrator.t * Msec
            # evolve the spike density and the distribution function here
            # so that
            f_eps_list, rho_eff_evol, rho_eff_list, dEdt_circ, DMS = spike_evolution_eff(p_init, e_init, t_init, p, e, t, M, mu, rho_sp, gamma_sp, f_eps_init)
            p_init = p
            e_init = e
            t_init = t
            rhs.eff_dens = rho_eff_evol
            f_eps_init = f_eps_list
            print("f_eps_init", f_eps_init)
            print("rho_eff_list", rho_eff_list)

            if (p - 6 -2*e) < 0.05:
                run = False


        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        # this needs to return only the quantities it contains in order to get waveforms from it
        # this is why i need spike_evolution, to hold the spike parameters
        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
dynamic_traj_eff = DF_evol_eff_Trajectory()
    

### 2.2 Dynamic DM spike - effective DM density - output [DO NOT USE FOR WAVEFORMS]
## This trajectory includes the evolution of the DM spike simultaneously with the EMRI.
# Keeps the trajectory output in a .csv file - use for plots (reduce runtime)

class DF_evol_eff_output:
    def __init__(self, epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens, T, p0, e0):
        # T, p0 and e0 are needed here only for the csv file naming.
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.f_eps_init = f_eps_init
        self.eff_dens = eff_dens
        self.T = T
        self.p0 = p0
        self.e0 = e0

        # Create a DataFrame to store the values of interest
        self.df = pd.DataFrame(columns=["t", "p", "e", "Phi_phi", "EdotGW", "EdotDF", "LdotGW", "LdotDF", "eff_dens"])
        filename = f"dynamic_traj_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T * (M * MTSUN_SI)/ YRSID_SI}_p0_{p0}_e0_{e0}.csv"
        self.output_dir = "few_trajectories_output/csv_files/dyn_traj"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_path = os.path.join(self.output_dir, filename)

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        eff_dens = self.eff_dens
        f_eps_init = self.f_eps_init

        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y
        f_eps = self.f_eps_init

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN, LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        # CHECK THIS 
        EdotDF = dEdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, eff_dens)
        LdotDF = dLdt_DF_ecc_eff(M, mu, p, e, gamma_sp, rho_sp, eff_dens)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF)
        Ldot = -epsilon*(LdotPN + LdotDF)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        
        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "t": t,
            "p": p,
            "e": e,
            "Phi_phi": Phi_phi,
            "EdotGW": EdotPN,
            "EdotDF": EdotDF,
            "LdotGW": LdotPN,
            "LdotDF": LdotDF,
            "eff_dens": eff_dens,
            "f_eps_init": f_eps_init
        }])

        # Concatenate new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return dydt
    
    def save_to_csv(self):
        """ Save the collected data to a CSV file """
        self.df.to_csv(self.file_path, index=False)
        print(f"Results saved to {self.file_path}")


# this is the actual class that implements the trajectory.
class DF_evol_eff_Trajectory_output(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        p_init = p0
        e_init = e0
        t_init = 0

        f_eps_init = args[2]
        eff_dens_init = args[3]

        rhs = DF_evol_eff_output(epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens_init, T, p0, e0)
        

        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T, rtol=1e-12, atol=1e-12, first_step=dt/Msec)


        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y

            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)

            t = integrator.t * Msec
            # evolve the spike density and the distribution function here
            # so that
            f_eps_list, rho_eff_evol, rho_eff_list, dEdt_circ = spike_evolution_eff(p_init, e_init, t_init, p, e, t, M, mu, rho_sp, gamma_sp, f_eps_init)
            p_init = p
            e_init = e
            t_init = t
            rhs.eff_dens = rho_eff_evol
            f_eps_init = f_eps_list

            if (p - 6 -2*e) < 0.05:
                run = False

        rhs.save_to_csv()
        
        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        # this needs to return only the quantities it contains in order to get waveforms from it
        # this is why i need spike_evolution, to hold the spike parameters
        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
dynamic_traj_eff_output = DF_evol_eff_Trajectory_output()

# to load the output data of an EMRI trajectory with a dynamic spike.
def load_dyn_traj_output(M, mu, rho_sp, gamma_sp, T, p0, e0):
    pattern = f"few_trajectories_output/csv_files/dyn_traj/dynamic_traj_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
    file_paths = glob.glob(pattern)
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}")
    elif len(file_paths) > 1:
        raise ValueError(f"Multiple files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}. Expected only one.")

    # Return the only file path in the list
    return file_paths[0]


### 2.3 Dynamic DM spike - effective DM density - rel
## This trajectory includes the evolution of the DM spike simultaneously with the EMRI.

# this class is instantiated and then run like the derivative function in the integrator (ex. dydt)
class DF_evol_eff_rel:
    def __init__(self, epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens):
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.f_eps_init = f_eps_init
        self.eff_dens = eff_dens

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        eff_dens = self.eff_dens

        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y
        f_eps = self.f_eps_init

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN, LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        EdotDF = dEdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, eff_dens)
        LdotDF = dLdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, eff_dens)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF)
        Ldot = -epsilon*(LdotPN + LdotDF)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        return dydt


# this is the actual class that implements the trajectory. 
class DF_evol_eff_Trajectory_rel(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        p_init = p0
        e_init = e0
        t_init = 0

        f_eps_init = args[2]
        eff_dens_init = args[3]

        rhs = DF_evol_eff_rel(epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens_init)
        

        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T, rtol=1e-12, atol=1e-12, first_step=dt/Msec)


        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y

            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)

            t = integrator.t * Msec
            # evolve the spike density and the distribution function here
            # so that
            f_eps_list, rho_eff_evol, rho_eff_list, dEdt_circ, DMS = spike_evolution_eff(p_init, e_init, t_init, p, e, t, M, mu, rho_sp, gamma_sp, f_eps_init)
            p_init = p
            e_init = e
            t_init = t
            rhs.eff_dens = rho_eff_evol
            f_eps_init = f_eps_list
            print("f_eps_init", f_eps_init)

            if (p - 6 -2*e) < 0.05:
                run = False


        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        # this needs to return only the quantities it contains in order to get waveforms from it
        # this is why i need spike_evolution, to hold the spike parameters
        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
dynamic_traj_eff_rel = DF_evol_eff_Trajectory_rel()


### 2.4 Dynamic DM spike - effective DM density - rel - output [DO NOT USE FOR WAVEFORMS]
## This trajectory includes the evolution of the DM spike simultaneously with the EMRI.
# Keeps the trajectory output in a .csv file - use for plots (reduce runtime)

class DF_evol_eff_rel_output:
    def __init__(self, epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens, T, p0, e0):
        # T, p0 and e0 are needed here only for the csv file naming.
        self.epsilon = epsilon
        self.M = M
        self.mu = mu
        self.rho_sp = rho_sp
        self.gamma_sp = gamma_sp
        self.f_eps_init = f_eps_init
        self.eff_dens = eff_dens
        self.T = T
        self.p0 = p0
        self.e0 = e0

        # Create a DataFrame to store the values of interest
        self.df = pd.DataFrame(columns=["t", "p", "e", "Phi_phi", "EdotGW", "EdotDF", "LdotGW", "LdotDF", "eff_dens"])
        filename = f"dynamic_traj_rel_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T * (M * MTSUN_SI)/ YRSID_SI}_p0_{p0}_e0_{e0}.csv"
        self.output_dir = "few_trajectories_output/csv_files/dyn_traj"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.file_path = os.path.join(self.output_dir, filename)

    def __call__(self, t, y):

        # mass ratio
        epsilon = self.epsilon
        M = self.M
        mu = self.mu
        rho_sp = self.rho_sp
        gamma_sp = self.gamma_sp
        eff_dens = self.eff_dens

        # extract the four evolving parameters
        p, e, Phi_phi, Phi_r = y
        f_eps = self.f_eps_init

        # guard against bad integration steps
        if e >= 1.0 or p < 6.0 or (p - 6 - 2* e) < 0.05:
            print('bad', y)
            return [0.0, 0.0, 0.0, 0.0]
        if e<1e-5:
            e = 1e-5

        # Azimuthal frequency
        # perform elliptic calculations
        Omega_phi, _, Omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)

        # GW
        EdotPN, LdotPN = Edot_GW(p,e)

        # dynamical friction parameters
        # CHECK THIS 
        EdotDF = dEdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, eff_dens)
        LdotDF = dLdt_DF_ecc_eff_rel(M, mu, p, e, gamma_sp, rho_sp, eff_dens)

        # flux
        Edot = -epsilon*(EdotPN + EdotDF)
        Ldot = -epsilon*(LdotPN + LdotDF)
        
        # time derivatives
        pdot = (-2*(Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*(3 + Power(e,2) - p)*Power(p,1.5) + Ldot*Power(-4 + p,2)*Sqrt(-3 - Power(e,2) + p)))/(4*Power(e,2) - Power(-6 + p,2))

        edot = -((Edot*Sqrt((4*Power(e,2) - Power(-2 + p,2))/(3 + Power(e,2) - p))*Power(p,1.5)*
        (18 + 2*Power(e,4) - 3*Power(e,2)*(-4 + p) - 9*p + Power(p,2)) +
        (-1 + Power(e,2))*Ldot*Sqrt(-3 - Power(e,2) + p)*(12 + 4*Power(e,2) - 8*p + Power(p,2)))/
        (e*(4*Power(e,2) - Power(-6 + p,2))*p))

        Phi_phi_dot = Omega_phi;

        Phi_r_dot = Omega_r;
        
        dydt = [pdot, edot, Phi_phi_dot, Phi_r_dot]
            
        if e<=1e-5:
            dydt[1] = 0.0

        
        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "t": t,
            "p": p,
            "e": e,
            "Phi_phi": Phi_phi,
            "EdotGW": EdotPN,
            "EdotDF": EdotDF,
            "LdotGW": LdotPN,
            "LdotDF": LdotDF,
            "eff_dens": eff_dens
        }])

        # Concatenate new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return dydt
    
    def save_to_csv(self):
        """ Save the collected data to a CSV file """
        self.df.to_csv(self.file_path, index=False)
        print(f"Results saved to {self.file_path}")

# this is the actual class that implements the trajectory.
class DF_evol_eff_Trajectory_rel_output(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass

    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, x0, *args, T=1.0, dt=10.0, **kwargs):

        # set up quantities and integrator
        y0 = [p0, e0, 0.0, 0.0]

        T = T * YRSID_SI / (M * MTSUN_SI)

        Msec = M * MTSUN_SI

        epsilon = mu/M

        M = M
        mu = mu
        gamma_sp = args[0]
        rho_sp = args[1]
        p_init = p0
        e_init = e0
        t_init = 0

        f_eps_init = args[2]
        eff_dens_init = args[3]

        rhs = DF_evol_eff_rel_output(epsilon, M, mu, rho_sp, gamma_sp, f_eps_init, eff_dens_init, T, p0, e0)
        

        # the tolerance is important!
        integrator = DOP853(rhs, 0.0, y0, T, rtol=1e-12, atol=1e-12, first_step=dt/Msec)


        t_out, p_out, e_out = [], [], []
        Phi_phi_out, Phi_r_out = [], []
        t_out.append(0.0)
        p_out.append(p0)
        e_out.append(e0)
        Phi_phi_out.append(0.0)
        Phi_r_out.append(0.0)

        # run the integrator down to T or separatrix
        run = True
        while integrator.t < T and run:

            integrator.step()
            
            p, e, Phi_phi, Phi_r = integrator.y

            t_out.append(integrator.t * Msec)
            p_out.append(p)
            e_out.append(e)
            Phi_phi_out.append(Phi_phi)
            Phi_r_out.append(Phi_r)

            t = integrator.t * Msec
            # evolve the spike density and the distribution function here
            # so that
            f_eps_list, rho_eff_evol, rho_eff_list, dEdt_circ = spike_evolution_eff(p_init, e_init, t_init, p, e, t, M, mu, rho_sp, gamma_sp, f_eps_init)
            p_init = p
            e_init = e
            t_init = t
            rhs.eff_dens = rho_eff_evol
            f_eps_init = f_eps_list

            if (p - 6 -2*e) < 0.05:
                run = False

        rhs.save_to_csv()
        
        # read out data. It must return length 6 tuple
        t = np.asarray(t_out)
        p = np.asarray(p_out)
        e = np.asarray(e_out)
        Phi_phi = np.asarray(Phi_phi_out)
        Phi_r = np.asarray(Phi_r_out)

        # need to add polar info
        Phi_theta = Phi_phi.copy()  # by construction
        x = np.ones_like(Phi_theta)

        # this needs to return only the quantities it contains in order to get waveforms from it
        # this is why i need spike_evolution, to hold the spike parameters
        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
    
dynamic_traj_eff_rel_output = DF_evol_eff_Trajectory_rel_output()

# to load the output data of an EMRI trajectory with a dynamic spike.
def load_dyn_traj_rel_output(M, mu, rho_sp, gamma_sp, T, p0, e0):
    pattern = f"few_trajectories_output/csv_files/dyn_traj/dynamic_traj_rel_M_{M}_mu_{mu}_rho_sp_{rho_sp}_gamma_sp_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}.csv"
    file_paths = glob.glob(pattern)
    
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}")
    elif len(file_paths) > 1:
        raise ValueError(f"Multiple files found for M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}. Expected only one.")

    # Return the only file path in the list
    return file_paths[0]


# Function to store data in a dictionary
def store_traj_data(load_func, M, mu, rho_sp, gamma_sp, T, p0, e0, data_store, traj_type):
    """
    Loads the data from the specified load_traj_output function and stores it in a dictionary
    with a key based on the parameters and trajectory type.
    load_traj_output options:
        - load_stat_traj_output
        - load_stat_traj_rel_output
        - load_dyn_traj_output
        - load_dyn_traj_rel_output
    traj_type options:
        - static_traj
        - static_traj_rel
        - dynamic_traj
        - dynamic_traj_rel
    """
    try:
        # Get file path using the provided load function
        file_path = load_func(M, mu, rho_sp, gamma_sp, T, p0, e0)
        
        # Read the data
        data = pd.read_csv(file_path)

        # Create a unique key based on parameters
        key = f"{traj_type}_M_{M}_mu_{mu}_rho_{rho_sp}_gamma_{gamma_sp}_T_{T}_p0_{p0}_e0_{e0}"

        # Store the data in the dictionary
        data_store[key] = {"data": data, "M": M, "mu": mu, "traj_type": traj_type, "rho_sp": rho_sp, "T": T, "gamma_sp": gamma_sp}

    except FileNotFoundError:
        print(f"File not found for parameters: M={M}, mu={mu}, rho_sp={rho_sp}, gamma_sp={gamma_sp}, T={T}, p0={p0}, e0={e0}")
    except ValueError as e:
        print(str(e))



# to get the waveforms we need to set up some keyword arguments

# settings for the elliptic integrals
mp.dps = 25
mp.pretty = True

# settings for the waveform
use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # 0 (1) we (don't) want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

# set omp threads one of two ways
num_threads = 4

# this is the general way to set it for all computations
#from few.utils.utility import omp_set_num_threads
#omp_set_num_threads(num_threads)

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    num_threads=num_threads,  # 2nd way for specific classes
)

### Time Domain
## Time Domain Waveforms vacuum
class ApproxSchwarzschildEccentricFluxVacuum(SchwarzschildEccentricWaveformBase):
    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            vac_Trajectory, # here it is the new trajectory!
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False
    
TD_wave_vacuum = ApproxSchwarzschildEccentricFluxVacuum()    

## Time Domain Waveforms with static DM spike
class ApproxSchwarzschildEccentricFluxDF(SchwarzschildEccentricWaveformBase):
    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            DFTrajectory, # here it is the new trajectory!
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False
    
TD_wave_stat = ApproxSchwarzschildEccentricFluxDF()
    
    ## Time Domain Waveforms with static relativistic DM spike

class ApproxSchwarzschildEccentricFluxDF_rel(SchwarzschildEccentricWaveformBase):
    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            DFTrajectory_rel, # here it is the new trajectory!
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False

TD_wave_stat_rel = ApproxSchwarzschildEccentricFluxDF_rel()

## Time Domain Waveforms with dynamic DM spike

class ApproxSchwarzschildEccentricFluxDF_evol(SchwarzschildEccentricWaveformBase):
    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            DF_evol_eff_Trajectory, # here it is the new trajectory!
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False
    
TD_wave_dyn = ApproxSchwarzschildEccentricFluxDF_evol()

## Time Domain Waveforms with dynamic relativistic DM spike
class ApproxSchwarzschildEccentricFluxDF_evol_rel(SchwarzschildEccentricWaveformBase):
    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            DF_evol_eff_Trajectory_rel, # here it is the new trajectory!
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False
    
TD_wave_dyn_rel = ApproxSchwarzschildEccentricFluxDF_evol_rel()