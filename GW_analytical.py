"""
GW_analytical.py is a Python routine that contains analytical
calculations relative to GW backgrounds produced by MHD turbulence
and sound waves in the early Universe. It also includes some useful mathematical
functions.

Author: Alberto Roper Pol
Created: 01/12/2021
Updated: 01/11/2023 (release of the cosmoGW code)

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave
signal from primordial magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RPNCBS23 - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer, D. Semikoz, "LISA and γ-ray telescopes as
multi-messenger probes of a first-order cosmological phase transition," arXiv:2307.10744 (2023)

RPPC23 - A. Roper Pol, S. Procacci, C. Caprini, "Characterization of the gravitational wave spectrum
from sound waves within the sound shell model," arXiv:2308.12943

RPCM23 - A. Roper Pol, C. Caprini, A. S. Midiri, "Gravitational wave spectrum from slowly
evolving sources: constant-in-time model," arXiv:23xx.xxxxx (2023)
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_sets

import os
HOME = os.getcwd()

### Reference values
cs2 = 1/3      # speed of sound

### Reference slopes
a_ref = 4      # Batchelor spectrum k^4
b_ref = 5/3    # Kolmogorov spectrum k^(-5/3)
alp_ref = 2

############### ANALYTICAL FUNCTIONS USED FOR A SMOOTHED BROKEN POWER LAW ###############

def smoothed_bPL(k, A=1, a=a_ref, b=b_ref, alp=alp_ref, norm=True, kpeak=1, Omega=False):
    
    """
    Function that returns the value of the smoothed broken power law (bPL) model
    for a spectrum of the form:
    
        zeta(K) = (b + abs(a))^(1/alp) K^a/[ b + c K^(alp(a + b)) ]^(1/alp),
        
    where K = k/kpeak, c = 1 if a = 0 or c = abs(a) otherwise. This spectrum is defined
    such that kpeak is the correct position of the peak and its maximum amplitude is given
    by A.
    
    If norm is set to Fale, then the non-normalized spectrum is used:
    
        zeta (K) = K^a/(1 + K^(alp(a + b)))^(1/alp)
    
    The function is only correctly defined when b > 0 and a + b >= 0
    
    Reference is RPCNS22, equations 6 and 8

    Arguments:

        k -- array of wave numbers
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        norm -- option to normalize the spectrum such that its peak is located at
                kpeak and its maximum value is A
        kpeak -- spectral peak, i.e., position of the break from k^a to k^(-b)
        Omega -- option to use the integrated energy density as the input A

    Returns:
        spectrum array
    """
    
    if b < max(0, -a):
        print('b has to be larger than 0 and -a')
        return 0*k**0
    
    c = abs(a)
    if a == 0: c = 1

    spec = A*(k/kpeak)**a
    if norm:
        m = (b + abs(a))**(1/alp)
        spec = m*spec/(b + c*(k/kpeak)**((a+b)*alp))**(1/alp)

    else: spec = spec/(1 + (k/kpeak)**((a+b)*alp))**(1/alp)

    if Omega: spec = spec/kpeak/calA(a=a, b=b, alp=alp)
    
    return spec

def complete_beta(a, b):
    
    '''
    Function that computes the complete beta function, only converges for
    positive arguments.
    
    B(a, b; x \to \infty) = \int_0^x u^(a - 1) (1 - u)^(b - 1) du
    
    Arguments:
        a, b -- arguments a, b
    
    Returns:
        B -- value of the complete beta function
    '''
    
    import math as m
    
    if a > 0 and b > 0: B = m.gamma(a)*m.gamma(b)/m.gamma(a + b)
    else:
        print('arguments of beta function need to be positive')
        B = 0

    return B

def Iabn(a=a_ref, b=b_ref, alp=alp_ref, n=0):
    
    '''
    Function that computes the moment n of the smoothed dbPL spectra:
    
    \int K^n zeta(K) dK

    Reference is RPCM23, appendix A

    Arguments:
        
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        n -- moment of the integration
        
    Returns: value of the n-th moment
    '''

    alp2 = 1/alp/(a + b)
    a_beta = (a + n + 1)*alp2
    b_beta = (b - n - 1)*alp2

    if b_beta > 0 and a_beta > 0:

        calI = calIab_n_alpha(alp=alp, a=a, b=b, n=n)
        comp_beta = complete_beta(a_beta, b_beta)
        return comp_beta*calI

    else:
        if b_beta <= 0:
            print('b + n has to be larger than 1 for the integral',
                  'to converge')
        if a_beta <= 0:
            print('a + n has to be larger than -1 for the integral',
                  'to converge')
        return 0

def calA(a=a_ref, b=b_ref, alp=alp_ref):
    
    '''
    Function that computes the parameter {\cal A} = Iab,0 that relates the
    peak and the integrated values of the smoothed_bPL spectrum

    References are RPCNS22, equation 8, and RPCM23, appendix A

    Arguments:

        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
    '''
    
    return Iabn(alp=alp, a=a, b=b, n=0)

def calC(a=a_ref, b=b_ref, alp=alp_ref, tp='vort'):
    
    '''
    Function that computes the parameter {\cal C} that allows to
    compute the TT-projected stress spectrum by taking the convolution of the
    smoothed bPL spectra over \kk and \kk - \pp.
    
    It gives the spectrum of the stress of Gaussian vortical non-helical fields
    as
    
    P_\Pi (0) = 2 \pi^2 EM*^2 {\cal C} / k*

    References are RPCNS22, equation 22, for vortical and RPPC23, equation 46,
    for compressional fields. Detailed reference is RPCM23, appendix A

    Arguments:
        
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        tp -- type of sourcing field: 'vort' or 'comp' available
    '''
    
    if tp == 'vort': pref = 28/15
    elif tp == 'comp': pref = 32/15

    else:
        print('tp has to be vortical (vort) or compressional (comp)')
        pref = 0.
    
    return pref*Iabn(alp=alp/2, a=a*2, b=b*2, n=-2)

############### ANALYTICAL TEMPLATE USED FOR A DOUBLE SMOOTHED BROKEN POWER LAW ###############

def smoothed_double_bPL(k, kpeak1, kpeak2, A=1, a=a_ref, b=1, c=b_ref, alp1=alp_ref, alp2=alp_ref, kref=1.):

    """
    Function that returns the value of the smoothed double broken power law (double_bPL) model
    for a spectrum of the form:
    
        zeta(K) = K^a/(1 + (K/K1)^[(a - b)*alp1])^(1/alp1)/(1 + (K/K2)^[(c + b)*alp2])^(1/alp2)
        
    where K = k/kref, K1 and K2 are the two position peaks, a is the low-k slope, c is the intermediate
    slope, and -c is the high-k slope. alp1 and alp2 are the smoothness parameters for each spectral
    transition.
    
    Reference is RPPC23, equation 50. Also used in RPNCBS23, equation 7

    Arguments:

        k -- array of wave numbers
        kpeak1, kpeak2 -- peak positions
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at intermediate wave numbers, k^b
        c -- slope of the spectrum at high wave numbers, k^(-c)
        alp1, alp2 -- smoothness of the transitions from one power law to the other
        kref -- reference wave number used to normalize the spectrum (default is 1)

    Returns:
        spectrum array
    """

    K = k/kref
    K1 = k/kref
    K2 = k/kref

    spec1 = (1 + (K/K1)**((a - b)*alp1))**(1/alp1)
    spec2 = (1 + (K/K2)**((c + b)*alp2))**(1/alp2)
    spec = A*K^a/spec1/spec2

    return spec