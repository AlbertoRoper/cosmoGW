"""
spectra.py is a Python routine that contains description for specific spectral templates,
postprocessing routines for numerical spectra, and other mathematical routines.

Author: Alberto Roper Pol
Created: 01/01/2021
Updated: 20/07/2023 (release of the cosmoGW code)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import plot_sets

################### GENERAL FUNCTIONS FOR NUMERICAL SPECTRA ###################

def compute_kpeak(k, E, tol=.01, quiet=False, exp=1):

    """
    Function that computes the maximum of the spectrum E and its spectral
    peak.

    Arguments:
        k -- array of wave numbers
        E -- array of the spectral values
        tol -- factor to avoid faulty maxima due to nearly flat spectrum
               (default 1%)
        quiet -- option to print out the result if quiet is False
                 (default False)
        exp -- exponent used to compensate the spectrum by k^exp to define the max in
               flat spectra (default is 1)

    Return:
        kpeak -- position of the spectral peak
        Emax -- maximum value of the spectrum
    """

    max1 = np.argmax(E)
    indmax = max1

    if E[max1] == 0:
        Emax = 0
        kpeak = 0
    else:
        max2 = np.argmax(k**exp*E)
        # if the maximum of the spectrum is within tol of the maximum of k*E,
        # then we take as the maximum value where k*E is maximum, to take into
        # account flat and nearly flat spectra
        if abs(E[max1] - E[max2])/E[max1] < tol: indmax = max2
        Emax = E[indmax]
        kpeak = k[indmax]

    if not quiet:
        print('The maximum value of the spectrum is ', Emax,
              ' and the spectral peak is ', kpeak)

    return kpeak, Emax

def max_E_kf(k, E, exp=0):

    """
    Function that computes the maximum of a spectrum compensated by the
    wave number, i.e., max(E*k^exp)

    Arguments:
        k -- array of wave numbers
        E -- array of spectral values
        exp -- exponent of k (default 0)
        
    Returns:
        max_k -- value of k at the peak of k^exp*E
        max_E -- value of E at the peak of k^exp*E
    """

    indmax = np.argmax(k**exp*E)
    max_k = k[indmax]
    max_E = E[indmax]

    return max_k, max_E

def characteristic_k(k, E, exp=1):

    """
    Function that computes the characteristic wave number from the spectrum.

    Arguments:
        k -- array of wave numbers
        E -- array of spectral values
        exp -- exponent used to define the characteristic wave number
               k_ch ~ (\int k^exp E dk/\int E dk)^(1/exp)
               (default 1)

    Returns:
        kch -- characteristic wave number defined with the power 'exp'
    """

    k = k[np.where(k != 0)]
    E = E[np.where(k != 0)]
    spec_mean = np.trapz(E, k)
    integ = np.trapz(E*k**exp, k)
    # avoid zero division
    if exp >= 0 and spec_mean == 0: spec_mean = 1e-30
    if exp < 0 and integ == 0: integ = 1e-30
    kch = (integ/spec_mean)**(1/exp)

    return kch

def min_max_stat(t, k, E, abs_b=True, indt=0, indtf=-1, plot=False, hel=False):

    """
    Function that computes the minimum, the maximum, and the averaged
    functions over time of a spectral function.

    Arguments:
        t -- time array
        k -- wave number array
        E -- spectrum 2d array (first index t, second index k)
        abs_b -- option to take absolute value of spectrum (default True)
        indt, indtf -- indices of time array to perform the average
                           from t[indt] to t[indtf] (default is 0 to final time)
        plot -- option to overplot all spectral functions from t[indt] to t[indtf]
        hel -- option for helical spectral functions where positive and
               negative values can appear (default False)
               It then returns min_E_pos, min_E_neg, max_E_pos, max_E_neg
               referring to the maximum/minimum absolute values of the positive
               and negative values of the helical funtion.

    Returns:
        min_E -- maximum values of the spectral function over time
        max_E -- maximum values of the spectral function over time
        stat_E -- averaged values of the spectral function over time
                   from t[indt] to t[-1]
        if hel return positive and negative values separately
            min_E_neg, min_E_pos, max_E_neg, max_E_pos
    """

    if hel:
        min_E_neg = np.zeros(len(k)) + 1e30
        max_E_neg = np.zeros(len(k)) - 1e30
        min_E_pos = np.zeros(len(k)) + 1e30
        max_E_pos = np.zeros(len(k)) - 1e30
    else:
        min_E = np.zeros(len(k)) + 1e30
        max_E = np.zeros(len(k)) - 1e30

    if indtf == -1: indtf = len(t) - 1
    for i in range(indt, indtf + 1):
        # split between positive and negative values
        if hel:
            if plot: plt.plot(k, abs(E[i,:]))
            x_pos, x_neg, f_pos, f_neg, color = red_blue_func(k, E[i, :])
            for j in range(0, len(k)):
                if k[j] in x_pos:
                    indx = np.where(x_pos == k[j])[0][0]
                    min_E_pos[j] = min(min_E_pos[j], f_pos[indx])
                    max_E_pos[j] = max(max_E_pos[j], f_pos[indx])
                else:
                    indx = np.where(x_neg == k[j])[0][0]
                    min_E_neg[j] = min(min_E_neg[j], abs(f_neg[indx]))
                    max_E_neg[j] = max(max_E_neg[j], abs(f_neg[indx]))
        else:
            if abs_b: E = abs(E)
            if plot: plt.plot(k, E[i,:])
            min_E = np.minimum(E[i,:], min_E)
            max_E = np.maximum(E[i,:], max_E)

    # averaged spectrum
    stat_E = np.trapz(E[indt:indtf, :], t[indt:indtf], axis=0)/(t[indtf] - t[indt])
    if plot: plt.plot(k, stat_E, lw=4, color='black')
    
    # correct min and max values with averaged values when have not been found
    if hel:
        min_E_pos[np.where(min_E_pos == 1e30)] = \
                abs(stat_E[np.where(min_E_pos == 1e30)])
        min_E_neg[np.where(min_E_neg == 1e30)] = \
                abs(stat_E[np.where(min_E_neg == 1e30)])
        max_E_pos[np.where(max_E_pos == -1e30)] = \
                abs(stat_E[np.where(max_E_pos == -1e30)])
        max_E_neg[np.where(max_E_neg == -1e30)] = \
                abs(stat_E[np.where(max_E_neg == -1e30)])
    else:
        min_E[np.where(min_E == 1e30)] = \
                abs(stat_E[np.where(min_E == 1e30)])
        if abs_b:
            max_E[np.where(max_E == -1e30)] = \
                    abs(stat_E[np.where(max_E == -1e30)])
        else:
            max_E[np.where(max_E == -1e30)] = \
                    (stat_E[np.where(max_E == -1e30)])
    if plot:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$k$')

    if hel: return min_E_pos, min_E_neg, max_E_pos, max_E_neg, stat_E
    else: return min_E, max_E, stat_E

def red_blue_func(x, f, col=0):

    """
    Function that splits an array into positive and negative values, and
    assigns colors (red to positive and blue to negative).

    Arguments:
        x -- array of x
        f -- array of the function values
        col -- option to choose blue and red (default 0 is red for positive
               and blue for negative, 1 is swapped)

    Returns:
        x_pos -- array of x values where f is positive
        x_neg -- array of x values where f is negative
        f_pos -- array of f values where f is positive
        f_neg -- array of f values where f is negative
        color -- array of colors assigned (blue and red)
    """

    N = len(f)
    color = []
    f_pos=[]; x_pos=[]
    f_neg=[]; x_neg=[]
    for i in range(0, N):
        sgn = np.sign(f[i])
        if sgn > 0:
            if col == 0: color.append('red')
            if col == 1: color.append('blue')
            f_pos.append(f[i])
            x_pos.append(x[i])
        else:
            if col == 0: color.append('blue')
            if col == 1: color.append('red')
            f_neg.append(f[i])
            x_neg.append(x[i])
    f_pos = np.array(f_pos)
    f_neg = np.array(f_neg)
    x_pos = np.array(x_pos)
    x_neg = np.array(x_neg)

    return x_pos, x_neg, f_pos, f_neg, color

def local_max(k, E, order=1):

    """
    Function that computes the local maxima of the spectrum.

    Arguments:
        k -- array of wave numbers
        E -- spectrum E
        order -- order of the local maximum solver, which uses
                 scipy.signal.argrelextrema

    Returns:
        kmax -- position of the local maxima
        Emax -- values of the local maxima
    """

    inds_model_max = argrelextrema(E, np.greater, order=order)
    kmax = k[inds_model_max]
    Emax = E[inds_model_max]

    return kmax, Emax

def mean_pos_loglog(k, E):

    """
    Function that computes the loglog middle values km, EM of the intervals
    of the arrays k, E

    Arguments:
        k -- array of wave numbers
        E -- array of spectrum values

    Returns:
        km -- array of middle log values of the k intervals
        Em -- array of middle log values of the E intervals
    """

    N = len(k)
    km = np.zeros(N + 1)
    Em = np.zeros(N + 1)
    km[0] = k[0]
    Em[0] = E[0]
    for i in range(1, N):
        km[i] = np.sqrt(k[i + 1]*k[i])
        Em[i] = np.sqrt(E[i + 1]*E[i])
    km[-1] = k[-1]
    Em[-1] = E[-1]

    return km, Em

def combine(k1, k2, E1, E2, fact=1., klim=10, exp=2):

    """
    Function that combines the spectra and wave number of two runs and uses
    the ratio between their magnetic amplitudes (facM) to compensate the
    spectrum by facM^exp.
    
    It uses E2/fact^exp at k < klim and E1 at k >= klim 

    Arguments:
        k1, k2 -- wave number arrays of runs 1 and 2
        E1, E2 -- spectral values arrays of runs 1 and 2
        fact -- ratio of the spectra amplitudes (default 1)
        klim -- wave number at which we switch from run2 to run 1
                (default 10)
        exp -- exponent used in fact to compensate the spectra (default 2,
               which correspond to GW spectra compensated by ratio
               between magnetic spectra)

    Returns:
        k -- combined wave number array
        E -- combined spectra
    """

    k = np.append(k2[np.where(k2 <= klim)], k1[np.where(k1 > klim)])
    E = np.append(E2[np.where(k2 <= klim)]/fact**exp, E1[np.where(k1 > klim)])

    return k, E

def slopes_loglog(k, E):

    """
    Function that computes numerically the power law slope of a function
    E(k), taken to be the exponent of the tangent power law, i.e.,
    (\partial \ln E)/(\partial \ln k)

    Arguments:
        k -- independent variable
        E -- dependent variable

    Returns:
        slopes -- slope of E at each k in a loglog plot
    """

    slopes = np.zeros(len(k))
    slopes[0] = (np.log10(E[1]) - np.log10(E[0]))/ \
                        (np.log10(k[1]) - np.log10(k[0]))
    slopes[1] = (np.log10(E[2]) - np.log10(E[0]))/ \
                        (np.log10(k[2]) - np.log10(k[0]))
    for i in range(2, len(k) - 2):
         slopes[i] = (np.log10(E[i + 2]) + np.log10(E[i + 1]) \
                            - np.log10(E[i - 2]) - np.log10(E[i - 1]))/\
                            (np.log10(k[i + 1])+ \
                            np.log10(k[i + 2])-np.log10(k[i - 1]) - \
                            np.log10(k[i - 2]))
    slopes[-1] = (np.log10(E[-1]) - np.log10(E[-2]))/\
                        (np.log10(k[-1]) - np.log10(k[-2]))
    slopes[-2] = (np.log10(E[-1]) - np.log10(E[-3]))/\
                        (np.log10(k[-1]) - np.log10(k[-3]))
    return slopes
