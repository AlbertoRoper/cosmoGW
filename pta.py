"""
pta.py is a Python routine with functions used in the analysis of observations
by pulsar timing array (PTA) collaborations: NANOGrav, PPTA, EPTA, and IPTA.

Author: Alberto Roper Pol
Created: 01/12/2021
Updated: 01/11/2023 (release of the cosmoGW code)

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave signal
from primordial magnetic fields in the Pulsar Timing Array frequency band," Phys.Rev.D 105,
12, 123502 (2022), arXiv:2201.05630.

NRPCS21 - A. Neronov, A. Roper Pol, C. Caprini, D. Semikoz, "NANOGrav signal from magnetohydrodynamic
turbulence at the QCD phase transition in the early Universe," Phys. Rev. D 103, L041302 (2021),
arXiv:2009.14174.

NG12.5 - [NANOGrav], "The NANOGrav 12.5 yr data set: Search for an isotropic stochastic gravitational-wave
background," Astrophys. J. Lett. 905, L34 (2020), arXiv:2009.04496.

GonPPTA21 - B. Goncharov et al., "On the evidence for a common-spectrum process in the search for
the nanohertz gravitational-wave background with the Parkes Pulsar Timing Array,"
Astrophys. J. Lett. 917, L19 (2021), arXiv:2107.12112.

EPTA21 - [EPTA], "Common-red-signal analysis with 24-yr high-precision timing of the European
Pulsar Timing Array: Inferences in the stochastic gravitational-wave background search,"
Mon. Not. Royal Astron. Soc. 508, 4970 (2021), arXiv:2110.13184.

AntIPTA22 - J. Antoniadis et al., "The International Pulsar Timing Array second data release:
Search for an isotropic Gravitational Wave Background," Mon. Not. Royal Astron. Soc. 510, 4873 (2022),
arXiv:2201.03980.

ChenGdR20 - Talk at the 3rd Annual Assembly GdR “Ondes Gravitationnneles" by S. Chen,
"25 years of European Pulsar Timing Array." (2020).
"""

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import spectra as sp
import pandas as pd
import os

HOME = os.getcwd()
dir0 = HOME + '/detector_sensitivity/PTA/'

# reference values
f_bend_ref = 1.035e-8*u.Hz
f_ref = 7e-8
flim_ref = 1.25e-8*u.Hz

# reference time span of observations (e.g., 12.5 yr for NANOGrav)
Tdata_EPTA = 24
Tdata_PPTA = 15
Tdata_NG = 12.5
Tdata_IPTA = 31
# reference time span in seconds
TT_NG = Tdata_NG*u.yr
TT_NG = TT_NG.to(u.s)
TT_EPTA = Tdata_EPTA*u.yr
TT_EPTA = TT_EPTA.to(u.s)
TT_PPTA = Tdata_PPTA*u.yr
TT_PPTA = TT_PPTA.to(u.s)
TT_IPTA = Tdata_IPTA*u.yr
TT_IPTA = TT_IPTA.to(u.s)

def get_gamma_A(file, dir0=dir0, beta_b=False, Omega_b=False, fref=0, disc=1000,
                plot=False, color='blue', alpha=.4, return_all=False,
                fill=True):

    """
    Function to read files with results of PTA amplitude vs slope of
    GW backgrounds.

    Arguments:
        file -- name of the file to read
        dir0 -- directory where the data is stored (default is 'detector_sensitivity/PTA')
        beta_b -- option to return the slope of the GW background
                  instead of gamma (beta = 5 - gamma) (default False)
        Omega_b -- option to return the amplitude of the GW energy density
                   instead of the amplitude of the characteristic strain
                   (default False)
        fref -- frequency used as reference to define the PL of the GW
                background (default is 1/(1 year))
        disc -- number of discretization points for the reconstructed A vs
                gamma function (default 1000)
        plot -- option to generate the plot (default False)
        color -- option to chose the color of the plot (default 'blue')
        alpha -- option to chose the transparency of the shaded region within
                 the range of allowed amplitudes for every slope
                 (default 0.4)
        return_all -- option to return all functions computed

    Returns:
        gamma -- slope of the power spectral density
        A -- amplitude
        if return_all is selected:
            gammas -- refined equidistant gamma values
            A1, A2 -- maximum and minimum allowed values of the amplitude
                      by the observations
    """

    df = pd.read_csv(dir0 + file)
    gamma = np.array(df['gamma'])
    A = np.array(df['A'])
    gammas, A1, A2 = gammas_As(gamma, A, disc=disc)
    if beta_b: gammas = 5 - gammas
    if Omega_b:
        if beta_b:
            A = cosmoGW.Omega_A(A=A, fref=fref, beta=gamma)
            A1 = cosmoGW.Omega_A(A=A1, fref=fref, beta=gammas)
            A2 = cosmoGW.Omega_A(A=A2, fref=fref, beta=gammas)
        else:
            A = cosmoGW.Omega_A(A=A, fref=fref, beta=5 - gamma)
            A1 = cosmoGW.Omega_A(A=A1, fref=fref, beta=5 - gammas)
            A2 = cosmoGW.Omega_A(A=A2, fref=fref, beta=5 - gammas)
    else:
        if fref != 0:
            if beta_b:
                alphas = gammas/2 - 1
                alpha0 = gamma/2 - 1
            else:
                alphas = .5*(3 - gammas)
                alpha0 = .5*(3 - gamma)
            fyr = 1/u.yr
            fyr = fyr.to(u.Hz)
            A *= (fref.value/fyr.value)**alpha0
            A1 *= (fref.value/fyr.value)**alphas
            A2 *= (fref.value/fyr.value)**alphas

    if plot:
        plt.plot(gammas, A1, lw=2, color=color)
        plt.plot(gammas, A2, lw=2, color=color)
        plt.vlines(gammas[0], A1[0], A2[0], color=color, lw=2)
        if fill: plt.fill_between(gammas, A1, A2, color=color, alpha=alpha)

    if return_all:
        return gamma, A, gammas, A1, A2
    else: return gamma, A

def gammas_As(gamma, A, disc=1000):

    """
    Function that uses the amplitude vs slope gamma data to divide into upper
    bound and lower bounds.
    It assumes that the data starts with the smallest gamma and it is given
    in counterclockwise order in the amplitude vs slope plot.

    Arguments:
        gamma -- array of slopes of the power spectral density
        A -- array of amplitudes of the characteristic strain of the background
        disc -- number of discretization points for the reconstructed A vs
                gamma function (default 1000)
    """

    # min and max values of gamma
    gam_0 = gamma[0]
    ind = np.argmax(gamma)
    gam_f = gamma[ind]
    # subdivide array of gammas and
    gamma1 = gamma[:ind]
    gamma2 = gamma[ind:]
    A1 = A[:ind]
    A2 = A[ind:]
    # reorder upper limit
    inds = np.argsort(gamma2)
    gamma2 = gamma2[inds]
    A2 = A2[inds]
    # discretize and interpolate for equidistant arrays in gamma
    gammas = np.linspace(gam_0, gam_f, disc)
    A1 = 10**np.interp(np.log10(gammas), np.log10(gamma1), np.log10(A1))
    A2 = 10**np.interp(np.log10(gammas), np.log10(gamma2), np.log10(A2))

    return gammas, A1, A2

def Sf_PL_PTA(A, f, gamma, f_bend=f_bend_ref,kappa=0.1, broken=True, h0=1.):

    """
    Function to compute the power spectral density Sf, the characteristic
    strain spectrum hc(f), and the GW energy density spectrum OmGW (f),
    from the amplitude A of the background of GW characteristic strain,
    which takes the following form:

    h_c(f) = A (f/fyr)^((3 - gamma)/2)
    
    Reference is RPCNS22, section III.

    Arguments:
        A -- amplitudes of the GW background
        f -- frequency array
        gamma -- negative slope of the power spectral density
        f_bend -- bend frequency of the broken power law model (default is
                 1.035e-8 Hz)
        kappa -- smoothing parameter of the broken power law model (default 0.1)
        broken -- allows to use a broken power law model, instead of a single
                  power law (default True)
        h0 -- parameterizes the uncertainties (Hubble tension) on the value
              of the Hubble rate (default 1)

    Returns:
        Sf -- power spectral density as a function of frequency
        hc -- characteristic strain spectrum
        OmGW -- GW energy density spectrum
    """

    import cosmoGW

    f1yr = 1/u.yr
    f1yr = f1yr.to(u.Hz)
    if broken:
        Sf = A**2/12/np.pi**2*(f.value/f1yr.value)**(-gamma)* \
            (1 + (f.value/f_bend.value)**(1/kappa))**(kappa*gamma)*f1yr**(-3)
    else: Sf = A**2/12/np.pi**2*(f.value/f1yr.value)**(-gamma)*f1yr**(-3)
    hc = cosmoGW.hc_Sf(f, Sf)
    OmGW = cosmoGW.hc_OmGW(f, hc, d=-1, h0=h0)

    return Sf, hc, OmGW

def read_PTA_data(dir0=dir0, beta_b=True, Omega_b=True, fref=0, return_all=False,
                  plot=False, fill_1s=True, fill_2s=True):

    """
    Function that reads the data from the PTA observations (in terms of region
    of the allowed amplitudes and slopes of the GW background).

    h_c(f) = A (f/fyr)^((3 - gamma)/2)

    Files are stored in detector_sensitivity/PTA, see README for a description of the files.
    
    References are: NG12.5 for 'NANOGrav_x', GonPPTA21 for 'PPTA_x', ChenGdR20 for 'EPTA_x_old',
    EPTA21 for 'EPTA_x', and AntIPTA22 for 'IPTA_x'.

    Arguments:
        dir0 -- directory where the data is stored (default is 'detector_sensitivity/PTA')
        beta_b -- option to return the slope of the GW background
                  instead of gamma (beta = 5 - gamma) (default True)
        Omega_b -- option to return the amplitude of the GW energy density
                   instead of the amplitude of the characteristic strain
                   (default True)
        fref -- frequency used as reference to define the PL of the GW
                background (default is 1/(1 year))
        return_all -- option to return all the computed amplitudes and slopes
        plot -- option to generate the plots of the GW background allowed by
                the regions of amplitudes A and slopes gamma

    Returns:
        A -- amplitudes of the GW background
        gamma -- slopes of the GW background
        if return_all is selected:
            gamma -- refined equidistant gamma values
            A1, A2 -- maximum and minimum allowed values of the amplitude
                      by the observations

            NG, P, E, and I indicate the NANOGrav, the PPTA, the EPTA, and the
            IPTA collaborations;
            sPL and bPL indicate single and broken power law fits;
            1s and 2s indicate the 1sigma and 2sigma confidence intervals
    """

    # NANOGrav results for a single power law with 1-sigma confidence
    file = 'NANOGrav_singlePL_1s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    fref=fref, disc=300, return_all=return_all,
                    plot=plot, color='green', alpha=.8, fill=fill_1s)
    if return_all:
        gamma_NG_sPL_1s, A1_NG_sPL_1s, A2_NG_sPL_1s = [_[2], _[3], _[4]]
    else: gamma_NG_sPL_1s, A_NG_sPL_1s = [_[0], _[1]]

    # NANOGrav results for a single power law with 1-sigma confidence
    file = 'NANOGrav_singlePL_2s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    fref=fref,  disc=5000, return_all=return_all,
                    plot=plot, color='green', fill=fill_2s)
    if return_all:
        gamma_NG_sPL_2s, A1_NG_sPL_2s, A2_NG_sPL_2s = [_[2], _[3], _[4]]
    else: gamma_NG_sPL_2s, A_NG_sPL_2s = [_[0], _[1]]

    # NANOGrav results for a broken power law with 1-sigma confidence
    file = 'NANOGrav_brokenPL_1s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    fref=fref, disc=300, return_all=return_all,
                    plot=plot, color='blue', alpha=.8, fill=fill_1s)
    if return_all:
        gamma_NG_bPL_1s, A1_NG_bPL_1s, A2_NG_bPL_1s = [_[2], _[3], _[4]]
    else: gamma_NG_bPL_1s, A_NG_bPL_1s = [_[0], _[1]]

    # NANOGrav results for a broken power law with 1-sigma confidence
    file = 'NANOGrav_brokenPL_2s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_2s,
                    fref=fref, plot=plot, color='blue')
    if return_all:
        gamma_NG_bPL_2s, A1_NG_bPL_2s, A2_NG_bPL_2s = [_[2], _[3], _[4]]
    else: gamma_NG_bPL_2s, A_NG_bPL_2s = [_[0], _[1]]

    # PPTA results with 1-sigma confidence
    file = 'PPTA_blue_1s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_1s,
                    fref=fref, plot=plot, color='red', alpha=.8)
    if return_all:
        gamma_P_b_1s, A1_P_b_1s, A2_P_b_1s = [_[2], _[3], _[4]]
    else: gamma_P_b_1s, A_P_b_1s = [_[0], _[1]]

    # PPTA results with 2-sigma confidence
    file = 'PPTA_blue_2s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_2s,
                    fref=fref, plot=plot, color='red')
    if return_all:
        gamma_P_b_2s, A1_P_b_2s, A2_P_b_2s = [_[2], _[3], _[4]]
    else: gamma_P_b_2s, A_P_b_2s = [_[0], _[1]]

    # EPTA results with 1-sigma confidence from old reference
    file = 'EPTA_singlePL_1s_old.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_1s,
                    fref=fref, plot=False, color='purple', alpha=.8)
    if return_all:
        gamma_E_1s_old, A1_E_1s_old, A2_E_1s_old = [_[2], _[3], _[4]]
    else: gamma_E_1s_old, A_E_1s_old = [_[0], _[1]]

    # EPTA results with 2-sigma confidence from old reference
    file = 'EPTA_singlePL_2s_old.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_2s,
                    fref=fref, plot=False, color='purple')
    if return_all:
        gamma_E_2s_old, A1_E_2s_old, A2_E_2s_old = [_[2], _[3], _[4]]
    else: gamma_E_2s_old, A_E_2s_old = [_[0], _[1]]

    # EPTA results with 1-sigma confidence
    file = 'EPTA_singlePL_1s_EP.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_1s,
                    fref=fref, plot=plot, color='purple', alpha=.8)
    if return_all:
        gamma_E_1s, A1_E_1s, A2_E_1s = [_[2], _[3], _[4]]
    else: gamma_E_1s, A_E_1s = [_[0], _[1]]

    # EPTA results with 2-sigma confidence
    file = 'EPTA_singlePL_2s_EP.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_2s,
                    fref=fref, plot=plot, color='purple')
    if return_all:
        gamma_E_2s, A1_E_2s, A2_E_2s = [_[2], _[3], _[4]]
    else: gamma_E_2s, A_E_2s = [_[0], _[1]]

    # IPTA results (DR2) with 1-sigma confidence
    file = 'IPTA_singlePL_1s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_1s,
                    fref=fref, plot=plot, color='black', alpha=.8)
    if return_all:
        gamma_I_1s, A1_I_1s, A2_I_1s = [_[2], _[3], _[4]]
    else: gamma_I_1s, A_I_1s = [_[0], _[1]]

    # IPTA results (DR2) with 2-sigma confidence
    file = 'IPTA_singlePL_2s.csv'
    _ = get_gamma_A(file, dir0=dir0, beta_b=beta_b, Omega_b=Omega_b,
                    return_all=return_all, fill=fill_2s,
                    fref=fref, plot=plot, color='black')
    if return_all:
        gamma_I_2s, A1_I_2s, A2_I_2s = [_[2], _[3], _[4]]
    else: gamma_I_2s, A_I_2s = [_[0], _[1]]

    if return_all:
        return (gamma_NG_sPL_2s, A1_NG_sPL_2s, A2_NG_sPL_2s, gamma_NG_sPL_1s,
                A1_NG_sPL_1s, A2_NG_sPL_1s, gamma_NG_bPL_2s, A1_NG_bPL_2s,
                A2_NG_bPL_2s, gamma_NG_bPL_1s, A1_NG_bPL_1s, A2_NG_bPL_1s,
                gamma_P_b_2s, A1_P_b_2s, A2_P_b_2s, gamma_P_b_1s, A1_P_b_1s,
                A2_P_b_1s, gamma_E_2s, A1_E_2s_old, A2_E_2s_old, gamma_E_1s_old,
                A1_E_1s_old, A2_E_1s_old, gamma_E_2s, A1_E_2s, A2_E_2s,
                gamma_E_1s, A1_E_1s, A2_E_1s, gamma_I_2s, A1_I_2s, A2_I_2s,
                gamma_I_1s, A1_I_1s, A2_I_1s)
    else:
        return (gamma_NG_sPL_2s, A_NG_sPL_2s, gamma_NG_sPL_1s, A_NG_sPL_1s,
            gamma_NG_bPL_2s, A_NG_bPL_2s, gamma_NG_bPL_1s, A_NG_bPL_1s,
            gamma_P_b_2s, A_P_b_2s, gamma_P_b_1s, A_P_b_1s, gamma_E_2s_old,
            A_E_2s_old, gamma_E_1s_old, A_E_1s_old, gamma_E_2s, A_E_2s,
            gamma_E_1s, A_E_1s, gamma_I_2s, A_I_2s, gamma_I_1s, A_I_1s)

def single_CP(f, gamma, A1, A2, beta=0, broken=True, plot=False, alpha=.1,
              T=Tdata_NG, color='blue'):

    """
    Function to compute the GW spectral maxima and minima at each frequency
    from the allowed amplitudes and slopes from the PTA results.

    Arguments:
        f -- array of frequencies to compute the background
        gamma -- slopes of the power spectral density allowed by PTA data
        A1 -- minimum amplitude allowed by PTA data for the slope gamma
        A2 -- maximum amplitude allowed by PTA data for the slope gamma
        beta -- slope used to compute the GW energy density spectrum
                (default is 0)
        broken -- option to plot the broken power law fit (default is True)
        plot -- option to plot the resulting spectra (default False)
        alpha -- transparency of the plot (default 1)
        color -- color of the plot (default 'blue')
        T -- duration of the mission (default is 12.5 years for NANOGrav)

    Returns:
        Sf -- power spectral density as a function of frequency
        hc -- characteristic strain spectrum
        OmGW -- GW energy density spectrum
            a, b indicate minima and maxima at each frequency
    """

    T = T*u.yr
    T = T.to(u.s)
    f1yr = 1/u.yr
    f1yr = f1yr.to(u.Hz)
    gam = 5 - beta
    inside = True
    if gam < min(gamma) or gam > max(gamma): inside = False
    As = np.linspace(np.interp(gam, gamma, A1), np.interp(gam, gamma, A2), 10)
    if not inside: As = np.zeros(10)
    ACP = np.sqrt(As[0]*As[-1])
    Sf_a, hc_a, OmGW_a = Sf_PL_PTA(As[0], f, gam, broken=broken)
    Sf_b, hc_b, OmGW_b = Sf_PL_PTA(As[-1], f, gam, broken=broken)
    Sf_c, hc_c, OmGW_c = Sf_PL_PTA(ACP, f, gam, broken=broken)

    if plot:
        if alpha < 1:
            plt.plot(f, np.sqrt(Sf_c/T), color=color)
            plt.plot(f, np.sqrt(Sf_a/T), color=color, ls='dashed', lw=1)
            plt.plot(f, np.sqrt(Sf_b/T), color=color, ls='dashed', lw=1)
        plt.fill_between(f, np.sqrt(Sf_a/T), np.sqrt(Sf_b/T),
                         alpha=alpha, color=color)

    return Sf_a, hc_a, OmGW_a, Sf_b, hc_b, OmGW_b

def CP_delay(betas, dir0=dir0, obs='NANOGrav_brokenPL_1s',
             plot=False, colors=[], alpha=.1, maxf=f_ref, Nf=300):
    
    '''
    Function that computes the power spectral density Sf, the characteristic
    strain spectrum hc(f), and the GW energy density spectrum OmGW (f), corresponding
    to the maximum and minimum amplitudes of the power law fits for a range of observatories
    obsss.
    
    Arguments:
        betas -- array of slopes used to compute the GW energy density spectrum
        dir0 -- directory where the data is stored (default is 'detector_sensitivity/PTA')
        obs -- specific observatory to be read (default is 'NANOGrav_brokenPL_1s')
        maxf -- maximum frequency used to generate the array of frequencies (default is 7e-8 Hz)
        Nf -- number of points in the frequency array (default is 300)
        plot -- option to plot the data (default is False)
        colors -- array of colors (one per value of beta) used for plotting the data (if plot is True)
        alpha -- opacitiy of the plots (default is 0.1)

    Returns:
        f -- array of frequencies
        Sf_a, hc_a, OmGW_a -- spectral density arrays for minimum of the allowed amplitudes
        Sf_c, hc_c, OmGW_c -- spectral density arrays for maximum of the allowed amplitudes
    '''

    # Compute the data of NANOGrav and PPTA
    _ = read_PTA_data(dir0=dir0, beta_b=False, Omega_b=False, plot=False,
                      return_all=True)
    gamma_NG_sPL_2s, A1_NG_sPL_2s, A2_NG_sPL_2s = [_[0], _[1], _[2]]
    gamma_NG_sPL_1s, A1_NG_sPL_1s, A2_NG_sPL_1s = [_[3], _[4], _[5]]
    gamma_NG_bPL_2s, A1_NG_bPL_2s, A2_NG_bPL_2s = [_[6], _[7], _[8]]
    gamma_NG_bPL_1s, A1_NG_bPL_1s, A2_NG_bPL_1s = [_[9], _[10], _[11]]
    gamma_P_b_2s, A1_P_b_2s, A2_P_b_2s = [_[12], _[13], _[14]]
    gamma_P_b_1s, A1_P_b_1s, A2_P_b_1s = [_[15], _[16], _[17]]
    gamma_E_2s_old, A1_E_2s_old, A2_E_2s_old = [_[18], _[19], _[20]]
    gamma_E_1s_old, A1_E_1s_old, A2_E_1s_old = [_[21], _[22], _[23]]
    gamma_E_2s, A1_E_2s, A2_E_2s = [_[24], _[25], _[26]]
    gamma_E_1s, A1_E_1s, A2_E_1s = [_[27], _[28], _[29]]
    gamma_I_2s, A1_I_2s, A2_I_2s = [_[30], _[31], _[32]]
    gamma_I_1s, A1_I_1s, A2_I_1s = [_[33], _[34], _[35]]

    # time of data considered for each of the PTA collaborations (in years)
    if 'NANOGrav' in obs: Tdata = Tdata_NG
    if 'PPTA' in obs: Tdata = Tdata_PPTA
    if 'EPTA' in obs: Tdata = Tdata_EPTA
    if 'IPTA' in obs: Tdata = Tdata_IPTA

    # duration of NANOGrav observations
    broken = True
    # available observatory data under detector_sensitivity/PTA
    if obsss == []:
        obsss = ['NANOGrav_brokenPL_1s', 'NANOGrav_brokenPL_2s',
                 'NANOGrav_singlePL_1s', 'NANOGrav_singlePL_2s', 'PPTA_blue_1s',
                 'PPTA_blue_2s', 'EPTA_1s_old', 'EPTA_2s_old', 'EPTA_1s',
                 'EPTA_2s', 'IPTA_1s', 'IPTA_2s']

    if obs == 'NANOGrav_brokenPL_1s':
        gamma = gamma_NG_bPL_1s
        A1 = A1_NG_bPL_1s
        A2 = A2_NG_bPL_1s
    elif obs == 'NANOGrav_brokenPL_2s':
        gamma = gamma_NG_bPL_2s
        A1 = A1_NG_bPL_2s
        A2 = A2_NG_bPL_2s
    elif obs == 'NANOGrav_singlePL_1s':
        broken = False
        gamma = gamma_NG_sPL_1s
        A1 = A1_NG_sPL_1s
        A2 = A2_NG_sPL_1s
    elif obs == 'NANOGrav_singlePL_2s':
        broken = False
        gamma = gamma_NG_sPL_2s
        A1 = A1_NG_sPL_2s
        A2 = A2_NG_sPL_2s
    elif obs == 'PPTA_blue_1s':
        broken = False
        gamma = gamma_P_b_1s
        A1 = A1_P_b_1s
        A2 = A2_P_b_1s
    elif obs == 'PPTA_blue_2s':
        broken = False
        gamma = gamma_P_b_2s
        A1 = A1_P_b_2s
        A2 = A2_P_b_2s
    elif obs == 'EPTA_1s_old':
        broken = False
        gamma = gamma_E_1s_old
        A1 = A1_E_1s_old
        A2 = A2_E_1s_old
    elif obs == 'EPTA_2s_old':
        broken = False
        gamma = gamma_E_2s_old
        A1 = A1_E_2s_old
        A2 = A2_E_2s_old
    elif obs == 'EPTA_1s':
        broken = False
        gamma = gamma_E_1s
        A1 = A1_E_1s
        A2 = A2_E_1s
    elif obs == 'EPTA_2s':
        broken = False
        gamma = gamma_E_2s
        A1 = A1_E_2s
        A2 = A2_E_2s
    elif obs == 'IPTA_1s':
        broken = False
        gamma = gamma_I_1s
        A1 = A1_I_1s
        A2 = A2_I_1s
    elif obs == 'IPTA_2s':
        broken = False
        gamma = gamma_I_2s
        A1 = A1_I_2s
        A2 = A2_I_2s
    else:
        gamma = 0
        A1 = 0
        A2 = 0
        print('The selected observation data \' should be one of',
                ' the available ', obsss)

    fmin = 1/Tdata/u.yr
    fmin = fmin.to(u.Hz)
    f = np.logspace(np.log10(fmin.value), np.log10(maxf), Nf)*u.Hz

    if len(colors) == 0: colors = ['blue']*len(betas)
    Sf_a = np.zeros((len(betas), len(f)))
    hc_a = np.zeros((len(betas), len(f)))
    OmGW_a = np.zeros((len(betas), len(f)))
    Sf_b = np.zeros((len(betas), len(f)))
    hc_b = np.zeros((len(betas), len(f)))
    OmGW_b = np.zeros((len(betas), len(f)))
    Sf_c = np.zeros((len(betas), len(f)))
    hc_c = np.zeros((len(betas), len(f)))
    OmGW_c = np.zeros((len(betas), len(f)))
    for i in range(0, len(betas)):
        Sf_a[i, :], hc_a[i, :], OmGW_a[i, :], Sf_c[i, :], \
                    hc_c[i, :], OmGW_c[i, :] = \
                            single_CP(f, gamma, A1, A2, beta=betas[i],
                                      color=colors[i], plot=plot,
                                      broken=broken, alpha=alpha,
                                      T=Tdata)

    return f, Sf_a, hc_a, OmGW_a, Sf_c, hc_c, OmGW_c

def OmGW_PTA(betas, dir0=dir0, ff='Om'):

    """
    Function that returns the GW energy density minima and maxima for a range
    of slopes using the reported amplitudes for 2 sigma confidence by NANOGrav,
    PPTA, EPTA, and IPTA.

    Arguments:
        betas -- array with values of the slopes beta
        dir0 -- directory where the data is stored (default is 'detector_sensitivity/PTA')
        ff -- option to choose which spectrum to compute (default is 'Om' for GW
              energy density, other available options are 'hc', 'Sf', or 'tdel' for
              the spectrum of time delay)
    """

    # NANOGrav broken PL
    _ = CP_delay(betas, dir0=dir0, obs='NANOGrav_brokenPL_2s')
    f_bPL_NG = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_NG)
        OmGW_b = np.sqrt(_[4]/TT_NG)
    min_OmGW_bPL_NG, max_OmGW_bPL_NG = sp.get_min_max(f_bPL_NG, OmGW_a,
                                                      OmGW_b)

    # NANOGrav single PL
    _ = CP_delay(betas, dir0=dir0, obs='NANOGrav_singlePL_2s')
    f_sPL_NG = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_NG)
        OmGW_b = np.sqrt(_[4]/TT_NG)
    min_OmGW_sPL_NG, max_OmGW_sPL_NG = sp.get_min_max(f_sPL_NG, OmGW_a,
                                                      OmGW_b)

    # PPTA single PL
    _ = CP_delay(betas, dir0=dir0, obs='PPTA_blue_2s')
    f_sPL_P = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_PPTA)
        OmGW_b = np.sqrt(_[4]/TT_PPTA)
    min_OmGW_sPL_P, max_OmGW_sPL_P = sp.get_min_max(f_sPL_P, OmGW_a,
                                                    OmGW_b)

    # EPTA single PL
    _ = CP_delay(betas, dir0=dir0, obs='EPTA_2s_old')
    f_sPL_E_old = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_EPTA)
        OmGW_b = np.sqrt(_[4]/TT_EPTA)
    min_OmGW_sPL_E_old, max_OmGW_sPL_E_old = sp.get_min_max(f_sPL_E_old,
                                                            OmGW_a, OmGW_b)

    # EPTA single PL
    _ = CP_delay(betas, dir0=dir0, obs='EPTA_2s')
    f_sPL_E = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_EPTA)
        OmGW_b = np.sqrt(_[4]/TT_EPTA)
    min_OmGW_sPL_E, max_OmGW_sPL_E = sp.get_min_max(f_sPL_E, OmGW_a,
                                                    OmGW_b)

    # IPTA single PL
    _ = CP_delay(betas, dir0=dir0, obs='IPTA_2s')
    f_sPL_I = _[0]
    if ff == 'Om':
        OmGW_a = _[3]
        OmGW_b = _[6]
    if ff == 'Sf':
        OmGW_a = _[1]
        OmGW_b = _[4]
    if ff == 'hc':
        OmGW_a = _[2]
        OmGW_b = _[5]
    if ff == 'tdel':
        OmGW_a = np.sqrt(_[1]/TT_IPTA)
        OmGW_b = np.sqrt(_[4]/TT_IPTA)
    min_OmGW_sPL_I, max_OmGW_sPL_I = sp.get_min_max(f_sPL_I, OmGW_a,
                                                    OmGW_b)

    return (f_bPL_NG, min_OmGW_bPL_NG, max_OmGW_bPL_NG, f_sPL_NG,
            min_OmGW_sPL_NG, max_OmGW_sPL_NG, f_sPL_P, min_OmGW_sPL_P,
            max_OmGW_sPL_P, f_sPL_E_old, min_OmGW_sPL_E_old, max_OmGW_sPL_E_old,
            f_sPL_E, min_OmGW_sPL_E, max_OmGW_sPL_E, f_sPL_I, min_OmGW_sPL_I,
            max_OmGW_sPL_I)

def plot_PTA_all(ff='tdel', dir0=dir0, betas=[], obs='all', lines=True, alp_bl=0.3, alp_E=0.2,
                 ret=True, beta_min=-2, beta_max=5, Nbeta=100, flim=flim_ref):

    """
    Function that overplots the GW spectra Omega_GW (f) = Omyr (f/fyr)^beta
    for the values of beta and Omyr reported by the PTA collaborations.

    Arguments:
        ff -- option to chose what function to plot (default 'tdel' for time
              delay in seconds, 'Sf' for power spectral density, 'hc' for
              characteristic amplitude spectrum, and 'Om' for GW energy density
              spectrum)
        dir0 -- directory where the data is stored (default is 'detector_sensitivity/PTA')
        betas -- range of slopes considered for the plot (default is all
                 possible values, from -2 to 5)
        obs -- observatory data used for the results
                (options are 'all', 'NGs', 'P', 'E' and 'I')
        lines -- option to explicitly plot the boundaring lines on top of the
                 allowed region.
    """

    if betas == []: betas = np.linspace(beta_min, beta_max, Nbetas)

    (f_bPL_NG, min_OmGW_bPL_NG, max_OmGW_bPL_NG, f_sPL_NG, min_OmGW_sPL_NG,
    max_OmGW_sPL_NG, f_sPL_P, min_OmGW_sPL_P, max_OmGW_sPL_P, f_sPL_E_old,
    min_OmGW_sPL_E_old, max_OmGW_sPL_E_old, f_sPL_E, min_OmGW_sPL_E,
    max_OmGW_sPL_E, f_sPL_I, min_OmGW_sPL_I, max_OmGW_sPL_I) = \
                    OmGW_PTA(betas, dir0=dir0, ff=ff)

    good = np.where(f_bPL_NG.value < flim.value)
    
    if obs == 'all' or obs == 'NGs':
        plt.fill_between(f_sPL_NG.value, min_OmGW_sPL_NG, max_OmGW_sPL_NG,
                     color='darkgreen', alpha=.1)
        if lines:
            plt.plot(f_sPL_NG.value, min_OmGW_sPL_NG, color='darkgreen', lw=2)
            plt.plot(f_sPL_NG.value, max_OmGW_sPL_NG, color='darkgreen', lw=2)
            
    if obs == 'all' or obs == 'NGb':
        plt.fill_between(f_bPL_NG[good].value, min_OmGW_bPL_NG[good],
                         max_OmGW_bPL_NG[good], color='blue', alpha=alp_bl)
        if lines:
            plt.plot(f_bPL_NG[good].value, min_OmGW_bPL_NG[good], color='blue', lw=2)
            plt.plot(f_bPL_NG[good].value, max_OmGW_bPL_NG[good], color='blue', lw=2)
            
    if obs == 'all' or obs == 'P':
        plt.fill_between(f_sPL_P.value, min_OmGW_sPL_P, max_OmGW_sPL_P,
                         color='red', alpha=.1)
        if lines:
            plt.plot(f_sPL_P.value, min_OmGW_sPL_P, color='red', lw=2)
            plt.plot(f_sPL_P.value, max_OmGW_sPL_P, color='red', lw=2)
            
    if obs == 'all' or obs == 'E':
        plt.fill_between(f_sPL_E.value, min_OmGW_sPL_E, max_OmGW_sPL_E,
                         color='purple', alpha=alp_E)
        if lines:
            plt.plot(f_sPL_E.value, min_OmGW_sPL_E, color='purple', lw=2)
            plt.plot(f_sPL_E.value, max_OmGW_sPL_E, color='purple', lw=2)
            
    if obs == 'all' or obs == 'I':
        plt.fill_between(f_sPL_I.value, min_OmGW_sPL_I, max_OmGW_sPL_I,
                         color='black', alpha=alp_E)
        if lines:
            plt.plot(f_sPL_I.value, min_OmGW_sPL_I, color='black', lw=2)
            plt.plot(f_sPL_I.value, max_OmGW_sPL_I, color='black', lw=2)

    if ret:
        return (f_bPL_NG, min_OmGW_bPL_NG, max_OmGW_bPL_NG, f_sPL_NG,
                min_OmGW_sPL_NG, max_OmGW_sPL_NG, f_sPL_P, min_OmGW_sPL_P,
                max_OmGW_sPL_P, f_sPL_E_old, min_OmGW_sPL_E_old, max_OmGW_sPL_E_old,
                f_sPL_E, min_OmGW_sPL_E, max_OmGW_sPL_E, f_sPL_I, min_OmGW_sPL_I,
                max_OmGW_sPL_I)