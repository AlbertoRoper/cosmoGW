"""
GW_models.py is a Python routine that contains analytical and semi-analytical
models and templates of cosmological GW backgrounds.

Author: Alberto Roper Pol
Created: 29/08/2024

Other contributors: Antonino Midiri, Simona Procacci

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave
signal from primordial magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RPNCBS23 - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer, D. Semikoz, "LISA and γ-ray telescopes as
multi-messenger probes of a first-order cosmological phase transition," arXiv:2307.10744 (2023)

RPPC23 - A. Roper Pol, S. Procacci, C. Caprini, "Characterization of the gravitational wave spectrum
from sound waves within the sound shell model," Phys.Rev.D 109 (2024) 6, 063531, arXiv:2308.12943

HH19 - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_sets

import os
HOME = os.getcwd()

### Reference values
cs2 = 1/3      # speed of sound

#################### SOUND-SHELL MODEL FOR SOUND WAVES IN PTs ####################

####### Kinetic spectra computed for the sound-shell model from f' and l functions
####### f' and l functions need to be previously computed from the self-similar
####### fluid perturbations induced by expanding bubbles (see hydro_bubbles.py)

def compute_kin_spec_dens(z, vws, fp, l, sp='sum', type_n='exp', cs2=cs2, min_qbeta=-4,
                          max_qbeta=5, Nqbeta=1000, min_TT=-1, max_TT=3, NTT=5000):
    
    '''
    Function that computes the kinetic power spectral density assuming exponential or simultaneous
    nucleation.
    
    Arguments:
        z -- array of values of z
        vws -- array of wall velocities
        fp -- function f'(z) computed from the sound-shell model using compute_fs
        l -- function lambda(z) computed from the sound-shell model using compute_fs
        sp -- type of function computed for the kinetic spectrum description
        type_n -- type of nucleation hystory (default is exponential 'exp',
                  another option is simultaneous 'sym')
        dens_spec -- option to return power spectral density (if True, default), or kinetic
                     spectrum (if False)
    '''
    
    if sp == 'sum': A2 = .25*(cs2*l**2 + fp**2)

    q_beta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT = np.logspace(min_TT, max_TT, NTT)
    q_ij, TT_ij = np.meshgrid(q_beta, TT, indexing='ij')
    Pv = np.zeros((len(vws), len(q_beta)))

    funcT = np.zeros((len(vws), len(q_beta), len(TT)))
    for i in range(0, len(vws)):
        if type_n == 'exp':
            funcT[i, :, :] = np.exp(-TT_ij)*TT_ij**6*np.interp(TT_ij*q_ij, z, A2[i, :])
        if type_n == 'sim':
            funcT[i, :, :] = .5*np.exp(-TT_ij**3/6)*TT_ij**8*np.interp(TT_ij*q_ij, z, A2[i, :])
        Pv[i, :] = np.trapz(funcT[i, :, :], TT, axis=1)

    return q_beta, Pv

def compute_kin_spec(vws, q_beta, Pv, corr=True, cs2=cs2):
    
    '''
    Function that computes the kinetic spectrum as a function of k Rast from
    the power spectral density as a function of q/beta
    '''

    EK = np.zeros((len(vws), len(q_beta)))
    cs = np.sqrt(cs2)
    if corr: Rstar_beta = (8*np.pi)**(1/3)*np.maximum(vws, cs)
    else: Rstar_beta = (8*np.pi)**(1/3)*vws
    kks = np.zeros((len(vws), len(q_beta)))

    for i in range(0, len(vws)):
        kks[i, :] = q_beta*Rstar_beta[i]
        pref = kks[i, :]**2/Rstar_beta[i]**6/(2*np.pi**2)
        EK[i, :] = pref*Pv[i, :]

    return kks, EK

##
## function to compute GW spectrum time growth using the approximation
## introduced in the first sound-shell model analysis of HH19
##
## The resulting GW spectrum is
##
##  Omega_GW (k) = (3pi)/(8cs) x (k/kst)^2 x (K/KK)^2 x TGW x Omm(k)
##

def OmGW_ssm_HH19(k, EK, Np=3000, Nk=60, plot=False, cs2=cs2):

    cs = np.sqrt(cs2)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

    p_inf = kp*(1 - cs)/2/cs
    p_sup = kp*(1 + cs)/2/cs

    Omm = np.zeros(len(kp))
    for i in range(0, len(kp)):

        p = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
        ptilde = kp[i]/cs - p
        z = -kp[i]*(1 - cs2)/2/p/cs2 + 1/cs

        EK_p = np.interp(p, k, EK)
        EK_ptilde = np.interp(ptilde, k, EK)

        Omm1 = (1 - z**2)**2*p/ptilde**3*EK_p*EK_ptilde
        Omm[i] = np.trapz(Omm1, p)

    return kp, Omm