#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
LAGESTIMATE ACF-FFT based sparse lag estimator
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
def LagEstimate(data,L0):
    from statsmodels.tsa.stattools import acf 
    from scipy.signal import find_peaks
    from numpy import where
    L = acf(data,adjusted=False,nlags=len(data),fft=True)
    p = find_peaks(L)
    if len(p[0])>0:
        q=where(L[p[0]]==L[p[0]].max())
        L = max(p[0][q[0][0]],L0)
    else:
        L = L0
    return L