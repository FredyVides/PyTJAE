#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:15:28 2022
SOURCEMODEL Source model computation using JAD based model order 
    reduction technology
    Code by Fredy Vides
    For Paper, "Computing Truncated Joint Approximate Eigenbases for 
    Model Order Reduction"
    by T. Loring, F. Vides
@author: Fredy Vides

# Examples:  
            from numpy import real
            from drJADE import drJADE
            from SourceModel import SourceModel
            from DataView import DataView
            from SyntheticSignals import SyntheticSignals
            from ErrorRandomnessTest import ErrorRandomnessTest
            X = SyntheticSignals(1e-1)
            X,T,Xr,Sr,C,IW,V=drJADE(X,22,18,1e-2)
            Lags,Ar,Ai,Sp = SourceModel(Sr,[0,Sr.shape[0]],0.15,1e-1,1e-3)
            Xp = IW@V@Sp+T
            DataView(X,real(Xp),Sp,[0,1])
            ErrorRandomnessTest([0,1],X,Xp)
            
            from numpy import real
            from pandas import read_csv
            from drJADE import drJADE
            from SourceModel import SourceModel
            from DataView import DataView
            from ErrorRandomnessTest import ErrorRandomnessTest
            X = read_csv("../DataSets/X.csv",header=None).values
            X,T,Xr,Sr,C,IW,V=drJADE(X,22,18,1e-2)
            Lags,Ar,Ai,Sp = SourceModel(Sr,[0,Sr.shape[0]],0.15,1e-1,1e-3)
            Xp = IW@V@Sp+T
            DataView(X,real(Xp),Sp,[0,1])
            ErrorRandomnessTest([0,1],X,Xp)
"""

def SourceModel(Sr,idx,sampling_proportion,delta,tol):
    from numpy import real,imag,zeros
    from LagEstimate import LagEstimate
    from SpAutoRegressor import SpAutoRegressor
    from SPARPredictor import SPARPredictor
    from time import time
    
    data = Sr[idx[0]:idx[1],:]
    N = data.shape[0]
    Ld = data.shape[1]
    
    Sp = zeros((N,Ld))
    Sp = Sp.astype(complex)
    
    Lags = []
    AR = []
    AI = []
    
    start = time()
    for k in range(N):
        Lag = LagEstimate(real(data[k,:].copy()),10)
        steps = Ld-Lag
        Lags.append(Lag)
        
        xr = real(data[k,:].copy())
        xi = imag(data[k,:].copy())
        mdr = xr.min()
        mdi = xi.min()
        Mdr = abs(xr - mdr).max()
        Mdi = abs(xi - mdi).max()
        xsr = 2*(xr-mdr)/Mdr-1
        xsi = 2*(xi-mdi)/Mdi-1
        
        Ar,h_0r = SpAutoRegressor(xsr,1/len(xsr),sampling_proportion,1,Lag,delta,tol)
        Ai,h_0i = SpAutoRegressor(xsi,1/len(xsi),sampling_proportion,1,Lag,delta,tol)
        
        AR.append(Ar)
        AI.append(Ai)
        
        yr = Mdr*(SPARPredictor(Ar,h_0r,steps)+1)/2+mdr
        yi = Mdi*(SPARPredictor(Ai,h_0i,steps)+1)/2+mdi
        
        Sp[k,:] = yr + 1j*yi
    print("Model computation and source reconstruction runtime: ",time()-start)
    return Lags,AR,AI,Sp