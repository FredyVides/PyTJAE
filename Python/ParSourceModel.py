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
            from ParSourceModel import ParSourceModel
            from DataView import DataView
            from SyntheticSignals import SyntheticSignals
            from ErrorRandomnessTest import ErrorRandomnessTest
            X = SyntheticSignals(1e-1)
            X,T,Xr,Sr,C,IW,V=drJADE(X,22,18,1e-2)
            Lags,Ar,Ai,Sp = ParSourceModel(Sr,[0,Sr.shape[0]],0.5,1e-1,1e-3)
            Xp = IW@V@Sp+T
            DataView(X,real(Xp),Sp,[0,1])
            ErrorRandomnessTest([0,1],X,Xp)
            
            from numpy import real
            from pandas import read_csv
            from drJADE import drJADE
            from ParSourceModel import ParSourceModel
            from DataView import DataView
            from ErrorRandomnessTest import ErrorRandomnessTest
            X = read_csv("../DataSets/X.csv",header=None).values
            X,T,Xr,Sr,C,IW,V=drJADE(X,22,18,1e-2)
            Lags,Ar,Ai,Sp = ParSourceModel(Sr,[0,Sr.shape[0]],0.5,1e-1,1e-3)
            Xp = IW@V@Sp+T
            DataView(X,real(Xp),Sp,[0,1])
            ErrorRandomnessTest([0,1],X,Xp)
"""

def ParSourceModel(Sr,idx,sampling_proportion,delta,tol):
    from numpy import real,imag,zeros
    from LagEstimate import LagEstimate
    from SPARPredictor import SPARPredictor
    from time import time
    from multiprocessing import Pool
    from SpARModelComputation import SpARModelComputation
    
    data = Sr[idx[0]:idx[1],:]
    N = data.shape[0]
    Ld = data.shape[1]
    
    Sp = zeros((N,Ld))
    Sp = Sp.astype(complex)
    
    Lags = []
    steps = []
    Mdr =  []
    Mdi = []
    mdr = []
    mdi = []
    
    start = time()
    RModelData = []
    IModelData = []
    for k in range(N):
        Lag = LagEstimate(real(data[k,:].copy()),10)
        steps.append(Ld-Lag)
        Lags.append(Lag)
        
        xr = real(data[k,:].copy())
        xi = imag(data[k,:].copy())
        mdr0 = xr.min()
        mdi0 = xi.min()
        Mdr0 = abs(xr - mdr0).max()
        Mdi0 = abs(xi - mdi0).max()
        
        Mdr.append(Mdr0)
        mdr.append(mdr0)
        Mdi.append(Mdi0)
        mdi.append(mdi0)
        
        xsr = 2*(xr-mdr0)/Mdr0-1
        xsi = 2*(xi-mdi0)/Mdi0-1
        
        RModelData.append((xsr,sampling_proportion,Lag,delta,tol))
        IModelData.append((xsi,sampling_proportion,Lag,delta,tol))
    print("Model data generation elapsed time: ",time()-start)
    
    start = time()
    with Pool(8) as p:
        RModel = p.starmap(SpARModelComputation, RModelData)
        IModel = p.starmap(SpARModelComputation, IModelData)
    print("Model computation elapsed time: ",time()-start)
    
    start = time()
    for k in range(N):
        yr = Mdr[k]*(SPARPredictor(RModel[k][0],RModel[k][1],steps[k])+1)/2+mdr[k]
        yi = Mdi[k]*(SPARPredictor(IModel[k][0],IModel[k][1],steps[k])+1)/2+mdi[k]
        Sp[k,:] = yr + 1j*yi
    print("Signal reconstruction elapsed time: ",time()-start)
    
    return Lags,RModel,IModel,Sp