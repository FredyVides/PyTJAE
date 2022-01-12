#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
SPARPREDICTOR  Sparse AR predictor for time series modeling
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""

def SPARPredictor(A,h,N):
    from numpy import append
    y=h
    L = len(h)
    for k in range(N):
        y = append(y,A.dot(h))
        h = y[k+1:L+k+1]
    return y