#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:00:46 2022

@author: doctor
"""

def SpARModelComputation(xs,sampling_proportion,Lag,delta,tol):
    from SpAutoRegressor import SpAutoRegressor
    r = []
    A,h_0 = SpAutoRegressor(xs,1/len(xs),sampling_proportion,1,Lag,delta,tol)
    r.append(A)
    r.append(h_0)
    return r