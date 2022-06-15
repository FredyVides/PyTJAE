#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 02:40:31 2022
MOR via Quadratic Localizer
@author: doctor
"""

def QLMOR(K1,K2,tol):
    from P_Joint_Diag import P_Joint_Diag
    from scipy.linalg import svd
    from numpy import hstack,zeros
    
    W1 = K1.T@K1
    W2 = K2.T@K2
    W3 = K1.T@K2+K2.T@K1
    
    A = hstack((W1,W2,W3))
    
    rk = svd(W3,full_matrices=0)[1]
    rk = sum(rk>tol)
    
    
    V = P_Joint_Diag(A,zeros((1,3)),'LM',rk,tol)
    k1 = V.conj().T@K1@V
    k2 = V.conj().T@K2@V
    
    return V,k1,k2