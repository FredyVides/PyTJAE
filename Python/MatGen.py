#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:24:58 2022
Generator of Commuting State Transition Matrices
@author: fredy
"""
def MatGen(L=1001,N=10,S=40):
    from numpy import sin,cos,pi,linspace,zeros,identity
    from scipy.linalg import pinv, hankel
    
    t = linspace(0,1.5,L)
    y1 = sin(2*pi*t)
    y2 = 0.7*y1 + 0.4*cos(6*pi*t)
    H1 = hankel(y1[:N],y1[(N-1):S])
    H2 = hankel(y2[:N],y2[(N-1):S])
    H10 = H1[:,:-2]
    H20 = H2[:,:-2]
    H11 = H1[:,1:-1]
    H21 = H2[:,1:-1]
    H1 = hankel(y1[:N],y1[(N-1):])
    H2 = hankel(y2[:N],y2[(N-1):])
    E = identity(N)
    C = H11@pinv(H10)
    C1 = zeros((2*N,2*N))
    C1[:(2*N-1):2,:(2*N-1):2] = C
    C1[1:(2*N):2,1:(2*N):2] = E
    C = H21@pinv(H20)
    C2 = zeros((2*N,2*N))
    C2[:(2*N-1):2,:(2*N-1):2] = E
    C2[1:(2*N):2,1:(2*N):2] = C
    H2d = zeros((2*N,H1.shape[1]))
    H2d[:(2*N-1):2,:] = H1
    H2d[1:(2*N):2,:] = H2
    return C1,C2,H2d