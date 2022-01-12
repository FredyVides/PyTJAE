#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 18:04:52 2022
partial joint_diag approximate joint diagonalization method
   Code by Fredy Vides
   For Paper, "Quadratic pseudospectrum and optimal joint approximate eigenvectors"
   by T. Loring, F. Vides
@author: Fredy Vides

Code based on the program jade.m
 by J.-F. Cardoso.
"""

def CumulantEstimate(X,m=0):
    from numpy import argsort,ones,sqrt,mean,zeros,reshape,conj,diag,dot
    from numpy.linalg import eig,inv
    from scipy.linalg import sqrtm
    n = X.shape[0]
    T = X.shape[1]
    if m==0:
        m=n
    nem = m
    if m<n:
        D,U = eig((X@X.conj().T)/T)
        k = argsort(D)
        puiss = D[k]
        ibl = sqrt(puiss[(n-m):n]-mean(puiss[0:(n-m)]))
        bl 	= ones(m)/ibl
        W	= diag(bl)@U[:,k[(n-m):n]].conj().T
        IW 	= U[:,k[(n-m):n]]@diag(ibl)
    else:
        IW 	= sqrtm((X@X.conj().T)/T)
        W	= inv(IW)
    Y = W@(X-reshape(mean(X,1),(n,1))@ones((1,T)))
    
    R	= (Y@Y.conj().T)/T
    C	= (Y@Y.T)/T
    
    Yl	= zeros((1,T))
    Ykl	= zeros((1,T))
    Yjkl = zeros((1,T))
    Q = zeros((m*m*m*m,1))
    Q = Q.astype(complex)
    index = 0
    
    for lx in range(m):
        Yl = Y[lx,:]
        for kx in range(m):
            Ykl = Yl*conj(Y[kx,:])
            for jx in range(m):
                Yjkl = Ykl*conj(Y[jx,:])
                for ix in range(m):
                    Q[index] = dot(Yjkl,Y[ix,:])/T -  R[ix,jx]*R[lx,kx] -  R[ix,kx]*R[lx,jx] -  C[ix,lx]*conj(C[jx,kx])
                    index = index + 1
                    
    D,U	= eig(reshape(Q,(m*m,m*m)).T)
    K = argsort(abs(D))
    la = D[K]
    
    C = zeros((m,nem*m))
    C = C.astype(complex)
    h	= m*m-1
    for u in range(0,nem*m,m):
        Z = reshape(U[:,K[h]],(m,m)).T
        C[:,u:(u+m)]	= la[h]*Z
        h = h-1
    
    return C,IW