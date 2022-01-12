#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 18:04:52 2022
partial joint_diag approximate joint diagonalization method
   Code by Fredy Vides
   For Paper, "Quadratic pseudospectrum and optimal joint approximate eigenvectors"
   by T. Loring, F. Vides
@author: Fredy Vides

Code based on the Matlab program joint_diag.m by
 J. F. Cardoso
"""

def P_Joint_Diag(A,lambda0,sigma0,L,delta):
    from numpy import identity,kron
    from scipy.sparse.linalg import eigs
    from numpy.linalg import eig
    from joint_diag import joint_diag
    m = A.shape[0]
    mn = A.shape[1]
    n=int(mn/m)
    F=identity(m)
    A=A.astype(complex)
    A=A-kron(lambda0,F)
    C=A@A.conj().T
    C=(C+C.conj().T)/2
    E=identity(n)
    if L<(m-1):
        V=eigs(C,k=L, M=None, sigma=None, which=sigma0, v0=None, ncv=None, maxiter=None, tol=delta)[1]
    else:
        L=m
        V=eig(C)[1]
    C0=V.conj().T@A@kron(E,V)
    for j in range(n):
        D0=C0[:,j*L:(j+1)*L].astype(complex)
        D0=(D0+D0.conj().T)/2
        C0[:,j*L:(j+1)*L]=D0
    W = joint_diag(C0,delta)[0]
    V=V@W
    return V