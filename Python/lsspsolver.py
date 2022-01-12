#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
LSSPSOLVER  Sparse linear least squares solver
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""

def lsspsolver(A,Y,L=100,tol=1e-2,delta=1e-2):
    from numpy.linalg import svd,lstsq,norm
    from numpy import zeros,dot,diag,argsort,inf
    N=A.shape[1]
    X=zeros((N,))
    u,s,v=svd(A,full_matrices=0)
    rk=sum(s>tol)
    u=u[:,:rk]
    s=s[:rk]
    s=1/s
    s=diag(s)
    v=v[:rk,:]
    A=dot(u.conj().T,A)
    Y=dot(u.conj().T,Y)
    X0=dot(v.conj().T,dot(s,Y))
    w=zeros((N,))
    K=1
    Error=1+tol
    c=X0
    x0=c
    ac=abs(c)
    f=argsort(-ac)
    N0=max(sum(ac[f]>delta),1)
    while (K<=L) & (Error>tol):
        ff=f[:N0]
        X=w
        c = lstsq(A[:,ff],Y,rcond=None)[0]
        X[ff]=c
        Error=norm(x0-X[:],inf)
        x0=X
        ac=abs(x0)
        f=argsort(-ac)
        N0=max(sum(ac[f]>delta),1)
        K=K+1
    return X