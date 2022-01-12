#!/usr/bin/env python3
"""
Created on Sat Jan  1 18:04:52 2022
joint_diag approximate joint diagonalization method
   Code by Fredy Vides
   For Paper, "Quadratic pseudospectrum and optimal joint approximate eigenvectors"
   by T. Loring, F. Vides
@author: Fredy Vides

This code is based on the matlab program joint_diag.m by J.-F. Cardoso whose
description is cited below:
    
 Joint approximate diagonalization
 
 Joint approximate of n (complex) matrices of size m*m stored in the
 m*mn matrix A by minimization of a joint diagonality criterion

 Usage:  [ V , D ] =  joint_diag(A,jthresh)

 Input :
 * the m*nm matrix A is the concatenation of n matrices with size m
   by m. We denote A = [ A1 A2 .... An ]
 * threshold is an optional small number (typically = 1.0e-8 see the M-file).

 Output :
 * V is an m*m unitary matrix.
 * D = V'*A1*V , ... , V'*An*V has the same size as A and is a
   collection of diagonal matrices if A1, ..., An are exactly jointly
   unitarily diagonalizable.


 The algorithm finds a unitary matrix V such that the matrices
 V'*A1*V , ... , V'*An*V are as diagonal as possible, providing a
 kind of `average eigen-structure' shared by the matrices A1 ,...,An.
 If the matrices A1,...,An do have an exact common eigen-structure ie
 a common orthonormal set eigenvectors, then the algorithm finds it.
 The eigenvectors THEN are the column vectors of V and D1, ...,Dn are
 diagonal matrices.
 
 The algorithm implements a properly extended Jacobi algorithm.  The
 algorithm stops when all the Givens rotations in a sweep have sines
 smaller than 'threshold'.

 In many applications, the notion of approximate joint
 diagonalization is ad hoc and very small values of threshold do not
 make sense because the diagonality criterion itself is ad hoc.
 Hence, it is often not necessary in applications to push the
 accuracy of the rotation matrix V to the machine precision.

 PS: If a numrical analyst knows `the right way' to determine jthresh
     in terms of 1) machine precision and 2) size of the problem,
     I will be glad to hear about it.
 

 This version of the code is for complex matrices, but it also works
 with real matrices.  However, simpler implementations are possible
 in the real case.


----------------------------------------------------------------
 Version 1.2

 Copyright 	: Jean-Francois Cardoso. 
 Author 	: Jean-Francois Cardoso. cardoso@iap.fr
 Comments, bug reports, etc are welcome.
----------------------------------------------------------------

 Revision history

 Version 1.2.  Nov. 2, 1997.
   o some Matlab tricks to have a cleaner code.
   o Changed (angles=sign(angles(1))*angles) to (if angles(1)<0 ,
   angles= -angles ; end ;) as kindly suggested by Iain Collings
   <i.collings@ee.mu.OZ.AU>.  This is safer (with probability 0 in
   the case of sample statistics)

 Version 1.1.  Jun. 97.
 	Made the code available on the WEB




----------------------------------------------------------------
 References:

 The 1st paper below presents the Jacobi trick.
 The second paper is a tech. report the first order perturbation
 of joint diagonalizers


@article{SC-siam,
  author       = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
  journal      = "{SIAM} J. Mat. Anal. Appl.",
  title 	= "Jacobi angles for simultaneous diagonalization",
  pages 	= "161--164",
  volume       = "17",
  number       = "1",
  month 	= jan,
  year 	= {1996}
  }



@techreport{PertDJ,
  author       = "Jean-Fran\c{c}ois Cardoso",
  institution  = "T\'{e}l\'{e}com {P}aris",
  title        = "Perturbation of joint diagonalizers. Ref\# 94D027",
  year	        = "1994"
}
"""

def joint_diag(A,jthresh):
    from numpy import array,zeros,identity,real,argsort,sqrt,conj
    from numpy.linalg import eig
    m = A.shape[0]
    nm = A.shape[1]
    A = A.astype(complex)
    B = array([[1,0,0],[0,1,1],[0,-1j,1j]])
    Bt = B.conj().T
    Ip = zeros((1,nm))
    Iq = zeros((1,nm))
    n = int(nm/m)
    g = zeros((3,n))
    g = g.astype(complex)
    G = zeros((2,2))
    vcp = zeros((3,3))
    D  = zeros((3,3))
    K = zeros((3,3))
    angles = zeros((3,1))
    pair = zeros((1,2))
    c = 0
    s = 0
    V = identity(m)
    V = V.astype(complex)
    encore	= 1
    while encore:
        encore=0
        for p in range(1,m):
            Ip = range(p-1,nm,m)
            for q in range(p+1,m+1):
                Iq = range(q-1,nm,m)
                g[0,:] = A[p-1,Ip]-A[q-1,Iq]
                g[1,:] = A[p-1,Iq]
                g[2,:] = A[q-1,Ip]
                D,vcp = eig(real(B@(g@g.conj().T)@Bt))
                K = argsort(D)
                angles  = vcp[:,K[2]]
                if angles[0]<0:
                    angles = -angles
                c = sqrt(0.5+angles[0]/2)
                s = 0.5*(angles[1]-1j*angles[2])/c
                if abs(s)>jthresh:
                    encore = 1
                    pair = array([p-1,q-1])
                    G = array([[c,-conj(s)],[s,c]])
                    V[:,pair] = V[:,pair]@G
                    A[pair,:] = G.conj().T @ A[pair,:]
                    AIp = A[:,Ip]
                    AIq = A[:,Iq]
                    A[:,Ip]=c*AIp+s*AIq
                    A[:,Iq]=-conj(s)*AIp+c*AIq
    D = A
    return V,D