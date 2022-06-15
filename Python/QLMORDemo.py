#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:19:01 2022
QLMOR Simulation
@author: doctor
"""
def QLMORDemo(tol = 1e-5):
    from MatGen import MatGen
    from QLMOR import QLMOR
    from numpy import zeros, real
    from matplotlib.pyplot import subplots, tight_layout, show
    
    K1,K2,H = MatGen(1500,200,500)
    V,k1,k2 = QLMOR(K1,K2,tol)
    
    h0 = zeros((H.shape[0],H.shape[1]))
    h1 = 1j*zeros((V.shape[1],H.shape[1]))
    
    h0[:,0] = H[:,0]
    h1[:,0] = V.conj().T@H[:,0]
    
    for k in range(H.shape[1]-1):
        h0[:,k+1] = K2@(K1@h0[:,k])
        h1[:,k+1] = k2@(k1@h1[:,k])
        
    h1 = real(V@h1)
    fig,axs = subplots(1,1)
    axs.plot(h0[0,:],h0[1,:])
    axs.plot(h1[0,:],h1[1,:],"r--")
    axs.axis('square')
    axs.grid(True)
    axs.set_xlabel('$y_1$')
    axs.set_ylabel('$y_2$')
    axs.legend(['System output','ROM output'],loc = 'lower center')
    tight_layout()
    show()
    fig.savefig('fig_MOR_Simulation.png',dpi=600,format='png')
    
    return V,k1,k2,h1[:,0]
    