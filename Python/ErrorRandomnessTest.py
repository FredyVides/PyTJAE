#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:38:39 2022
Randomness Test Function for Errors
@author: FredyVides
"""

def ErrorRandomnessTest(idx,X,Xp):
    from matplotlib.pyplot import hist,subplot,figure,grid,boxplot,semilogy
    from scipy.stats import normaltest
    from numpy import real
    from randtest import random_score
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    Noise = real(X[idx[0],:]-Xp[idx[0],:])
    k1,p0 = normaltest(Noise)
    fig_0 = figure(1)
    subplot(211)
    h0 = hist(Noise,bins=30,density=1,alpha=0.6, color='b')
    print('Randomness test result: ',random_score(Noise))
    print('Normality test p-value: ',p0)
    Noise = real(X[idx[1],:]-Xp[idx[1],:])
    k1,p0 = normaltest(Noise)
    subplot(212)
    h0 = hist(Noise,bins=30,density=1,alpha=0.6, color='b')
    print('Randomness test result: ',random_score(Noise))
    print('Normality test p-value: ',p0)
    rmse = []
    for k in range(X.shape[0]):
        rmse.append(sqrt(mean_squared_error(real(X[k,:]), real(Xp[k,:]))))
    fig_0.savefig('fig_results_summary_0'+str(3)+'.png',dpi=600,format='png')
    fig_1 = figure(2)
    subplot(211)
    semilogy(range(1,X.shape[0]+1),rmse),semilogy(range(1,X.shape[0]+1),rmse,'ro'),grid('on')
    subplot(212),boxplot(rmse),grid('on')
    fig_1.savefig('fig_results_summary_0'+str(4)+'.png',dpi=600,format='png')