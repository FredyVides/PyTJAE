#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:38:39 2022
Randomness Test Function for Errors
@author: FredyVides
"""

def ErrorRandomnessTest(idx,X,Xp):
    from matplotlib.pyplot import hist,subplot,figure
    from scipy.stats import normaltest
    from numpy import real
    from randtest import random_score
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
    fig_0.savefig('fig_results_summary_0'+str(3)+'.png',dpi=600,format='png')