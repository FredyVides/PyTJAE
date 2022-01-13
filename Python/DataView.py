""" Example: 
    from pandas import read_csv
    from drJADEDemo import drJADEDemo
    from DataView import DataView
    from numpy import real
    X = read_csv("../DataSets/X.csv",header=None).values
    X,T,Xr,Sr,C,IW,V=drJADEDemo(X,22,18,1e-2)
    DataView(X,real(Xr),Sr,[0,1])
"""
def DataView(X,Y,S,idx):
    from matplotlib.pyplot import subplot,figure,plot,show
    from numpy import real,imag
    fig_0=figure(1)
    subplot(221)
    plot(X[idx[0],:]),plot(Y[idx[0],:],'r')
    subplot(222)
    plot(real(X[idx[0],:]-Y[idx[0],:]))
    subplot(223)
    plot(X[idx[1],:]),plot(Y[idx[1],:],'r')
    subplot(224)
    plot(real(X[idx[1],:]-Y[idx[1],:]))
    show()
    fig_1=figure(2)
    subplot(221)
    plot(real(S[idx[0],:]))
    subplot(222)
    plot(imag(S[idx[0],:]))
    subplot(223)
    plot(real(S[idx[1],:]))
    subplot(224)
    plot(imag(S[idx[1],:]))
    show()
    fig_0.savefig('fig_results_summary_0'+str(1)+'.png',dpi=600,format='png')
    fig_1.savefig('fig_results_summary_0'+str(2)+'.png',dpi=600,format='png')