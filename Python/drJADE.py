""" Example: 
    from pandas import read_csv
    from drJADE import drJADE
    X = read_csv("../DataSets/X.csv",header=None).values
    X,T,Xr,Sr,C,IW,V=drJADE(X,50,18,1e-2)
    DataView(X,real(Xr),Sr,[0,1])
"""
def drJADE(X,m,L,tol):
    from numpy import reshape,zeros,mean,ones
    from numpy.linalg import lstsq
    from CumulantEstimate import CumulantEstimate
    from P_Joint_Diag import P_Joint_Diag
    from time import time
    start = time()
    C,IW=CumulantEstimate(X,m)
    end=time()
    print("Cumulant estimate runtime: ",end-start)
    start = time()
    V=P_Joint_Diag(C,zeros((1,m)),'LM',L,tol)
    end = time()
    print("Approximate joint diagonalization runtime: ",end-start)
    Sr = V.conj().T@lstsq(IW,X,rcond=None)[0]
    T = reshape(mean(X,1),(X.shape[0],1))@ones((1,X.shape[1]))
    Xr = IW@V@Sr+T
    return X,T,Xr,Sr,C,IW,V