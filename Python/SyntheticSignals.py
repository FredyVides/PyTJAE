# Example: X=SyntheticSignals(1e-1)
def SyntheticSignals(noise):
    from numpy import sin,cos,reshape,linspace,pi,zeros
    from numpy.random import randn
    t = linspace(0,30,15000)
    X = reshape(range(10),(10,1))@reshape(t,(1,15000))
    Y = zeros((50,15000))
    Y[:10,:] = sin(pi*X)
    Y[10:20,:] = cos(pi*X)
    Y[20:30,:] = sin(pi*(X+20))
    Y[30:40,:] = cos(pi*(X+20))
    Y[40:44,:] = sin(pi*(X[:4,:]+10))
    Y[44:48,:] = cos(pi*(X[:4,:]+10))
    Y[48:,:] = noise*randn(2,15000)
    A=zeros((50,50))
    A[:,:5] = randn(50,5)
    A[:,5:10] = noise*randn(50,5)
    A[:,10:15] = randn(50,5)
    A[:,15:20] = noise*randn(50,5)
    A[:,20:23] = randn(50,3)
    A[:,23:25] = noise*randn(50,2)
    A[:,25:27] = randn(50,2)
    A[:,27:] = randn(50,23)*noise
    X = A@Y
    return X