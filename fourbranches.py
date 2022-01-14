import numpy as np

def f(X):
    return mean(X) + np.random.normal(0, std(X))

def mean(X, deviation=5): 
    X = np.array(X)
    if X.ndim == 1:
        X=np.atleast_2d(X)
    X1 = 3 + 0.1*(X[:,0]-X[:,1])**2 - (X[:,0]+X[:,1])/np.sqrt(2)
    X2 = 3 + 0.1*(X[:,0]-X[:,1])**2 + (X[:,0]+X[:,1])/np.sqrt(2)
    X3 = (X[:,0]-X[:,1]) + 6/np.sqrt(2)
    X4 = (X[:,1]-X[:,0]) + 6/np.sqrt(2)
    mid=np.stack((X1,X2,X3,X4)) 
    return -np.amin(mid,axis=0) + deviation

def std(X): 
    return mean(X) / 6