import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.base import clone  
import scipy.optimize


class VHGP(object):
    '''
    This is a Python version of Variational Heteroscedastic Gaussian Process 
    Regression following 
    Variational Heteroscedastic Gaussian Process Regression (2011) 
    Miguel Lazaro-Gredilla && Michalis K. Titsias
    
    Parameters
    ---------
    kernelf : kernel instance 
        The (initial) kernel specifying the covariance function of the mean 
        process 
    kernelg : kernel instance 
        The (initial) kernel specifying the covariance function of the variance 
        (of noise) process 
    
    Attributes
    ----------
    X_train_ :  array-like of shape (n_samples, n_features)
        Input of training data
    
    y_train_ :  array-like of shape (n_samples,)
        Output of training data
        
    kernelf_ : kernel instance 
        The kernel used for training and prediction of the mean process 
    
    kernelg_ : kernel instance 
        The kernel used for training and prediction of the variance (of noise) 
        process 
    
    n : int 
        The number of samples 
        
    logA : array-like of shape (n_samples, )
        The diagnal array of log Lambda matrix
    
    mu0 : float
        The hyper mean for variance process
    
    '''
    
    def __init__(self, kernelf, kernelg):
        self.kernelf = kernelf
        self.kernelg = kernelg
        
    def fit(self, X, y, fixhyp = 0): 
        """Fit VHGP model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) 
            Target values
        fixhyp: int 
            0-  Nothing is fixed (all variational parameters and hyperparameters 
                are learned).
            1 - Hyperparameters for f and g are fixed.
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        self.kernelf_ = clone(self.kernelf) 
        self.kernelg_ = clone(self.kernelg) 
       
        
        self.X_train_ = np.copy(X)  
        self.y_train_ = np.copy(y)  
        self.n = X.shape[0]
        bounds = np.concatenate((np.repeat([[-6,2]], self.n, axis = 0), 
                                 self.kernelf.bounds, self.kernelg.bounds, 
                                 np.array([[-6,4]])), axis = 0) 
        
        if fixhyp == 0:
            initial_para = np.concatenate(([np.log(0.5)]*self.n, self.kernelf_.theta, self.kernelg_.theta, [np.log(np.var(self.y_train_)/6)]))
            optimal = scipy.optimize.minimize(self.ELBOFull, initial_para, method = "L-BFGS-B", jac = True, bounds = bounds, options={'gtol': 1e-2})
            self.logA = optimal.x[0 : self.n]    
            self.kernelf_.theta = optimal.x[self.n : self.n + self.kernelf_.theta.shape[0]]
            self.kernelg_.theta = optimal.x[self.n + self.kernelf_.theta.shape[0] : self.n + 2 * self.kernelf_.theta.shape[0]]
            self.mu0 = optimal.x[self.n + 2 * self.kernelf_.theta.shape[0]]    
            self.optimal = optimal
        
        if fixhyp == 1:
            initial_para = np.concatenate(([np.log(0.5)]*self.n, [np.var(self.y_train_)/6]))
            args = (self.kernelf_(self.X_train_), self.kernelg_(self.X_train_))
            optimal = scipy.optimize.minimize(self.ELBOFixhyp, initial_para, args, method="BFGS", jac=True, options={'gtol': 1e-3})
            self.logA = optimal.x[0 : self.n]    
            self.mu0 = optimal.x[self.n]
            self.optimal = optimal
            
        return self
        
    def predict(self, X):
        '''
        Predicting using the optimized logA, loghyper and mu0
        
        Parameters
        ----------
        X : array-like of shape (samples, dimensions)
        
        Returns
        ---------
        f_mean : array-like of shape (samples, )
            Predictive mean for mean process 
        f_var: 
            Predictive variance for mean process
        g_mean
            Predictive mean for variance of noise process
        g_var
            Predictive variance variance of noise process
        
        '''
        
        K_trans_f = self.kernelf_(X, self.X_train_)
        K_trans_g = self.kernelg_(X, self.X_train_)
        
        Kf = self.kernelf_(self.X_train_)  # Matrix 
        Kg = self.kernelg_(self.X_train_)  # Matrix 
        
        A = np.exp(self.logA)
        sqA = np.sqrt(A)  
        Basic = np.eye(self.n) + Kg * np.outer(sqA, sqA)   # (n,n) 
        cinvB = solve_triangular(cholesky(Basic,lower=True), np.eye(self.n), lower=True) #(n,n)
        cinvBs = cinvB * np.outer(np.ones(self.n), sqA)  #(n,n)
        beta = A - 0.5                            # (n,)
        mu = Kg @ beta + self.mu0                      # (n,)

        hBLK2 = cinvBs @ Kg                       # (n,n)
        Sigma = Kg - hBLK2.T @ hBLK2              # (n,n) 

        R = np.exp(mu-np.diag(Sigma)/2)         # (n,) 
        Lf = cholesky(Kf + np.diag(R),lower=True) # (n,n)
        alpha = cho_solve((Lf, True), self.y_train_)           # (n,)  
        invKfRs = cho_solve((Lf, True), np.eye(self.n))  
    
        f_mean = K_trans_f @ alpha                #(n,)
        f_var = self.kernelf_.diag(X) - np.einsum('ij,ij->i', K_trans_f @ invKfRs, K_trans_f)
        g_mean = K_trans_g @ beta + self.mu0
        g_var = self.kernelg_.diag(X) - np.einsum('ij,ij->i', K_trans_g @ (cinvBs.T @ cinvBs), K_trans_g)
        
        return f_mean, f_var, g_mean, g_var

    def ELBOFull(self, para): 
        
        '''
        Compute the ELBO of parameters for training. The hyperparameters are also need to optimized
        
        Parameters
        ----------
        para: array-like of shape (n+2+2+1,)
            [logA, hyperf, hyperg, mu] 
        '''
        
        A = np.exp(para[0 : self.n])                               # vector 
        self.kernelf_.theta = para[self.n : self.n + self.kernelf_.theta.shape[0]]            # vector 
        self.kernelg_.theta = para[self.n + self.kernelf_.theta.shape[0] : self.n + 2 * self.kernelf_.theta.shape[0]]        # vector 
        mu0 = para[self.n + 2 * self.kernelf_.theta.shape[0]]                                     # vector 
        # Theta 是log 以后的 hyperparameter 值
        
        Kf, KfD = self.kernelf_(self.X_train_, eval_gradient=True)  # Matrix 
        Kg, KgD = self.kernelg_(self.X_train_, eval_gradient=True)  # Matrix 
        
        Kf = Kf + np.eye(self.n) * 1e-4
        Kg = Kg + np.eye(self.n) * 1e-4
        
        KfD0 , KfD1 = KfD[:,:,0], KfD[:,:,1]       
        KgD0 , KgD1 = KgD[:,:,0], KgD[:,:,1]              # two hyperparameters gradient w.r.t. log theta

        sqA = np.sqrt(A)  
        Basic = np.eye(self.n) + Kg * np.outer(sqA, sqA)   # (n,n) 
        cinvB = solve_triangular(cholesky(Basic,lower=True), np.eye(self.n), lower=True) #(n,n)
        cinvBs = cinvB * np.outer(np.ones(self.n), sqA)  #(n,n)
        beta = A - 0.5                            # (n,)
        mu = Kg @ beta + mu0                      # (n,)

        hBLK2 = cinvBs @ Kg                       # (n,n)
        Sigma = Kg - hBLK2.T @ hBLK2              # (n,n) 

        R = np.exp(mu-np.diag(Sigma)/2) 
        self.At = A
        self.Sigmat = Sigma
        self.mut = mu# (n,) 
        self.Rt = R
        self.Kft = Kf
        #Lf = np.linalg.cholesky(Kf + np.diag(R)) # (n,n)
        Lf = cholesky(Kf + np.diag(R),lower=True) # (n,n)
        alpha = cho_solve((Lf, True), self.y_train_)           # (n,)  
 
        Fv1 = - 0.5 * np.dot(alpha,self.y_train_) - np.sum(np.log(np.diag(Lf))) 
        Fv2 =  np.sum(np.log(np.diag(cinvB))) - 0.5 * np.sum(cinvB ** 2) - 0.5 * np.dot(beta, mu-mu0)
        Fv = Fv1 + Fv2 - 0.25 * np.trace(Sigma) 
        #--------------derivatives--------------------------------------# 
        
        # Preparations
        invKfRs = cho_solve((Lf, True), np.eye(self.n))    # (n,n)
        betahat = -0.5 * (np.diag(invKfRs) * R- alpha**2 * R) # (n,)
        Ahat =  betahat  + 0.5                                # (n,) 

        # Derivative A 
        W = - (Kg + 0.5 * Sigma**2) @ (A-Ahat)    # (n,)
        DifA = A * W                           

        # Derivative mu
        Difmu0 = np.sum(betahat)                   
        
        # Derivative hyperf 
        W = np.outer(alpha,alpha) - invKfRs
        Difhyperf = [np.sum(KfD0 * W)/2, np.sum(KfD1 * W)/2]  
        
        # Derivative hyperg 
        invBs = cinvB.T @ cinvBs   
        W = np.outer(beta,beta) + 2 * np.outer((betahat- beta), beta) - \
        invBs.T @ (np.outer(Ahat/A-1, np.ones(self.n)) * invBs) - cinvBs.T @ cinvBs 
        Difhyperg = [np.sum(KgD0 * W)/2, np.sum(KgD1 * W)/2]         

        return -Fv, -np.concatenate((DifA, Difhyperf, Difhyperg, [Difmu0]))

    def ELBOFixhyp(self, para, Kf, Kg): 
        
        '''
        Compute the ELBO of parameters for training. The hyperparameters are fixed.
        
        Parameters
        ----------
        para: array-like of shape (n+1,)
            [logA, mu] 
        '''

        A = np.exp(para[0 : self.n])                               # vector 
        mu0 = para[self.n]                                     # vector 
        
        sqA = np.sqrt(A)                            # (n,)
        Basic = np.eye(self.n) + Kg * np.outer(sqA, sqA)   # (n,n) 
        cinvB = solve_triangular(cholesky(Basic,lower=True), np.eye(self.n), lower=True) #(n,n)
        cinvBs = cinvB * np.outer(np.ones(self.n), sqA)  #(n,n)
        beta = A - 0.5                            # (n,)
        mu = Kg @ beta + mu0                      # (n,)

        hBLK2 = cinvBs @ Kg                       # (n,n)
        Sigma = Kg - hBLK2.T @ hBLK2              # (n,n) 

        R = np.exp(mu-np.diag(Sigma)/2)           # (n,) 
        Lf = cholesky(Kf + np.diag(R),lower=True) # (n,n)
        alpha = cho_solve((Lf, True), self.y_train_)           # (n,)  

        Fv1 = - 0.5 * np.dot(alpha,self.y_train_) - np.sum(np.log(np.diag(Lf))) 
        Fv2 =  np.sum(np.log(np.diag(cinvB))) - 0.5 * np.sum(cinvB ** 2) - 0.5 * np.dot(beta, mu-mu0)
        Fv = Fv1 + Fv2 - 0.25 * np.trace(Sigma) 
        
        #--------------derivatives--------------------------------------# 
        
        # Preparations
        invKfRs = cho_solve((Lf, True), np.eye(self.n))    # (n,n)
        betahat = -0.5 * (np.diag(invKfRs) * R- alpha**2 * R) # (n,)
        Ahat =  betahat  + 0.5                                # (n,) 

        # Derivative A 
        W = - (Kg + 0.5 * Sigma**2) @ (A-Ahat)    # (n,)
        DifA = A * W                           

        # Derivative mu
        Difmu0 = np.sum(betahat)                   
        
        return -Fv, -np.concatenate((DifA, [Difmu0]))