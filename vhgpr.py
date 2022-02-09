import numpy as np
import scipy.optimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.base import clone  


class VHGPR(object):
    '''
    This is a Python version of Variational Heteroscedastic Gaussian Process 
    Regression following 
    Variational Heteroscedastic Gaussian Process Regression (2011) 
    Miguel Lazaro-Gredilla && Michalis K. Titsias.
    
    Parameters
    ----------
    kernelf : instance of kernels in sklearn Gaussian process 
        The (initial) kernel specifying the covariance function of the mean 
        process.
    kernelg : instance of kernels in sklearn Gaussian process 
        The (initial) kernel specifying the covariance function of the variance 
        (of noise) process.
    n_restarts_optimizer : int
        The number of restart times in optimizing the ELBO (to obtain optimal
        hyperparameters).
    
    Attributes
    ----------
    X_train_ :  array-like of shape (n_samples, n_features)
        Input of training data.
    
    y_train_ :  array-like of shape (n_samples,)
        Output of training data.
        
    kernelf_ : instance of kernels in sklearn Gaussian process 
        The kernel used for training and prediction of the mean process.
    
    kernelg_ : instance of kernels in sklearn Gaussian process 
        The kernel used for training and prediction of the variance (of noise) 
        process.
    
    n : int 
        The number of samples.
        
    logA : array-like of shape (n_samples, )
        The diagnal array of log Lambda matrix.
    
    mu0 : float
        The hyper mean for variance process.
    
    '''
    
    def __init__(self, kernelf, kernelg, n_restarts_optimizer=0):
        self.kernelf = kernelf
        self.kernelg = kernelg
        self.n_restarts_optimizer = n_restarts_optimizer
        
    def fit(self, X, y): 
        """Find the optimal hyperparameters. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) 
            Target values.
        """

        whether_success=False

        self.kernelf_ = clone(self.kernelf) 
        self.kernelg_ = clone(self.kernelg)
       
        self.X_train_ = np.copy(X)  
        self.y_train_ = np.copy(y)  
        self.n = X.shape[0]
        bounds = np.concatenate((np.repeat([[-2,2]], self.n, axis = 0), 
                                 self.kernelf.bounds, self.kernelg.bounds, 
                                 np.array([[-2,4]])), axis = 0) 

        initial_para = np.concatenate(([np.log(0.5)]*self.n, 
                                        self.kernelf_.theta, 
                                        self.kernelg_.theta, 
                                        [np.log(np.var(self.y_train_)/6)]))     
        optimal = [self._optimizer(initial_para, bounds)]

        if self.n_restarts_optimizer > 1:
            for i in range(self.n_restarts_optimizer):
                initial_para_ = np.random.uniform(bounds[:, 0], bounds[:, 1])
                optimal.append(self._optimizer(initial_para_, bounds))
        
        # generate more staritng points if no converged results
        for jj in range(50): 
            if np.min([i.fun for i in optimal])==10000:
                print("fit one more time!")
                initial_para_ = np.random.uniform(bounds[:, 0], bounds[:, 1])
                optimal.append(self._optimizer(initial_para_, bounds))
            else: 
                whether_success=True
                break
        assert whether_success

        optimal = optimal[np.argmin([i.fun for i in optimal])]
        self.logA = optimal.x[0 : self.n]
        self.kernelf_.theta = optimal.x[self.n : self.n + 
                                        self.kernelf_.theta.shape[0]]
        self.kernelg_.theta = optimal.x[self.n +self.kernelf_.theta.shape[0] 
                                    : self.n + self.kernelf_.theta.shape[0] 
                                        + self.kernelg_.theta.shape[0]]
        self.mu0 = optimal.x[-1]

        return self

    def predict(self, X):
        '''
        Predict using the optimal hyperparameters
        
        Parameters
        ----------
        X : array-like of shape (samples, dimensions)
            Input of tested samples.
        
        Returns
        ----------
        f_mean : array-like of shape (samples, )
            Predictive mean for mean process.
        f_var : same as f_mean
            Predictive variance for mean process.
        g_mean : same as f_mean
            Predictive mean for variance of noise process.
        g_var : same as f_mean
            Predictive variance variance of noise process.
        
        '''
        
        K_trans_f = self.kernelf_(X, self.X_train_)
        K_trans_g = self.kernelg_(X, self.X_train_)
        
        Kf = self.kernelf_(self.X_train_)  # Matrix 
        Kg = self.kernelg_(self.X_train_)  # Matrix 
        
        A = np.exp(self.logA)
        sqA = np.sqrt(A)  
        Basic = np.eye(self.n) + Kg * np.outer(sqA, sqA)   # (n,n) 
        cinvB = solve_triangular(cholesky(Basic,lower=True), np.eye(self.n), 
                                 lower=True) #(n,n)
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
        f_var = self.kernelf_.diag(X) - np.einsum('ij,ij->i', 
                                             K_trans_f@invKfRs, K_trans_f)
        g_mean = K_trans_g @ beta + self.mu0
        g_var = self.kernelg_.diag(X) - np.einsum('ij,ij->i', 
                                K_trans_g @ (cinvBs.T @ cinvBs), K_trans_g)
        
        return f_mean, f_var, g_mean, g_var

    def ELBOFull(self, para): 
        '''
        Compute the (negative) ELBO and its derivative.

        Parameters
        ----------
        para : array-like of shape (number of hyperparameters,)
            [logA, hyperf, hyperg, mu].

        Returns
        ----------
        Fv : float 
            The ELBO
        DFv : array-like of shape (number of hyperparameters,)
            The derivative of ELBO to hyperparameters.
        '''
        
        A = np.exp(para[0 : self.n])                               # vector 
        self.kernelf_.theta = para[self.n : self.n+self.kernelf_.theta.shape[0]]
        self.kernelg_.theta = para[self.n + self.kernelf_.theta.shape[0] : 
                                   self.n + self.kernelf_.theta.shape[0]
                                          + self.kernelg_.theta.shape[0]]
        mu0 = para[-1]                                     # vector 
        
        Kf, KfD = self.kernelf_(self.X_train_, eval_gradient=True)  # Matrix 
        Kg, KgD = self.kernelg_(self.X_train_, eval_gradient=True)  # Matrix 
        
        Kf = Kf + np.eye(self.n) * 1e-4
        Kg = Kg + np.eye(self.n) * 1e-4
        

        sqA = np.sqrt(A)  
        Basic = np.eye(self.n) + Kg * np.outer(sqA, sqA)   # (n,n) 
        cinvB = solve_triangular(cholesky(Basic,lower=True), np.eye(self.n), 
                                                         lower=True) #(n,n)
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
        Fv2 =  (np.sum(np.log(np.diag(cinvB))) - 0.5 * np.sum(cinvB ** 2) 
                                               - 0.5 * np.dot(beta, mu-mu0))
        Fv = Fv1 + Fv2 - 0.25 * np.trace(Sigma) 
        #--------------derivatives--------------------------------------# 
        
        # Preparations
        invKfRs = cho_solve((Lf, True), np.eye(self.n))    # (n,n)
        betahat = -0.5 * (np.diag(invKfRs) * R- alpha**2 * R) # (n,)
        Ahat =  betahat  + 0.5                                # (n,) 

        # Derivative A 
        W = - (Kg + 0.5 * Sigma**2) @ (A-Ahat)    # (n,)
        DifA = A * W                              # because of the log

        # Derivative mu
        Difmu0 = np.sum(betahat)                   
        
        # Derivative hyperf 
        KfD_ = [KfD[:,:,i] for i in range(self.kernelf_.theta.shape[0])]
        W = np.outer(alpha,alpha) - invKfRs
        Difhyperf = [np.sum(KfD_[i] * W)/2 for i in 
                        range(self.kernelf_.theta.shape[0])]
        
        # Derivative hyperg 
        KgD_ = [KgD[:,:,i] for i in range(self.kernelg_.theta.shape[0])]
        invBs = cinvB.T @ cinvBs   
        W = (np.outer(beta,beta) + 2 * np.outer((betahat- beta), beta) 
             - invBs.T @ (np.outer(Ahat/A-1, np.ones(self.n)) * invBs) 
             - cinvBs.T @ cinvBs)
        Difhyperg = [np.sum(KgD_[i] * W)/2 for i in 
                        range(self.kernelg_.theta.shape[0])]      

        DFv = np.concatenate((DifA, Difhyperf, Difhyperg, [Difmu0]))
        return -Fv, -DFv

    def _optimizer(self, init, bounds):
        try:
            result = scipy.optimize.minimize(self.ELBOFull, init,
                                              method="L-BFGS-B", 
                                              jac = True, bounds = bounds, 
                                              options={'gtol': 1e-2})
        except Exception:
            result = scipy.optimize.OptimizeResult()
            result.fun = 10000
            print('diverge')
        return result
