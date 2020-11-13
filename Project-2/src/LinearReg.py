#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os


class LinearReg:
    '''
    Linear Regression 
    
    '''
    def __init__(self, eta=0.1, eta_scal=False, eta_type ='linear', lamb=0, tol=1e-5, early_stop_tol = 0.0, early_stop_nochange=10):
        
        self.eta                 = eta   
        self.eta_0               = eta
        self.eta_f               = 1.0e-6
        self.eta_scal            = eta_scal
        self.eta_type            = eta_type 
        self.lamb                = lamb
        self.early_stop_tol      = early_stop_tol
        self.early_stop_nochange = early_stop_nochange
        self.beta                = float()
    
    def fit(self, xtrain, ytrain, xval, yval, Niter, batch_size = 1, solver='gd'):
        
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval   = xval
        self.yval   = yval

        self.batch_size = batch_size
        self.beta       = np.random.uniform(0, 1, self.xtrain.shape[1]) 
        self.beta       = self.beta.reshape(-1,1)
        xaxis           = []
        self.cost       = []
        self.eta_vec    = []
        self.cost_val   = []
        indexes         = np.arange(xtrain.shape[0])
        self.xtrain_tmp = xtrain
        self.ytrain_tmp = ytrain
        self.xval_tmp   = xval
        self.yval_tmp   = yval
        
        
        for iter in range(Niter):
            
            xaxis.append(iter+1)
            
            if solver == 'sgd':
                
                mbatch          = np.int(self.xtrain.shape[0]/self.batch_size)
                datapoints      = np.random.choice(indexes, size=mbatch, replace=False)
                self.xtrain_tmp = self.xtrain[datapoints,:]
                self.ytrain_tmp = self.ytrain[datapoints]
                
                if self.eta_scal and self.eta_type == 'linear':
                    factor = iter/(Niter-1)
                    self.learning_schedule_linear(factor)
                    
                elif self.eta_scal and self.eta_type == 'exp':
                    factor = iter/(Niter-1)
                    self.learning_schedule_exp(factor)
 
                self.eta_vec.append(self.eta)
                self.oneiteration_sgd()
                self.costs()
                
            elif solver == 'gd':

                self.oneiteration_gd()
                self.costs()

        return self.cost_val, self.cost, xaxis, self.beta, self.eta_vec

    def learning_schedule_linear(self,t):
        self.eta  = self.eta_0 + (self.eta_f - self.eta_0)*t
        
    def learning_schedule_exp(self,t):
        self.eta  = self.eta_0*np.exp(np.log(self.eta_f/self.eta_0)*t)
        
    def gradient_gd(self, X, Y):
    
        m     = len(Y)
        Pred  = X @ self.beta
        grad  = (2.0/m)*(X.T @ (Pred - Y) + self.lamb * self.beta)
        return grad
    
    def gradient_sgd(self, X, Y):
        
        m     = len(Y)
        Pred  = X @ self.beta
        grad  = 2 * (X.T @ (Pred - Y) + self.lamb * self.beta)

        return grad

    def cost_ols(self, X, Y):
        
        m    = len(Y)
        Pred = X@self.beta
        loss = (1.0/m)*(np.linalg.norm((Pred - Y), 2) ** 2)
    
        return loss
    
    def costs(self):

        self.cost_val.append(self.cost_ols(self.xval_tmp, self.yval_tmp))
        self.cost.append(self.cost_ols(self.xtrain_tmp, self.ytrain_tmp))

    def oneiteration_gd(self):

        self.beta -= self.eta*self.gradient_gd(self.xtrain_tmp, self.ytrain_tmp)
        
    def oneiteration_sgd(self):

        self.beta -= self.eta*self.gradient_sgd(self.xtrain_tmp, self.ytrain_tmp)

    def reshaper(self, nk, data):

        output = []
        j      = int(np.ceil(len(data)/nk))
        for i in range(nk):
            if i<nk:
                output.append(data[i*j:(i+1)*j])
            else:
                output.append(data[i*j:])
        return np.asarray(output)
    
    def k_fold_reshaper(self, nk, indata, indesign, shuffle=True):

        mask = np.arange(indata.shape[0])
        if shuffle:
            np.random.shuffle(mask)
        data   = self.reshaper(nk, indata[mask])
        design = self.reshaper(nk, indesign[mask])
        return data, design
    
    def predict(self, X):

        pred = X@self.beta
        return pred
    
    def FrankeFunction(self, x, y, noise_level=0):
        
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        noise = noise_level*np.random.randn(len(x),len(y))
        return term1 + term2 + term3 + term4 + noise


    def OridinaryLeastSquares(self, design, data, test):

        inverse_term   = np.linalg.inv(design.T@design)
        alpha          = (inverse_term@design.T)@data
        pred_test      = test @ alpha
        pred_train     = design @ alpha
        return alpha, pred_test, pred_train

    def RidgeRegression(self, design, data, test, _lambda=0):
        inverse_term   = np.linalg.inv(design.T @ design + _lambda*np.eye((design.shape[1])))
        beta           = inverse_term @ (design.T) @ (data)
        pred_test      = test @ beta
        pred_train     = design @ beta
        return beta, pred_test, pred_train 

    def VarianceBeta_OLS(self, design, data, pred):
        N,p    = np.shape(design)
        sigma  = 1/(N-p-1) * np.sum((data - pred)**2)
        Bvar   = np.diag(np.linalg.inv(design.T @ design)*sigma)
        conf95 = 1.96*np.sqrt(Bvar)
        return Bvar, conf95

    def VarianceBeta_Ridge(self, design, data, pred, _lambda=0):
        N,p    = np.shape(design)
        sigma  = 1/(N-p-1) * np.sum((data - pred)**2)
        x      = design.T @ design
        W      = np.linalg.inv(x + _lambda*np.eye(x.shape[0]))@x
        Bvar   = np.diag(sigma*W@np.linalg.inv(x + _lambda*np.eye(x.shape[0])).T)
        conf95 = 1.96*np.sqrt(Bvar)
        return Bvar, conf95

    def MSE(self, y, ytilde):
        m = y.shape[0]
        return (1/m)*(np.sum((y-ytilde)**2))


    def R2Score(self, y, ytilde):
        m = y.shape[0]
        return 1 - ((np.sum((y-ytilde)**2))/(np.sum((y-((np.sum(y))/m))**2)))

    def DesignDesign(self, x, y, power):
        
        concat_x   = np.array([0,0])
        concat_y   = np.array([0,0])


        for i in range(power):
            toconcat_x = np.arange(i+1,-1,-1)
            toconcat_y = np.arange(0,i+2,1)
            concat_x   = np.concatenate((concat_x,toconcat_x))
            concat_y   = np.concatenate((concat_y,toconcat_y))

        concat_x     = concat_x[1:len(concat_x)]
        concat_y     = concat_y[1:len(concat_y)]

        X,Y          = np.meshgrid(x,y)
        X            = np.ravel(X)
        Y            = np.ravel(Y)
        DesignMatrix = np.empty((len(X),len(concat_x)))
        for i in range(len(concat_x)):
            DesignMatrix[:,i]   = (X**concat_x[i])*(Y**concat_y[i])

        return DesignMatrix
