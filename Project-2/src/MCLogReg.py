#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import os


class MCLogReg:
    '''
    Multi Class Logistic Regression (Softmax Regression)
    
    '''
    def __init__(self, eta=0.1, lamb=0, k=10, tol=1e-5, early_stop_tol = 0.0, early_stop_nochange=10):
        
        self.eta                 = eta
        self.lamb                = lamb
        self.k                   = k
        self.early_stop_tol      = early_stop_tol
        self.early_stop_nochange = early_stop_nochange
        self.beta                = float()

        
        
    def fit(self, xtrain, ytrain, xval, yval, Niter, batch_size = 200, solver='gd'):
        
        self.xtrain = self.augment_feature_vector(xtrain)
        self.ytrain = ytrain
        self.xval   = self.augment_feature_vector(xval)
        self.yval   = yval

        self.beta       = np.random.randn(self.k,self.xtrain.shape[1])
        xaxis           = []
        self.cost       = []
        self.cost_val   = []
        indexes         = np.arange(xtrain.shape[0])
        self.xtrain_tmp = self.augment_feature_vector(xtrain)
        self.ytrain_tmp = ytrain
        self.xval_tmp   = self.augment_feature_vector(xval)
        self.yval_tmp   = yval
        
        
        for iter in range(Niter):
            
            xaxis.append(iter+1)
            
            if solver == 'sgd':
                
                datapoints      = np.random.choice(indexes, size=batch_size, replace=False)
                self.xtrain_tmp = self.xtrain[datapoints,:]
                self.ytrain_tmp = self.ytrain[datapoints]
                self.oneiteration()
                self.costs()
                
            elif solver == 'gd':
                
                self.oneiteration()
                self.costs()

        return self.cost_val, self.cost, xaxis, self.beta

    def augment_feature_vector(self, X):
        
        return np.hstack((np.ones([len(X), 1]), X))
    
    def compute_probabilities(self, X):

        score   = self.beta@np.transpose(X)
        exp_mat = np.exp(score - np.amax(score, axis = 0))
        sum_vec = np.sum(exp_mat, axis = 0)
        prob    = exp_mat/sum_vec
    
        return prob

    def gradient_softmax(self, X, Y):
    
        n     = len(Y)
        datag = [1]*n
    
        H = self.compute_probabilities(X)
        M = sparse.coo_matrix((datag, (Y, range(n))), shape=(self.k,n)).toarray()
    
        first_term  = ((M - H)@X)*(-1/n)
        second_term = self.lamb*self.beta
        grad        = first_term + second_term

        return grad

    def compute_accuracy(self, predic, Y):

        acc = np.mean(predic == Y)
        return acc

    def cost_softmax(self, X, Y):
    
        n    = len(Y)
        data = [1]*n
        
        H = self.compute_probabilities(X)
        M = sparse.coo_matrix((data, (Y, range(n))), shape=(self.k,n)).toarray()
    
        first_term  = np.sum(M * np.log(H))*(-1/n)
        second_term = self.lamb/2*np.sum(self.beta**2)
        loss        = first_term + second_term
    
        return loss
    
    def costs(self):

        self.cost_val.append(self.cost_softmax(self.xval_tmp, self.yval_tmp.T))
        self.cost.append(self.cost_softmax(self.xtrain_tmp, self.ytrain_tmp.T))

    def oneiteration(self):

        self.beta -= self.eta*self.gradient_softmax(self.xtrain_tmp, self.ytrain_tmp.T)
    
    def predict(self, X):

        X             = self.augment_feature_vector(X)
        probabilities = self.compute_probabilities(X)
        classes       = np.argmax(probabilities, axis = 0)
        return classes
    
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
    
    