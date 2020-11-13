import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
import LinearReg as lreg
import functions as fx
from sklearn.model_selection import train_test_split

m = 1000
x = 5*np.random.rand(m,1)
y = 6 + 7*x + 0.1*np.random.randn(m,1)
X = np.c_[np.ones((m,1)), x]

# Train and test split
train_size = 0.8
test_size  = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=train_size,test_size=test_size)

# SGD parameters
eta        = 0.01
lamb       = 0.001
epochs     = 10000
batch_size = 300

def test_SGD_OLS_beta():
    
    lr = lreg.LinearReg(eta = eta, lamb=0.0, eta_type ='linear',eta_scal=True)
    _, _, _, beta_SGD, _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty=None, alpha=0.0, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())

    assert np.all(abs(beta_SGD.T - SGD_sk.coef_) < 0.25)

def test_SGD_Ridge_beta():
    
    lr = lreg.LinearReg(eta = eta, lamb=lamb, eta_type ='linear',eta_scal=True)
    _, _, _, beta_SGD, _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty='l2', alpha=lamb, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())

    assert np.all(abs(beta_SGD.T - SGD_sk.coef_) < 1e-1)

def test_SGD_OLS_MSE():
    
    lr = lreg.LinearReg(eta = eta, lamb=0.0, eta_type ='linear',eta_scal=True)
    _, _, _, _ , _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    pred_SGD_test = lr.predict(X_test)
    MSE_test      = lr.MSE(Y_test,pred_SGD_test)
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty=None, alpha=0.0, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())
    
    pred_SGD_test_sk = SGD_sk.predict(X_test)
    MSE_test_sk      = mean_squared_error(Y_test,pred_SGD_test_sk)
    
    assert (abs(MSE_test - MSE_test_sk) < 1e-1)
    
def test_SGD_Ridge_MSE():
    
    lr = lreg.LinearReg(eta = eta, lamb=lamb, eta_type ='linear',eta_scal=True)
    _, _, _, _ , _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    pred_SGD_test = lr.predict(X_test)
    MSE_test      = lr.MSE(Y_test,pred_SGD_test)
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty='l2', alpha=lamb, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())
    
    pred_SGD_test_sk = SGD_sk.predict(X_test)
    MSE_test_sk      = mean_squared_error(Y_test,pred_SGD_test_sk)
    
    assert (abs(MSE_test - MSE_test_sk) < 1e-1)
    
def test_SGD_OLS_R2():
    
    lr = lreg.LinearReg(eta = eta, lamb=0.0, eta_type ='linear',eta_scal=True)
    _, _, _, _ , _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    pred_SGD_test = lr.predict(X_test)
    r2_test       = lr.R2Score(Y_test,pred_SGD_test)
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty=None, alpha=0.0, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())
    
    pred_SGD_test_sk = SGD_sk.predict(X_test)
    r2_test_sk       = r2_score(Y_test,pred_SGD_test_sk)
    
    assert (abs(r2_test - r2_test_sk) < 1e-1)
    
def test_SGD_Ridge_R2():
    
    lr = lreg.LinearReg(eta = eta, lamb=lamb, eta_type ='linear',eta_scal=True)
    _, _, _, _ , _    = lr.fit(X_train, Y_train, X_test, Y_test, Niter=epochs,batch_size=batch_size, solver='sgd')
    
    pred_SGD_test = lr.predict(X_test)
    r2_test       = lr.R2Score(Y_test,pred_SGD_test)
    
    SGD_sk = SGDRegressor(max_iter = epochs, penalty='l2', alpha=lamb, eta0=eta,fit_intercept=False)
    SGD_sk.fit(X_train,Y_train.ravel())
    
    pred_SGD_test_sk = SGD_sk.predict(X_test)
    r2_test_sk       = r2_score(Y_test,pred_SGD_test_sk)
    
    assert (abs(r2_test - r2_test_sk) < 1e-1)