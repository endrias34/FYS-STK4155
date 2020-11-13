## Functions
import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.linalg as scl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def FrankeFunction(x, y, noise_level=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise


def OridinaryLeastSquares(design, data, test):
    inverse_term   = np.linalg.inv(design.T.dot(design))
    beta           = inverse_term.dot(design.T).dot(data)
    pred_test      = test @ beta
    pred_train     = design @ beta
    return beta, pred_test, pred_train


def OridinaryLeastSquares_SVD(design, data, test):
    U, s, V    = np.linalg.svd(design)
    beta       = V.T @ scl.pinv(scl.diagsvd(s, U.shape[0], V.shape[0])) @ U.T @ data
    pred_test  = test @ beta
    pred_train = design @ beta
    return beta, pred_test, pred_train


def RidgeRegression(design, data, test, _lambda=0):
    inverse_term   = np.linalg.inv(design.T @ design + _lambda*np.eye((design.shape[1])))
    beta           = inverse_term @ (design.T) @ (data)
    pred_test      = test @ beta
    pred_train     = design @ beta
    return beta, pred_test, pred_train 

def VarianceBeta_OLS(design, data, pred):
    N,p    = np.shape(design)
    sigma  = 1/(N-p-1) * np.sum((data - pred)**2)
    Bvar   = np.diag(np.linalg.inv(design.T @ design)*sigma)
    conf95 = 1.96*np.sqrt(Bvar)
    return Bvar, conf95

def VarianceBeta_Ridge(design, data, pred, _lambda=0):
    N,p    = np.shape(design)
    sigma  = 1/(N-p-1) * np.sum((data - pred)**2)
    x      = design.T @ design
    W      = np.linalg.inv(x + _lambda*np.eye(x.shape[0]))@x
    Bvar   = np.diag(sigma*W@np.linalg.inv(x + _lambda*np.eye(x.shape[0])).T)
    conf95 = 1.96*np.sqrt(Bvar)
    return Bvar, conf95

def MSE(y, ytilde):
    return (np.sum((y-ytilde)**2))/y.size


def R2Score(y, ytilde):
    return 1 - ((np.sum((y-ytilde)**2))/(np.sum((y-((np.sum(y))/y.size))**2)))


def MAE(y, ytilde):
    return (np.sum(np.abs(y-ytilde)))/y.size


def MSLE(y, ytilde):
    return (np.sum((np.log(1+y)  -  np.log(1+ytilde))**2))/y.size


def DesignDesign(x, y, power):
    '''
    This function employs the underlying pattern governing a design matrix
    on the form [1,x,y,x**2,x*y,y**2,x**3,(x**2)*y,x*(y**2),y**3 ....]
    x_power=[0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,...,n,n-1,...,1,0]
    y_power=[0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,...,0,1,...,n-1,n]
    '''

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

    #DesignMatrix = np.concatenate((np.ones((len(X),1)),DesignMatrix), axis = 1)
    return DesignMatrix

def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)  # Number of elements in beta
        X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def reshaper(k, data):
    output = []
    j = int(np.ceil(len(data)/k))
    for i in range(k):
        if i<k:
            output.append(data[i*j:(i+1)*j])
        else:
            output.append(data[i*j:])
    return np.asarray(output)


def k_fold_cv(k, indata, indesign, predictor, _lambda=0, shuffle=False):

    '''
    Usage: k-fold cross validation employing either RidgeRegression, OridinaryLeastSquares or ols_svd
    Input: k = number of folds
           indata = datapoints
           indesign = user defined design matrix
           predictor = RidgeRegression, OridinaryLeastSquares or ols_svd
           _lambda = hyperparameter/penalty paramter/tuning parameter for RidgeRegression
           shuffle = False, input data will not be shuffled
                     True, input data will be shuffled
    output: r2_out/k = averaged out sample R2-score
            mse_out/k = averaged out sample MSE
            r2_in/k = averaged in sample R2-Score
            mse_in/k = averaged in sample MSE
    '''
    mask = np.arange(indata.shape[0])
    if shuffle:
        np.random.shuffle(mask)
    data    = reshaper(k, indata[mask])
    design  = reshaper(k, indesign[mask])
    r2_out  = 0
    r2_in   = 0
    mse_out = 0
    mse_in  = 0

    for i in range(k):
        train_design = design[np.arange(len(design))!=i]      # Featch all but the i-th element
        train_design = np.concatenate(train_design,axis=0)
        train_data   = data[np.arange(len(data))!=i]
        train_data   = np.concatenate(train_data,axis=0)
        test_design  = design[i]
        test_data    = data[i]

        if _lambda != 0:
            beta, pred_ts, pred_tr = predictor(train_design, train_data, test_design, _lambda)
        else:
            beta, pred_ts, pred_tr = predictor(train_design, train_data, test_design)

        r2_out  += R2Score(test_data, pred_ts)
        r2_in   += R2Score(train_data, pred_tr)
        mse_out += MSE(test_data, pred_ts)
        mse_in  += MSE(train_data, pred_tr)

    return r2_out/k, mse_out/k, r2_in/k, mse_in/k

def norm_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          cmap=plt.cm.Blues):
    '''
    Computes a normalized or non-nonrmalized confusion matrix.
    '''

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]




    return cm
