import numpy as np
import scipy.sparse
'''
This file includes the activation functions and cost functions used by
NeuralNetwork.py
and
LogisticRegression.py
'''

def sigmoid(prediction):
    '''
    Sigmoid activation function, from logistic regression slides.
    '''
    return 1. / (1. + np.exp(-prediction))

def sigmoid_deriv(activation):
    '''
    Returns derivative of sigmoid activation function.
    '''
    derivative = activation*(1-activation)
    return derivative

def relu(prediction):
    '''
    ReLU activation function.
    '''
    out = np.copy(prediction)
    out[np.where(prediction < 0)]=0
    out = np.clip(out,-300,300)
    return out

def relu_deriv(prediction):
    '''
    Returns the derivative of ReLU.
    '''
    derivative = np.copy(prediction)
    derivative[np.where(prediction < 0)] = 0
    derivative[np.where(prediction >= 0)] = 1
    return derivative

def leaky_relu(prediction):
    '''
    Leaky_ReLU activation function.
    '''
    out = np.copy(prediction)
    out[np.where(prediction < 0)] = 0.01 * out[np.where(prediction < 0)]
    out = np.clip(out,-300,300)
    return out

def leaky_relu_deriv(prediction):
    '''
    Returns the derivative of Leaky_ReLU.
    '''
    derivative = np.copy(prediction)
    derivative[np.where(prediction < 0)] = 0.01
    derivative[np.where(prediction >= 0)] = 1
    return derivative

def elu(prediction):
    '''
    ELU activation function.
    '''
    out = np.copy(prediction)
    out[np.where(prediction < 0)] = 0.01 * (np.exp(out[np.where(prediction < 0)])-1)
    out = np.clip(out,-300,300)
    return out

def elu_deriv(prediction):
    '''
    Returns the derivative of ELU.
    '''
    derivative = np.copy(prediction)
    derivative[np.where(prediction < 0)] = 0.01 * np.exp(derivative[np.where(prediction < 0)])
    derivative[np.where(prediction >= 0)] = 1
    return derivative    
                                                          
def nooutact(prediction):
    '''
    Can be used for activation in output layer in case of regression.
    '''
    return prediction                                                                                                                    
def nooutact_deriv(prediction):
    out = np.ones(prediction.shape)
    return out


def augment_feature_vector(X):
    return np.hstack((np.ones([len(X), 1]), X))

def compute_probabilities(X, theta):

    theta_XT   = np.matmul(theta, np.transpose(X))
    c          = np.amax(theta_XT, axis = 0)
    exp_matrix = np.exp(theta_XT - c)
    sum_vector = np.sum(exp_matrix, axis = 0)
    prob       = exp_matrix/sum_vector
    
    return prob

def gradient_softmax(X, Y, theta, lamb):
    
    n    = len(Y)
    k    = theta.shape[0]
    data = [1]*n
    
    H = compute_probabilities(X, theta)
    M = sparse.coo_matrix((data, (Y, range(n))), shape=(k,n)).toarray()
    
    first_term  = np.matmul(M - H, X)*(-1/n)
    second_term = lamb*theta
    grad        = first_term + second_term

    return grad

def compute_accuracy(pred, Y):

    acc = np.mean(pred == Y)
    return acc

def cost_softmax(X, Y, theta, lamb):
    
    n    = len(Y)
    k    = theta.shape[0]
    data = [1]*n
    
    H = compute_probabilities(X, theta)
    M = sparse.coo_matrix((data, (Y, range(n))), shape=(k,n)).toarray()
    
    first_term  = np.sum(M * np.log(H))*(-1/n)
    second_term = lamb/2*np.sum(theta*theta)
    loss        = first_term + second_term
    
    return loss


def cost_mse_ols(design, data, beta):
    '''
    Mean squared error
    '''
    return (data - design.dot(beta)).T*(data - design.dot(beta))

def cost_grad_ols(design, data, beta):
    '''
    Calculates the first derivative of MSE w.r.t beta.
    '''
    return (2/len(data))*design.T.dot(design.dot(beta)-data) #logistic regression slides

def cost_log_ols(prediction, data):
    '''
    Logisitic regression cost function
    '''
    length = data.shape[1]
    prediction = prediction.ravel()
    data = data.ravel()
    calc = -data.dot(np.log(sigmoid(prediction)+ 1e-16)) - ((1 - data).dot(np.log(1 - sigmoid(prediction) + 1e-16)))
    norm = calc/length
    return norm

def gradient_ols(design, data, p):
    '''
    Gradient w.r.t log
    '''
    return np.dot(design.T, (p - data)) / data.shape[0]

def reshaper(k, data):
    '''
    Usage: Manages the data for k_fold_cv
    Input: k = number of folds
           data = shuffled input data or input design matrix
    output: Splitted data
    '''
    output = []
    j = int(np.ceil(len(data)/k))
    for i in range(k):
        if i<k:
            output.append(data[i*j:(i+1)*j])
        else:
            output.append(data[i*j:])
    return np.asarray(output)


def k_fold_reshaper(k, indata, indesign, shuffle=True):

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
    data = reshaper(k, indata[mask])
    design = reshaper(k, indesign[mask])
    return data,design
