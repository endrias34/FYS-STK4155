import numpy as np
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import NeuralNet as nn
import functions as fx
import functions_NN as lrf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

m = 1000
x = 5*np.random.rand(m,1)
y = 6 + 7*x + 0.1*np.random.randn(m,1)
X = np.c_[np.ones((m,1)), x]

# Train and test split
train_size = 0.8
test_size  = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=train_size,test_size=test_size)

# SGD parameters
epochs     = 1000
batch_size = int(len(Y_train)/32)
n_features = X_train.shape[1] 
eta        = 0.001
lmb        = 1e-4

activation = [lrf.leaky_relu, lrf.nooutact]
derivative = [lrf.leaky_relu_deriv, lrf.nooutact_deriv]


# Creating the network object and defining the hyperparameters
neural_net = nn.ANN(lmb = lmb, bias = 0.01, eta = eta, early_stop_tol = 1e-7,\
                                early_stop_nochange = 2000, mode = 'regression', regularization = 'l2')
# Adding layers
neural_net.add_layers(n_features=[n_features,20], n_neurons = [20,1] , n_layers=2)
            
# Training the network
neural_net.train(epochs, batch_size, X_train, Y_train, activation, derivative \
                             , X_test, Y_test, verbose = False)           
# performance metrics
pred = neural_net.feed_out(X_test, activation)

reg = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(20),activation='logistic',
                                          batch_size=batch_size,learning_rate='adaptive',
                                          learning_rate_init=eta,alpha=lmb,max_iter=epochs,tol=1e-5,
                                          verbose=False)
reg = reg.fit(X_train, Y_train.ravel())

# performance metrics
pred_sk = reg.predict(X_test)
    
def test_NN_MSE():

    test_loss    = fx.MSE(pred.ravel(), Y_test.T)
    test_loss_sk = mean_squared_error(Y_test.ravel(), pred_sk)
    
    assert (abs(test_loss_sk - test_loss) < 1e-1)
    
def test_NN_R2():
    
    test_r2     = fx.R2Score(pred.ravel(), Y_test.T)
    test_r2_sk  = reg.score(X_test, Y_test.ravel())
    
    assert (abs(test_r2_sk - test_r2) < 1e-1)
