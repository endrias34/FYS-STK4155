import numpy as np
import function as fx
from keras.models import Model, load_model

def test_SNR():
    pred_test = np.array([1,1,1,1])
    y_test   = np.array([1,1,1,0.5])
    SNR_test  = fx.SNR(pred_test, y_test)
    assert (SNR_test == 9.030899869919436)


def test_Boost():
    eta       = 0.00025
    lmbd      = 1e-6
    prev_model  = '../model/eta{}_lmd{}_model.h5'.format(eta, lmbd)
    prev_weights = '../weights/eta{}_lmd{}_weights.h5'.format(eta, lmbd)
    model      = load_model(prev_model)
    weights    = model.load_weights(prev_weights)
    pred_test   = np.ones((1, 256, 128, 1))
    y_test     = np.ones((1, 256, 128, 1))
    Boost_test  = fx.Boost(1, 0, model, pred_test, y_test)
    assert np.all(abs(np.mean(Boost_test - y_test)) < 0.1)
    
def test_R2():
    pred_test = np.ones((256, 1))+ 1.0e-4*np.random.randn(256,1)
    y_test   = np.ones((256, 1))
    R2_test  = fx.R2Score(pred_test.ravel(), y_test.ravel())
    assert (abs(R2_test) < 0.1)