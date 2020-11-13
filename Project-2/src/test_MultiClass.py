import numpy as np
import MCLogReg as mclr


def test_accuracy():
    
    logreg    = mclr.MCLogReg()
    pred_test = np.array([1,1,1,1])
    y_test    = np.array([1,1,0,0])
    acc_test  = logreg.compute_accuracy(pred_test, y_test)
    assert (acc_test == 0.5)
    
