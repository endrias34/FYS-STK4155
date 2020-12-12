import numpy as np

def Boost(rho, tau, model, noisy_test, denoised):
    denoised = tau * model.predict(noisy_test + rho*denoised) - (rho*tau + tau - 1)*denoised
    return denoised

def R2Score(y, ytilde):
    return 1 - ((np.sum((y-ytilde)**2))/(np.sum((y-((np.sum(y))/y.size))**2)))

def SNR(clean, denoised):
    return 10*np.log10(np.linalg.norm(clean**2)/np.linalg.norm(clean - denoised)**2)
       

