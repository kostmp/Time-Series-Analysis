import mpld3
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import arma_generate_sample, arma_acf
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMAfrom statsmodels.tsa.arima.model import ARIMA
import pmdarima as pmd
from nolitsa import data, delay, dimension, d2, utils
import nolds
from sklearn.neighbors import KDTree

def get_acf(xV, lags=200, alpha=0.05, show=True):
    '''
    calculate acf of timeseries xV to lag (lags) and show
    figure with confidence interval with (alpha)
    '''
    acfV = acf(xV, nlags=lags)[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(xV.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), acfV, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('ACF')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return acfV


def get_pacf(xV, lags=10, alpha=0.05, show=True):
    '''
    calculate pacf of timeseries xV to lag (lags) and show
    figure with confidence interval with (alpha)
    '''
    pacfV = pacf(xV, nlags=lags, method='ols-adjusted')[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(xV.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), pacfV, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('PACF')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return pacfV

!pip install pmdarima
import pmdarima as pmd
def arimamodel(xV):
    '''
    BUILT-IN SOLUTION FOR DETECTING BEST ARIMA MODEL MINIMIZING AIC
    https://alkaline-ml.com/pmdarima/index.html
    '''
    autoarim = pmd.auto_arima(xV,start_p=1, start_q=1,
                                     max_p=10, max_q=10,
                                     test="adf", stepwise=False,
                                     trace=True, information_criterion='aic')
    return autoarim

#------------------------------------------------------------------------------------------------------------------------------
#Φόρτωση δεδομένων αρχείων σε πίνακες
xv1989 = np.loadtxt('gdrive/MyDrive/NOTEBOOKS/EruptionData/eruption1989.dat')
xv2000 = np.loadtxt('gdrive/MyDrive/NOTEBOOKS/EruptionData/eruption2000.dat')
xv2011 = np.loadtxt('gdrive/MyDrive/NOTEBOOKS/EruptionData/eruption2011.dat')

#Έλεγχος για το αν η χρονοσειρά του 1989 είναι λευκός θόρυβος ή όχι μέσω των διαγραμμάτων της αυτοσυσχέτισης
fig = get_acf(xv1989, lags=20)
plt.title("Autocorrelation of 1989 year Series")
plt.show()
fig = get_pacf(xv1989, lags=100)
plt.title("Partial Autocorrelation of 1989 year Series")
plt.show()

#Παρομοίως για την χρονοσειρά του 2000
fig = get_acf(xv2000, lags=20)
plt.title("Autocorrelation of 1989 year Series")
plt.show()
fig = get_pacf(xv2000, lags=100)
plt.title("Partial Autocorrelation of 1989 year Series")
plt.show()

#Παρομοίως για την χρονοσειρά του 2011
fig = get_acf(xv2011, lags=20)
plt.title("Autocorrelation of 1989 year Series")
plt.show()
fig = get_pacf(xv2011, lags=100)
plt.title("Partial Autocorrelation of 1989 year Series")
plt.show()


#Εύρεση του καταλληλότερου μοντέλου πρόβλεψης για την χρονοσειρά του 1989
xv1989train = xv1989[0:295]
xv1989test = xv1989[295:298]
model = arimamodel(xv1989train)

model = ARIMA(xv1989train, order=(1,0,1))
model_fit = model.fit()
#Πρόβλεψη για ένα χρονικό βήμα μπροστά
forecast = model_fit.forecast(1)
print(forecast)

#Πρόβλεψη για 3 χρονικά βήματα μπροστά
forecast = model_fit.forecast(3)
print(forecast)


