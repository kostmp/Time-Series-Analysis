#Η ΑΝΑΛΥΣΗ ΧΡΟΝΟΣΕΙΡΩΝ ΑΡΧΙΖΕΙ ΑΠΟ ΤΗΝ ΓΡΑΜΜΗ 250
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

!pip install -U nolds 
import nolds
def correlationdimension(xV, tau, m_max, fac=4, logrmin=-1e6, logrmax=1e6, show=False):
    m_all = np.arange(1, m_max + 1)
    corrdimV = []
    logrM = []
    logCrM = []
    polyM = []

    for m in m_all:
        corrdim, *corrData = nolds.corr_dim(xV, m, debug_data=True)
        corrdimV.append(corrdim)
        logrM.append(corrData[0][0])
        logCrM.append(corrData[0][1])
        polyM.append(corrData[0][2])
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(m_all, corrdimV, marker='x', linestyle='-')
        ax.set_xlabel('m')
        ax.set_xticks(m_all)
        ax.set_ylabel('v')
        ax.set_title('Corr Dim vs m')

    return corrdimV, logrM, logCrM, polyM

def localpredictnrmse(xV, nlast, m, tau=1, Tmax=1, nnei=1, q=0, show=''):
    xV = xV.reshape(-1, )
    n = xV.shape[0]
    if nlast > n - 2 * m * tau:
        print('test set too large')
    n1 = n - nlast
    if n1 < 2 * (m - 1) * tau - Tmax:
        print('the length of training set is too small for this data size')
    n1vec = n1 - (m - 1) * tau - 1
    xM = np.full(shape=(n1vec, m), fill_value=np.nan)
    for j in np.arange(m):
        xM[:, m - j - 1] = xV[j * tau:n1vec + j * tau]
    from scipy.spatial import KDTree
    kdtreeS = KDTree(xM)

    # For each target point, find neighbors, apply the linear models and keep track
    # of the predicted values each prediction time.
    ntar = nlast - Tmax + 1;
    preM = np.full(shape=(ntar, Tmax), fill_value=np.nan)
    winnowM = np.full(shape=(ntar, (m - 1) * tau + 1), fill_value=np.nan)

    ifirst = n1 - (m - 1) * tau;
    for i in np.arange(m * tau):
        winnowM[:, i] = xV[ifirst + i - 1:ifirst + ntar + i - 1]

    for T in np.arange(1, Tmax + 1):
        targM = winnowM[:, :-(m + 1) * tau:-tau]
        _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)
        for i in np.arange(ntar):
            neiM = xM[nneiindM[i], :]
            yV = xV[nneiindM[i] + (m - 1) * tau + 1]
            if q == 0 or nnei == 1:
                preM[i, T - 1] = np.mean(yV)
            else:
                mneiV = np.mean(neiM, axis=0)
                my = np.mean(yV)
                zM = neiM - mneiV
                [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
                Sx = np.diag(Sx)
                Vx = Vx.T
                tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
                lsbV = tmpM @ (yV - my)
                preM[i, T - 1] = my + (targM[i, :] - mneiV) @ lsbV
        winnowM = np.concatenate([winnowM, preM[:, [T - 1]]], axis=1)
    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)

    start_idx = (n1vec + (m - 1) * tau)
    end_idx = start_idx + preM.shape[0]
    for t_idx in np.arange(1, Tmax + 1):
        nrmseV[t_idx - 1] = nrmse(trueV=xV[start_idx + t_idx:end_idx + t_idx], predictedV=preM[:, t_idx - 1])
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
        ax.axhline(1., color='yellow')
        ax.set_title(f'NRMSE(T), m={m}, tau={tau}, q={q}, n={n}, nlast={nlast}')
    return nrmseV, preM

def localfitnrmse(xV, tau, m, Tmax, nnei, q, show=''):
    if q > m:
        q = int(m)
    n = xV.shape[0]

    if n < 2 * (m - 1) * tau - Tmax:
        print('too short timeseries')
        return

    nvec = n - (m - 1) * tau - Tmax
    xM = np.full(shape=(nvec, m), fill_value=np.nan)

    for j in np.arange(m):
        xM[:, [m - j - 1]] = xV[j * tau:nvec + j * tau]
    from scipy.spatial import KDTree
    kdtreeS = KDTree(xM)
    preM = np.full(shape=(nvec, Tmax), fill_value=np.nan)
    _, nneiindM = kdtreeS.query(xM, k=nnei + 1, p=2)
    nneiindM = nneiindM[:, 1:]
    for i in np.arange(nvec):
        neiM = xM[nneiindM[i]]
        yV = xV[nneiindM[i] + m * tau]
        if q == 0 or nnei == 1:
            preM[i, 0] = np.mean(yV)
        else:
            mneiV = np.mean(neiM, axis=0)
            my = np.mean(yV)
            zM = neiM - mneiV
            [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
            Sx = np.diag(Sx)
            Vx = Vx.T
            tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
            lsbV = tmpM @ (yV - my)
            preM[i] = my + (xM[i,] - mneiV) @ lsbV
    if Tmax > 1:
        winnowM = np.full(shape=(nvec, (m - 1) * tau + 1), fill_value=np.nan)
        for i in np.arange(m * tau):
            winnowM[:, [i]] = xV[i:nvec + i]
        for T in np.arange(2, Tmax + 1):
            winnowM = np.concatenate([winnowM, preM[:, [T - 2]]], axis=1)
            targM = winnowM[:, :-(m + 1) * tau:-tau]
            _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)

            for i in np.arange(nvec):
                neiM = xM[nneiindM[i], :]
                yV = xV[nneiindM[i] + (m - 1) * tau + 1]
                if q == 0 or nnei == 1:
                    preM[i, T - 1] = np.mean(yV)
                else:
                    mneiV = np.mean(neiM, axis=0)
                    my = np.mean(yV)
                    zM = neiM - mneiV
                    [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
                    Sx = np.diag(Sx)
                    Vx = Vx.T
                    tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
                    lsbV = tmpM @ (yV - my)
                    preM[i, T - 1] = my + (targM[i, :] - mneiV) @ lsbV

    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)
    idx = (np.arange(nvec) + (m - 1) * tau).astype(np.int)
    for t_idx in np.arange(1, Tmax + 1):
        nrmseV[t_idx - 1] = nrmse(trueV=xV[idx + t_idx,], predictedV=preM[:, [t_idx - 1]])
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
    return nrmseV, preM


def plot_timeseries(xV, get_histogram=False, title='', savepath=''):
    # #plot timeseries
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(xV, marker='x', linestyle='--', linewidth=2);
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    if len(title) > 0:
        ax.set_title(title, x=0.5, y=1.0);
    plt.tight_layout()
    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title}_xM.jpeg')
    # #plot histogram
    if get_histogram:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.hist(xV, alpha=0.8, rwidth=0.9);
        ax.set_xlabel('value')
        ax.set_title('Histogram')
        plt.tight_layout()
        if len(title) > 0:
            ax.set_title(title, x=0.5, y=1.0);
        plt.tight_layout()
        if len(savepath) > 0:
            plt.savefig(f'{savepath}/{title}_hist.jpeg')

def mutualInformation(data, delay, nBins):
    "This function calculates the mutual information given the delay"
    I = 0;
    xmax = max(data);
    xmin = min(data);
    delayData = data[delay:len(data)];
    shortData = data[0:len(data)-delay];
    sizeBin = abs(xmax - xmin) / nBins;
    
    probInBin = {};
    conditionBin = {};
    conditionDelayBin = {};
    for h in range(0,nBins):
        if h not in probInBin:
            conditionBin.update({h : (shortData >= (xmin + h*sizeBin)) & (shortData < (xmin + (h+1)*sizeBin))})
            probInBin.update({h : len(shortData[conditionBin[h]]) / len(shortData)});
        for k in range(0,nBins):
            if k not in probInBin:
                conditionBin.update({k : (shortData >= (xmin + k*sizeBin)) & (shortData < (xmin + (k+1)*sizeBin))});
                probInBin.update({k : len(shortData[conditionBin[k]]) / len(shortData)});
            if k not in conditionDelayBin:
                conditionDelayBin.update({k : (delayData >= (xmin + k*sizeBin)) & (delayData < (xmin + (k+1)*sizeBin))});
            Phk = len(shortData[conditionBin[h] & conditionDelayBin[k]]) / len(shortData);
            if Phk != 0 and probInBin[h] != 0 and probInBin[k] != 0:
                I -= Phk * math.log( Phk / (probInBin[h] * probInBin[k]));
    return I;


def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')    
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData;


from sklearn.neighbors import NearestNeighbors 
def false_nearest_neighours(data,delay,embeddingDimension):
    "Calculates the number of false nearest neighbours of embedding dimension"    
    embeddedData = takensEmbedding(data,delay,embeddingDimension);
    #the first nearest neighbour is the data point itself, so we choose the second one
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embeddedData.transpose())
    distances, indices = nbrs.kneighbors(embeddedData.transpose())
    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    epsilon = np.std(distances.flatten())
    nFalseNN = 0
    for i in range(0, len(data)-delay*(embeddingDimension+1)):
        if (0 < distances[i,1]) and (distances[i,1] < epsilon) and ( (abs(data[i+embeddingDimension*delay] - data[indices[i,1]+embeddingDimension*delay]) / distances[i,1]) > 10):
            nFalseNN += 1;
    return nFalseNN

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------


#Φόρτωση των δεδομένων της χρονοσειράς του 2007
xv = np.loadtxt('gdrive/MyDrive/NOTEBOOKS/EruptionData/eruption2007.dat')
xv500 = xv[0:500]

##Σχεδιάζουμε για το πρώτο ερώτημα τις χρονοσειρές
plot_timeseries(xv, get_histogram=False, title='Time series of 2007', savepath='')
plot_timeseries(xv500, get_histogram=False, title='First 500 data of timeseries 2007', savepath='')

##Στο δεύτερο ερώτημα κάνουμε τον στατιστικό έλεγχο με βάση την αυτοσυσχέτιση (Portmanteau test)
print(acorr_ljungbox(xv, lags=30))
print(acorr_ljungbox(xv500, lags=30))

##Στο τρίτο ερώτημα βρίσκουμε την υστέρηση τ
#Πρώτα για ολόκληρη την χρονοσειρά του 2007
datDelayInformation = []
for i in range(1,100):
    datDelayInformation = np.append(datDelayInformation,[mutualInformation(xv,i,500)])
plt.plot(range(1,100),datDelayInformation);
plt.xlabel('delay');
plt.ylabel('mutual information');

#Για τις πρώτες 500 μετρήσεις της χρονοσειράς του 2007
datDelayInformation = []
for i in range(1,20):
    datDelayInformation = np.append(datDelayInformation,[mutualInformation(xv500,i,50)])
plt.plot(range(1,20),datDelayInformation);
plt.xlabel('delay');
plt.ylabel('mutual information');


##Στο τέταρτο ερώτημα η εκτίμηση της διάστασης m σύμφωνα με το κριτήριο των ψευδών κοντινότερων γειτόνων
#Για ολόκληρη την χρονοσειρά του 2007
nFNN = []
for i in range(1,20):
    nFNN.append(false_nearest_neighours(xv,1,i) / len(xv))
plt.plot(range(1,20),nFNN);
plt.xlabel('embedding dimension');
plt.ylabel('Fraction of fNN');

#Για τις πρώτες 500 μετρήσεις
nFNN = []
for i in range(1,20):
    nFNN.append(false_nearest_neighours(xv500,2,i) / len(xv500))
plt.plot(range(1,20),nFNN);
plt.xlabel('embedding dimension');
plt.ylabel('Fraction of fNN');

##Για το πέμπτο ερώτημα
#Ολόκληρη χρονοσειρά του 2007 και τοπικό μοντέλο μέσου όρου
nrmseV, preM = localfitnrmse(xv, 1, 7, 1, 500, 0, show=True)
plt.figure(figsize=(14,8))
plt.plot(xv)
plt.plot(preM)

#Για τοπικό γραμμικό μοντέλο
nrmseV, preM = localfitnrmse(xv, 1, 7, 1, 500, 8, show=True)
plt.figure(figsize=(14,8))
plt.plot(xv)
plt.plot(preM)

#Για την χρονοσειρά των 500 μετρήσεων και τοπικο μοντέλο μέσου όρου
nrmseV, preM = localfitnrmse(xv500, 2, 5, 1, 50, 0, show=True)
plt.figure(figsize=(14,8))
plt.plot(xv)
plt.plot(preM)

#Για τοπικό γραμμικό μοντέλο
nrmseV, preM = localfitnrmse(xv, 2, 5, 1, 50, 7, show=True)
plt.figure(figsize=(14,8))
plt.plot(xv)
plt.plot(preM)

##Για το έκτο ερώτημα
nrmseV, preM = localpredictnrmse(xv,1,7,1,1,500,8,True)
nrmseV, preM = localpredictnrmse(xv500,1,5,2,1,50,7,True)

#Για το έβδομο ερώτημα
corrdimV, logrM, logCrM, polyM = correlationdimension(xv, 1,10, fac=24, logrmin=-1e6, logrmax=1e6, show=True)
corrdimV, logrM, logCrM, polyM = correlationdimension(xv500, 2,10, fac=24, logrmin=-1e6, logrmax=1e6, show=True)