#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy
import scipy
from scipy import spatial
import pyeeg
from scipy import signal
from scipy.integrate import simps
from sklearn.preprocessing import normalize
import pyhht

# Bandwidth function
def calcBandwidth(y_true, y_pred):
    sum1 = 0
    tres = 0.1 * (numpy.max(y_true) - numpy.min(y_true))
    for t,p in zip(y_true, y_pred):
        var = spatial.distance.euclidean(p,t)
        if var < tres:
            sum1 += 1
    return 1/len(y_true)*sum1

# Fractual Dimensions function
def calcFD(features, kmax):
    result = []
    for n, row in enumerate(features):
        temp_list = []
        for channel in row:
            temp_list += [pyeeg.hfd(channel,kmax)]
        result += [temp_list]
        if n % 400 == 0:
            print(n)
    return numpy.array(result)

# Band powers calculation
def calcBandPowers(features):
    sf = 666
    result = []
    lDelta, hDelta = 0.5, 4
    lTheta, hTheta = 4, 8
    lAlpha, hAlpha = 8, 12
    lBeta, hBeta = 12, 30
    
    win = 2 * sf
    for n, row in enumerate(features):
        if n % 400 == 0:
            print(n)
        temp_list = []
        for channel in row:
            freqs, psd = scipy.signal.welch(channel, sf, nperseg=win)
            
            Delta = numpy.logical_and(freqs >= lDelta, freqs <= hDelta)
            Theta = numpy.logical_and(freqs >= lTheta, freqs <= hTheta)
            Alpha = numpy.logical_and(freqs >= lAlpha, freqs <= hAlpha)
            Beta = numpy.logical_and(freqs >= lBeta, freqs <= hBeta)
            
            freq_res = freqs[1] - freqs[0]
            
            temp_list  += [simps(psd[Delta], dx=freq_res)]
            temp_list  += [simps(psd[Theta], dx=freq_res)]
            temp_list  += [simps(psd[Alpha], dx=freq_res)]
            temp_list  += [simps(psd[Beta], dx=freq_res)]
        result += [temp_list]
    return numpy.array(result)

# Calculate second difference (part of the statistics function)
def secDif(list):
    sum1 = 0
    for n, i in enumerate(list):
        if (n < len(list)-2):
            sum1 += numpy.sqrt(numpy.square(list[n+2]-i))
    result = (1/(len(list)-2)) * sum1
    return result

# Get statistics
def getStatistics(features):
    result = []
    for row in features:
        temp = []
        for channel in row:
            temp2 = []
            temp2 += [numpy.mean(channel)]
            temp2 += [numpy.std(channel)]
            temp2 += [(1/(len(channel)-1)) * numpy.sum(numpy.sqrt(numpy.square(numpy.diff(channel))))]
            norm = normalize(channel.reshape(-1, 1), axis = 0).reshape(-1)
            temp2 += [(1/(len(norm)-1)) * numpy.sum(numpy.sqrt(numpy.square(numpy.diff(norm))))]
            temp2 += [secDif(channel)]
            temp2 += [secDif(norm)]
            
            temp += [temp2]
        result += [temp]
    
    return numpy.array(result)

# Get statistics from the IMFs
def getIMFstats(features):
    result = numpy.array([])
    time = numpy.linspace(0,2, 1332)
    for n, row in enumerate(features):
        temp_result = numpy.array([])
        for n2, channel in enumerate(row):
            decomposer = pyhht.EMD(channel)
            imfs = decomposer.decompose()
            imfs = imfs[:7,]
            if imfs.shape[0] == 6:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            if imfs.shape[0] == 5:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            if imfs.shape[0] == 4:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            if imfs.shape[0] == 3:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            if imfs.shape[0] == 2:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            if imfs.shape[0] == 1:
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
                imfs = numpy.vstack((imfs, numpy.linspace(0,0,1332)))
            stats = getStatistics(imfs.reshape(1, -1, 1332))
            temp_result = numpy.hstack((temp_result, stats.reshape(-1)))
        if n % 400 == 0:
            print(n)
        if n == 0:
            result = temp_result
        else:
            result = numpy.vstack((result, temp_result))
    return result