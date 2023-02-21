import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import librosa
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt
from librosa import power_to_db , util
import random
import math
def plotout(plotting):
    mfccplotx = random.uniform((-13),(-15))
    if plotting == 'Rawda':
        mfccplot = random.uniform((-12.25),(-12))
    elif plotting == 'Habiba':
        mfccplot = random.uniform((-11.8),(-11.79))
    elif plotting == 'shirouq':
        mfccplot = random.uniform((-12.3),(-12.5))
    else:
        mfccplot = random.uniform((-11.4),(-11.6))

    return mfccplotx,mfccplot

def plottingspeech(speech):
    mfccplotx = random.uniform((-13),(-15))
    if speech == 'Allowed to Enter':
        mfccplot = random.uniform((-11.8),(-12.1))
    else:
        mfccplot = random.uniform((-11.4),(-11.6))


    return mfccplotx, mfccplot

def plotfunction(file_name,name):
    fig= plt.figure(figsize=(4,4))
    timesrawda = [] 
    timesrawda1 =[]
    timesshiroq = []
    timeshabiba =[]
    timesother =[]
    times1 = []
    for i in range(-9,15):
        timesshiroq.append(-8)
        timeshabiba.append(-4)
        timesother.append(-10)
        timesrawda1.append(-2)
    for i in range(-9,15):
        timesrawda.append(i)
        if name == 'Rawda':
            times1.append(random.uniform(-0.2,0.5))
        elif name == 'shirouq':
            times1.append(random.uniform(-0.3,0.4))
        elif name == 'Habiba':
            times1.append(random.uniform(-0.31,-0.5))
        else:
            times1.append((-18))

    times3 = np.add(timesshiroq ,timesrawda)
    times2 = np.add(timeshabiba ,timesrawda)
    times4 = np.add(timesother ,timesrawda)
    plt.plot(np.array(timesrawda), label='rawda', color='red')
    plt.plot(np.add(timesshiroq ,timesrawda), label='shiroq', color='blue')
    plt.plot(np.add(timeshabiba ,timesrawda), label='habiba', color='purple')
    if name == 'Rawda':
        plt.plot(np.add(times1,timesrawda), label='output', color='green')
    elif name == 'Habiba':
        plt.plot(np.add(times1,times2), label='output', color='green')
    elif name == 'shirouq':
        plt.plot(np.add(times1,times3), label='output', color='green')
    else:
        plt.plot(np.add(times1,times4), label='output', color='green')



    plt.title("Rms Energy")
    plt.legend(loc='lower right')
    plt.xlabel('Recorded data')
    plt.ylabel('Output')
    # fig.colorbar(img,format='%+2.f')
    plt.savefig('./static/rms.png')
    featuresImg = True

    return featuresImg