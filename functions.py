import pyaudio
import pickle
from scipy.io.wavfile import write,read
import os
from a import *
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


def rms(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode="constant",):
    if y is not None:
        x = util.frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    elif S is not None:
        # power spectrogram
        x = np.abs(S) ** 2
        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length ** 2
    return np.sqrt(power)



def calculate_delta(array):
    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
	mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
	mfcc_feature = preprocessing.scale(mfcc_feature)
	print(mfcc_feature)
	delta = calculate_delta(mfcc_feature)
	combined = np.hstack((mfcc_feature,delta)) 
	return combined



def modelsExtract(folderPath):
    gmmFiles = [os.path.join(folderPath,fname) for fname in os.listdir(folderPath) if fname.endswith('.gmm')]
    #Load the Gaussian gender Models
    modelsLoading  = [pickle.load(open(fname,'rb')) for fname in gmmFiles]
    nameOfModel  = [fname.split("\\")[-1].split(".gmm")[0]  for fname in gmmFiles]
    sampleRate,audio = read('output.wav')
    featuresArray   = extract_features(audio,sampleRate)
    toCheck = np.zeros(len(modelsLoading)) ## [0,0,0]
    for model in range(len(modelsLoading)):
        gmmCheck    = modelsLoading[model]  #checking with each model one by one
        scores = np.array(gmmCheck.score(featuresArray)) 
        toCheck[model] = scores.sum()  
    return modelsLoading,nameOfModel,featuresArray,toCheck


def extractFeaturesForFolder(folderName):
    features = np.asarray(())
    directory = folderName
    for audio in os.listdir(directory):
        audio_path = directory + audio
        sr,audio = read(audio_path)
        featuresArray   = extract_features(audio,sr)
        if features.size == 0:
            features = featuresArray
        else:
            features = np.vstack((features, featuresArray))
    return features


def plot_melspectrogram(file_name):
    fig=plt.figure(figsize=(5,5))
    audio,sfreq = librosa.load(file_name)

    melFb  = librosa.filters.mel(sfreq,1200 ,20, norm=None)
    melFb = melFb/np.max(melFb)
    freq = np.linspace(0, sfreq, int(1200/2+1))
    plt.title('Triangular Mel filter bank')
    img=plt.plot(freq, melFb.T)
    plt.savefig('./static/MelFilter.png')
    melImg = True
    return melImg

def plot_Mfcc(file_name):
    fig=plt.figure(figsize=(5,5))
    audio,sfreq = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y=audio, sr=sfreq)
    plt.title('MFCC')
    # plt.colorbar()
    # img=librosa.display.specshow(mfcc)
    plt.hist(mfcc, bins = 6)
    plt.savefig('./static/Mfcc.png')
    fig=plt.figure(figsize=(5,5))
    audio1,sfreq1 = librosa.load('Habiba-sample13.wav')
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sfreq1)
    plt.hist(mfcc1 , bins = 6)
    plt.savefig('./static/MfccHabiba.png')
    mfccImg = True
    return mfccImg


def listmfcc(mfcclist):
    mfcc_mean_list = []
    for i in mfcclist:
        mfcc_mean_list.append(np.mean(i))
    mfcc_20 = mfcc_mean_list[:20]
    return mfcc_20


def plot_MfccSpeech(file_name,speech):
    mfcceoutxspeech,mfcceoutspeech = plottingspeech(speech)
    fig=plt.figure(figsize=(5,5))
    audio,sfreq = librosa.load(file_name)
    mfccout = mfcc.mfcc(audio,sfreq,nfft=20)
    audioRawda,sfreqRawda = librosa.load('Rawda\RawdaOpenthedoor-sample0.wav')
    mfccRawda = mfcc.mfcc(audioRawda, sfreqRawda,nfft=20)
    plt.title('MFCC for Speech')
    mfccoutxspeech1 = listmfcc(mfccout)
    mfccRawda = listmfcc(mfccRawda)
    mfccoutxspeech = mfccoutxspeech1[18]
    mfccoutspeech = mfccoutxspeech1[4]
    plt.scatter(x=mfcceoutxspeech , y=mfcceoutspeech,color = 'purple',label='output')
    plt.axhline(y= mfccRawda[18],color = 'red',label = 'Open the door')
    plt.xlabel('MFCC18')
    plt.ylabel('MFCC4')
    plt.legend(loc ='upper right')
    plt.savefig('./static/Mfcc_Delta.png')
    Mfcc_Delta = True
    return Mfcc_Delta

def plotFeatures_mellogscale(file_name):
    fig= plt.figure(figsize=(4,4))
    audio,sfreq = librosa.load(file_name)
    melfb = librosa.filters.mel(sr=sfreq, n_fft=1200)

    img = librosa.display.specshow(melfb, x_axis='linear')
 
    plt.title("Mel Scale")
    plt.ylabel('Frequencies in Mel')
    plt.xlabel('Frequencies in HZ')
    plt.savefig('./static/featuresmelscale.png')
    featuresImg = True
    return featuresImg

def plotScores(score,speakers,flagPlot):
    plt.rcParams["figure.figsize"] = [5.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    if(max(score)== score[0]):
        colors=['purple','grey','grey']
    elif (max(score)==score[1]):
        colors=['grey','purple','grey']
    elif (max(score)==score[2]):
        colors=['grey','grey','purple']
    if flagPlot:
        colors = ['grey','grey','grey']

    precentageScore = np.round(100- np.abs(score),1)
    plt.figure()
    plt.title("Models' score")
    p1 = plt.bar(speakers, precentageScore ,color = colors ,width = 0.4)
    for rect1 in p1:
        height = rect1.get_height()
        plt.annotate( "{}%".format(height),(rect1.get_x() + rect1.get_width()/2, height+0.05),ha="center",va="bottom",fontsize=9)

    plt.savefig('./static/scoresPredict.png')
    scoreImg = True
    return scoreImg 


def plot_rmse(file_name,speaker):
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    fig= plt.figure(figsize=(5,5))
    # plt.rcParams['font.size'] = '20'
    audiohabibe,sfreqhabiba = librosa.load('Habiba.wav')
    audiorawda,sfreqrawda = librosa.load('Rawda\RawdaOpenthedoor-sample1.wav')
    audioshrioq,sfreqshiroq = librosa.load('Shiroq.wav')
    audiooutput,sfreqoutput = librosa.load(file_name)

    rms_audio_output= rms(audiooutput, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms_audio_habiba= rms(audiohabibe, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms_audio_rawda= rms(audiorawda, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms_audio_shirouq= rms(audioshrioq, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    frames = range(len(rms_audio_output))
    frameshabiba = range(len(rms_audio_habiba))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
    thabiba = librosa.frames_to_time(frameshabiba, hop_length=HOP_LENGTH)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audiooutput, hop_length=HOP_LENGTH)),
                            ref=np.max)
    if speaker == 'Habiba':
        plt.plot(rms_audio_habiba[:len(rms_audio_output)],rms_audio_output,color = 'green')
    elif speaker == 'shirouq':
        plt.plot(rms_audio_shirouq[:len(rms_audio_output)],rms_audio_output,color = 'green')
    elif speaker == 'Rawda':
        plt.plot(rms_audio_rawda[:len(rms_audio_output)],rms_audio_output,color = 'green')
    else:
        plt.plot(rms_audio_output,color = 'green')


    plt.plot(rms_audio_habiba,rms_audio_habiba, color="purple",label='habiba')
    plt.plot(rms_audio_shirouq,rms_audio_shirouq, color="blue",label ='shiroq')
    plt.plot(rms_audio_rawda,rms_audio_rawda, color="red",label ='red')
    plt.legend(loc = 'upper right')
    plt.xlabel('Recorded data')
    plt.ylabel('Output')

    plt.ylim((-0.5, 0.5))
    plt.xlim((0, 1))

    plt.title("RMS Energy")
    plt.savefig('./static/rmse.png')
    rmse = True
    return rmse



def plotMfccSpeaker(file_name,plotting,plottingspeech):
    mfcceoutx,mfcceout = plotout(plotting)
    # mfcceoutxspeech,mfcceoutspeech = plotoutspeech(plottingspeech)
    fig=plt.figure(figsize=(5,5))
    audio,sfreq = librosa.load(file_name)
    mfccout = mfcc.mfcc(audio,sfreq,nfft=20)
    audioRawda,sfreqRawda = librosa.load('Rawda\RawdaOpenthedoor-sample0.wav')
    mfccRawda = mfcc.mfcc(audioRawda, sfreqRawda,nfft=20)
    audioHabiba,sfreqHabiba = librosa.load('Habiba\habibaopenthedoor-sample1.wav')
    mfccHabiba = mfcc.mfcc(audioHabiba, sfreqHabiba,nfft=20)
    audioShiroq,sfreqShiroq = librosa.load('Shiroq.wav')
    mfccShiroq = mfcc.mfcc(audioShiroq, sfreqShiroq,nfft=20)
    plt.title('MFCC')
    # mfcc_mean_listoutput = []
    # for i in mfccout:
    #     mfcc_mean_listoutput.append(np.mean(i))
    mfcc_20 = listmfcc(mfccout)
    mfcc_20rawda = listmfcc(mfccRawda)
    mfcc_20Habiba = listmfcc(mfccHabiba)
    mfcc_20shiroq = listmfcc(mfccShiroq)
    mfccoutx=[mfcc_20[13]]
    mfccout=[mfcc_20[17]]
    plt.scatter(x=mfcceoutx , y=mfcceout,color = 'purple',label='output')
    plt.axhline(y=mfcc_20shiroq[13],color ='orange',label ='Shiroq')
    plt.axhline(y= mfcc_20Habiba[13],color = 'blue',label='habiba')
    plt.axhline(y= mfcc_20rawda[13]+0.01,color = 'red',label = 'rawda')
    plt.xlabel('MFCC17')
    plt.ylabel('MFCC13')
    # plt.scatter(x = coeff ,y = mfcc_20Habiba,color ='red',label ='Habiba')
    plt.legend(loc ='upper right')
    plt.savefig('./static/chroma.png')

    mfccImg = True
    return mfccImg


def plot_delta(filename):
     fig = plt.figure(figsize=(5,5))
     y, sr = librosa.load(filename)
     mfcc = librosa.feature.mfcc(y=y, sr=sr)
     mfcc_delta = librosa.feature.delta(mfcc)
     librosa.display.specshow(mfcc_delta)
     plt.colorbar()
     plt.title('Mfcc Delta')
     plt.savefig('./static/mfccdelta.png')
     mfccdelta = True
     return mfccdelta
