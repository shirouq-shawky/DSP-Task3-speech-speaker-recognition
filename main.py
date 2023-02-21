from flask import Flask,request,render_template,redirect, url_for
from functions import *
import wave
from a import *
from matplotlib.figure import Figure
from librosa import power_to_db , util
import matplotlib.pyplot as plt
import librosa.display


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('allow.html')

@app.route("/preProcessing")
def test_model():
    flagForOthers = False
    modelsLoading,nameOfModel,featuresArray,toCheck = modelsExtract("trained_models")
    rightOne = np.argmax(toCheck) 
    flagList = toCheck - max(toCheck)
    for result in range(len(flagList)):
        if flagList[result] == 0:
            continue
        if abs(flagList[result])<0.4:
            flagForOthers = True

    flagPlot = False
    if flagForOthers:
         speaker = "Other"
         flagPlot = True
    else:
        speaker = nameOfModel[rightOne]


    modelsLoadingOfSpeech,nameOfModelOfSpeech,featuresArrayOfspeech,toCheckOfSpeech = modelsExtract("trained_model_words")
    rightSpeech = np.argmax(toCheckOfSpeech)
    if nameOfModelOfSpeech[rightSpeech] != 'Openthedoor' or speaker == "Other":
        nameOfModelOfSpeech[rightSpeech] = 'Not allowed to enter'
    else:
        nameOfModelOfSpeech[rightSpeech] = 'Allowed to Enter'
    rmsimg = plot_rmse('output.wav',speaker)
    featuresImg = plotFeatures_mellogscale('output.wav')
    scoreImg = plotScores(toCheck,nameOfModel,flagPlot)
    mfccImg = plot_Mfcc('output.wav')
    melImg = plot_melspectrogram('output.wav')
    rmsimg = plotfunction('output.wav',speaker)
    spectralimg = plotMfccSpeaker('output.wav',speaker,nameOfModelOfSpeech[rightSpeech])
    mfccdelta = plot_delta('output.wav')
    return render_template('allow.html',speaker = "{}".format(speaker),flag =flagForOthers, speech = "{}".format( nameOfModelOfSpeech[rightSpeech]),plot = featuresImg,scoreImg = scoreImg,mfccImg = mfccImg ,melImg = melImg,spectralimg = spectralimg,rmsimg = rmsimg,mfccdelta = mfccdelta)

    
@app.route("/",methods = ['GET','POST'])
def record_audio_test():
    if request.method == "POST":
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 2.5
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,input_device_index = 1,
                            frames_per_buffer=CHUNK)
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                Recordframes.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()

        OUTPUT_FILENAME="output.wav"
        waveFile = wave.open(OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()
        return redirect(url_for('test_model'))






if __name__ == "__main__":
    app.run(debug = True)


