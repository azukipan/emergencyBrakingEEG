"""
This code produces an EEG dataset according to the methodology described by Haufe et al. in "EEG potentials predict upcoming emergency brakings during simulated driving." dx.doi.org/10.1088/1741-2560/8/5/056001. Instead of the original modeling approach, we train and test other approaches for all test subjects from the Haufe et al. study. 

The original EEG data are part of a dataset from Haufe et al. called "Emergency braking during simulated driving." The dataset is available at http://bnci-horizon-2020.eu/database/data-sets
"""
import numpy as np
import h5py
import subprocess
import os
import sys
from glob import glob
from tqdm import tqdm
from statistics import mean
from pathlib import Path
    
from datasets import createDatasetFromEEGEvents, createDatasetFromEEGWithoutEvents, createDatasets
from modelDev import trainModel, evaluateModel
from models import getModel

def main():
    #Get test subject data
    print("Downloading dataset...")
    os.chdir(r"../EEG_Dataset_Haufe/")
    cmd = "wget -i links.txt -c --no-check-certificate"
    subprocess.check_call(cmd, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
    os.chdir(r"../code/")
    testSubjectDataFilePaths =  glob('../EEG_Dataset_Haufe/*.mat')

    PSDComponentsUpperLimit = 129 #129 = Maximum number of power spectral density (PSD) components for samplingRate = 200 hz.

    channelsOfInterest = ['POz'] 
    #['P9'] 
    #['P10'] 
    #['FCz'] 
    #['CPz']
    #['P9', 'P10', 'CPz', 'FCz'] 
    #['all channels'] 
    
    #Setup progress bar
    pbar = tqdm(range(1, PSDComponentsUpperLimit+1)) #Number of PSD components from 1 to maximum in increments of 20.
    #pbar = tqdm(range(129, 130)) 
    
    modelOptions = ["Perceptron",
                    "LDA", 
                    "SVM",
                    "Logistic Regression"
                   ]
    
    #Create model development results folder.
    p = Path("../results")
    p.mkdir(exist_ok=True)
    
    for selectedModel in modelOptions:
        #Create log file for AUCs and number of PSD components.
        AUCLog = open("../results/{}_AUC.csv".format(selectedModel), "a+")
        AUCLog.write("Number of Components,Area Under Curve\n") #Headers
        AUCLog.close()

        pbar.set_description("Model:{}\nCalculating AUC for each number of PSD components.".format(selectedModel))
        for numberOfPSDComponents in pbar: #Iterate through all user-specified time segments to obtain AUCs.
            allTestSubjectAUCs = []
            for path in testSubjectDataFilePaths:
                f = h5py.File(path,'r')
                '''
                Read, sort and assign experimental to variables for: 
                signal channel names: cnt.clab
                sampling frequency: cnt.fs
                time-series data: cnt.x
                '''
                cnt = f.get('cnt')
                cnt.clab = np.array(cnt['clab'])
                cnt.fs = np.array(cnt['fs'])
                cnt.x = np.array(cnt['x']) 

                samplingRate = cnt.fs[0][0] #Down-/upsample rate for all data = 200Hz.

                #Read data for events corresponding to experimental data.
                mrk = f.get('mrk')
                mrk.classNames = mrk['className']
                mrk.time = mrk['time']
                mrk.y = mrk['y']
                mrk.events = mrk['event']

                #Find all car braking events (brake lights of lead vehicle turn on) and store corresponding timestamps.
                carBrakeTime = []
                for i in range(0, len(mrk.y)):
                    if mrk.y[i][1] == 1: #Check if car is braking, i.e. y[i] = 1
                        carBrakeTime.append(mrk.time[i][0]) #Store timestamp 

                #Create train and validation datasets
                brakingEvent_eeg_PSD_train = []
                brakingEvent_eeg_PSD_val = []

                noEvent_eeg_PSD_train = []
                noEvent_eeg_PSD_val = []

                #Iterate through all EEG channels.
                for channelNum in range(0, 61):

                    if channelsOfInterest != ['all channels']:
                        ref = cnt.clab[channelNum][0]
                        channelName = bytes(np.array(cnt[ref]).ravel().tolist()).decode('UTF-8')

                        if channelName not in channelsOfInterest:
                            continue

                    data = cnt.x[channelNum]
                    event_eeg_PSD_train, event_eeg_PSD_val = createDatasetFromEEGEvents(carBrakeTime, 
                                                                                        data, 
                                                                                        samplingRate,
                                                                                        numberOfPSDComponents)
                    _noEvent_eeg_PSD_train, _noEvent_eeg_PSD_val = createDatasetFromEEGWithoutEvents(mrk.time, 
                                                                                                     data, 
                                                                                                     samplingRate, 
                                                                                                     numberOfPSDComponents)

                    for array in event_eeg_PSD_train: brakingEvent_eeg_PSD_train.append(array)
                    for array in event_eeg_PSD_val: brakingEvent_eeg_PSD_val.append(array)
                    for array in _noEvent_eeg_PSD_train: noEvent_eeg_PSD_train.append(array)
                    for array in _noEvent_eeg_PSD_val: noEvent_eeg_PSD_val.append(array)
                
                trainData, trainLabels, valData, valLabels = createDatasets(brakingEvent_eeg_PSD_train,
                                                                            noEvent_eeg_PSD_train,
                                                                            brakingEvent_eeg_PSD_val,
                                                                            noEvent_eeg_PSD_val)
                model = getModel(selectedModel)
                
                trainedModel = trainModel(model, trainData, trainLabels)
                AUCaccuracy = evaluateModel(trainedModel, selectedModel, valData, valLabels)
                allTestSubjectAUCs.append(AUCaccuracy)
            grandAverageAUC = round(mean(allTestSubjectAUCs), 3)
            print("Number of PSD components = ", numberOfPSDComponents, " | AUC accuracy = ", grandAverageAUC)
            
            #Save AUC and number of PSD components.
            AUCLog = open("../results/{}_AUC.csv".format(selectedModel), "a+")
            AUCLog.write("{},{}\n".format(numberOfPSDComponents, grandAverageAUC))
            AUCLog.close()
        
if __name__ == "__main__":
    main()
