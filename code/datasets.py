from scipy.signal import spectrogram
from statistics import mean
import numpy as np

#Definitions for dataset generation.

#Create training data from braking event EEG via these steps:
#Get segments of braking event EEG.
#Covert to PSD.
#Store PSD components of each segment in variable for training.
def createDatasetFromEEGEvents(timestamps, data, samplingRate, numberOfPSDComponents = 4):
    dt = 1/samplingRate #Time increment in seconds
    
    dt1_index = 0
    dt2_index = int(100/1000/dt) #Covert timestamps to seconds and divde by time increment to get index of datapoint at 100 ms.
    baselineCorrection_eeg = mean(data[dt1_index:dt2_index+1])
    
    #Define variables to split data in first 1/2 for training and second 1/2 for validation
    brakingEvent_eeg_PSD_train = []
    brakingEvent_eeg_PSD_val = []

    dt = 1/samplingRate #Time increment in seconds

    for time in timestamps: #Iterate through event timestamps in milliseconds
        #index = int(time/1000/dt) #Covert timestamps to seconds and divde by time increment to get index of datapoint
        dt1_index = int((time-300)/1000/dt) #Index of datapoint 300 ms before event datapoint.
        dt2_index = int((time+1200)/1000/dt) #Index of datapoint 1200 ms after event datapoint.

        brakingEvent_eeg = data[dt1_index:dt2_index+1]-baselineCorrection_eeg
        #Normalize signal data WRT to max and find generate power spectral density 
        freq_data, time_data, pwr_spectral_density_data = spectrogram(
                                                            np.array([brakingEvent_eeg]),
                                                            samplingRate
                                                            )
        if time < len(data)*1000*dt/2:
            #brakingEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
            brakingEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
            continue
        #brakingEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
        brakingEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
        
    return brakingEvent_eeg_PSD_train, brakingEvent_eeg_PSD_val


#Create baseline training EEG data containing no braking event EEG via these steps:
#Get 100 ms EEG segment at beginning of data to use for baseline correction.
#Get segments of EEG without braking events and subract 100 ms EEG segment.
#Covert to PSD.
#Store PSD components of each segment in variable for training.
def createDatasetFromEEGWithoutEvents(timestamps, data, samplingRate, numberOfPSDComponents=4):
    dt = 1/samplingRate #Time increment in seconds
    
    dt1_index = 0
    dt2_index = int(100/1000/dt) #Covert timestamps to seconds and divde by time increment to get index of datapoint at 100 ms.
    baselineCorrection_eeg = mean(data[dt1_index:dt2_index+1])

    noEvent_eeg_PSD_train = []
    noEvent_eeg_PSD_val = []

    for i in range(0, len(timestamps)): #Iterate through all event timestamps in milliseconds
        if timestamps[i][0] < 4500: #Skip iteration if there is not enough time to get an EEG segment between time of first datapoint and time of first event. 
            continue

        if i > 0:
            if timestamps[i][0]-timestamps[i-1][0] < 7500: #Skip iteration if there is not enough time to get EEG segment between current and previous timestamps.
                continue

        numberOfSegments = int((timestamps[i][0]-timestamps[i-1][0]-6000)/2000) #Calculate how many user-specified EEG segments can fit between two events.
        
        for segmentNum in range(0, numberOfSegments):
            #Add 500 ms between each EEG segment, except for segment closest in time to event
            dt1_index = int((timestamps[i][0]-5000-(2000*segmentNum)-500)/1000/dt) #500 represents 500 ms offset between each EEG segment.
            dt2_index = int((timestamps[i][0]-3000-(2000*segmentNum))/1000/dt)
            
            noEvent_eeg = data[dt1_index:dt2_index+1]-baselineCorrection_eeg #Get EEG segment immediately prior to current event.
           
            #Normalize signal data WRT to max and find generate power spectral density 
            freq_data, time_data, pwr_spectral_density_data = spectrogram( 
                                                                np.array([noEvent_eeg]),#/max(noEvent_eeg)]), 
                                                                samplingRate
                                                                )
            
            if timestamps[i][0] < len(data)*1000*dt/2: #Check if timestamp is less half than total time of EEG data.
                #noEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
                noEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
                continue
            #noEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
            noEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
           
        if i == len(timestamps): #If iteration reaches last event timestamp, set indices to get any possible EEG segment beyond timestamp.
            numberOfSegments = int((len(data)*1000*dt/2-timestamps[i][0])/2000) #Calculate how many user-specified EEG segments can fit between two events.

            for segmentNum in range(0, numberOfSegments):
                #Add 500 ms between each EEG segment, except for segment closest in time to event
                dt1_index = int((timestamps[i][0]+3000+(2000*segmentNum))/1000/dt)
                dt2_index = int((timestamps[i][0]+5000+(2000*segmentNum)-500)/1000/dt) #500 represents 500 ms offset between each EEG segment.

                noEvent_eeg = data[dt1_index:dt2_index+1]  #Get EEG segment 
                #Normalize signal data WRT to max and find generate power spectral density 
                freq_data, time_data, pwr_spectral_density_data = spectrogram( 
                                                                    np.array([noEvent_eeg]),#/max(noEvent_eeg)]), 
                                                                    samplingRate
                                                                    )
                if timestamps[i][0] < len(data)*1000*dt/2:
                    #noEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
                    noEvent_eeg_PSD_train.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
                    continue
                #noEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[-numberOfPSDComponents:None].tolist())
                noEvent_eeg_PSD_val.append(np.sort(np.sum(pwr_spectral_density_data[0],1))[0:numberOfPSDComponents].tolist())
        
    return noEvent_eeg_PSD_train, noEvent_eeg_PSD_val

#Combine datasets with and without braking events.
def createDatasets(brakingEvent_eeg_PSD_train, noEvent_eeg_PSD_train, brakingEvent_eeg_PSD_val, noEvent_eeg_PSD_val):
    #Label = 0 indicates no event; label = 1 indicates EEG braking event
    trainData = np.concatenate((brakingEvent_eeg_PSD_train, noEvent_eeg_PSD_train))
    trainLabels_event = np.ones(len(brakingEvent_eeg_PSD_train),dtype=int) 
    trainLabels_noEvent = np.zeros(len(noEvent_eeg_PSD_train),dtype=int) 
    trainLabels = np.concatenate((trainLabels_event, trainLabels_noEvent))

    valData = np.concatenate((brakingEvent_eeg_PSD_val, noEvent_eeg_PSD_val))
    valLabels_event = np.ones(len(brakingEvent_eeg_PSD_val),dtype=int) 
    valLabels_noEvent = np.zeros(len(noEvent_eeg_PSD_val),dtype=int) 
    valLabels = np.concatenate((valLabels_event, valLabels_noEvent))
    
    return trainData, trainLabels, valData, valLabels
