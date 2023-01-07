from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import numpy as np

#Definitions for model development.
def trainModel(model, trainData, trainLabels):
    
    model.fit(np.array(trainData), trainLabels)
    
    return model

def evaluateModel(model, selectedModel, valData, valLabels):
    #Validate model
    #eventPredictions = model.predict(valData).tolist()
    #Accuracy as percent of correct predictions out of total predictions.
    #accuracy = abs(sum(eventPredictions-valLabels))/len(eventPredictions)
    
    #Choose compatible prediction method based on model type.
    '''
    switch={
        "Perceptron":model.decision_function(np.array(valData)),
        "SVM":model.predict_proba(np.array(valData))[:, 1],
        "Logistic Regression":model.predict_proba(np.array(valData))[:, 1],
        "LDA":model.predict_proba(np.array(valData))[:, 1]
    }
    '''
    if selectedModel == "Perceptron":
        AUCaccuracy = roc_auc_score(valLabels, model.decision_function(np.array(valData)))
    else:
        AUCaccuracy = roc_auc_score(valLabels, model.predict_proba(np.array(valData))[:, 1])
    
    #Area under the curve score
    #AUCaccuracy = roc_auc_score(valLabels, switch.get(selectedModel))
    
    return AUCaccuracy #,accuracy

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU

def trainMLPModel(trainData, trainLabels, valData, valLabels):
    # Number of input columns 
    n_inputs = trainData.shape[1]
    visible = Input(shape=(n_inputs,))
    # Hidden layers
    x = Dense(64)(visible)
    x = LeakyReLU()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dense(16)(x)
    x = LeakyReLU()(x)
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    # Define model
    model = Model(inputs=visible, outputs=output)
    # Compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Accuracy()])
    # Rescale data to be between 0 and 1
    scaler = MinMaxScaler()
    scaler.fit(trainData)
    trainData = scaler.transform(trainData)
    valData = scaler.transform(valData)
    # Train model
    model.fit(trainData, trainLabels, epochs=200, batch_size=32, verbose=2, validation_split = 0.2)
    # Validate model
    valResults = evaluateMLPModel(model, valData, valLabels)
    return valResults

def evaluateMLPModel(model, valData, valLabels):
    valResults = model.evaluate(valData, valLabels, batch_size=1)
    return valResults
