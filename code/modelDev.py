from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
from tensorflow.keras.layers import BatchNormalization

def trainMLPModel(trainData, trainLabels, valData, valLabels):
    # number of input columns for autoencoder
    n_inputs = trainData.shape[1]
    # Hidden layers
    visible = Input(shape=(n_inputs,))
    x = Dense(64)(visible)
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    # output layer
    output = Dense(1, activation='sigmoid')(x)
    #output = Dense(1, activation='sigmoid')(visible)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Accuracy()])
    # fit the autoencoder model to reconstruct input
    model.fit(trainData, trainLabels, epochs=50, batch_size=1, verbose=2, validation_split = 0.2)
    valResults = evaluateModel(model, valData, valLabels)
    return valResults

def evaluateMLPModel(model, valData, valLabels):
    valResults = model.evaluate(valData, valLabels, batch_size=1)
    return valResults
