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
