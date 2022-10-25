from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Models to be trained and evaluated.

def getModel(selectedModel):
    switch={
        "SVM":make_pipeline(StandardScaler(), SVC(gamma="auto", probability = True)),
        "Perceptron":make_pipeline(StandardScaler(), SGDClassifier(loss = "perceptron")),
        "Logistic Regression":make_pipeline(StandardScaler(), SGDClassifier(loss = "log")),
        "LDA":make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) 
    }
    
    model = switch.get(selectedModel)
    
    return model