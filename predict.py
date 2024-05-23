import joblib
from embedding import embedding


def PredictByTrainedModel(text):
    clf = joblib.load('svm_model.pkl')
    embeds = embedding(text)
    predictions = clf.predict(embeds)
    return predictions
