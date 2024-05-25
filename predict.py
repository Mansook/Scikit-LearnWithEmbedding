import joblib
from embedding import embedding


def PredictByTrainedModel(text):
    clf = joblib.load('svm_modelWithLargeData.pkl')
    embeds = embedding(text)
    predictions = clf.predict(embeds)
    return predictions


print(PredictByTrainedModel("이걸 욕으로 구분못해?"))
