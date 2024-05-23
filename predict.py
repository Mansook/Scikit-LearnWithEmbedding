import pandas as pd
import joblib
from sklearn.metrics import classification_report
from embedding import embedding


clf = joblib.load('svm_model.pkl')

data = "아 고아새끼가 진짜  "
embeds = embedding(data)

predictions = clf.predict(embeds)

for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted label: {prediction}")
