import pandas as pd
import joblib
from sklearn.metrics import classification_report
from embedding import embedding

# 모델 로드
clf = joblib.load('svm_model.pkl')

data = "아 진짜 어머 없네ㅋ"
embeds = embedding(data)
# 예측 수행
predictions = clf.predict(embeds)

# 예측 결과 출력
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted label: {prediction}")
