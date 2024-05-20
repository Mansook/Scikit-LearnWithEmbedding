import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from dataParsing import labels

# 엑셀 파일에서 임베딩 데이터 로드
df = pd.read_excel('embeddingl.xlsx', header=None)
embeddings = df.values

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

hyperparameters = {'C': 0.5348371932446554,
                   'kernel': 'linear', 'gamma': 'auto'}
# SVM 분류 모델 생성 및 훈련
clf = SVC(kernel='linear', gamma='auto', C=0.5348371932446554, random_state=42)
clf.fit(X_train, y_train)

# 학습된 모델 저장
joblib.dump(clf, 'svm_model.pkl')

# 예측 및 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
