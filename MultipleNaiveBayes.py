import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from dataParsing import labels

# 엑셀 파일에서 임베딩 데이터 로드
df = pd.read_excel('embeddings.xlsx', header=None)
embeddings = df.values

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

# 다항 나이브 베이즈 분류 모델 생성 및 훈련
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
