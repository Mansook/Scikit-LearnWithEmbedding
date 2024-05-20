import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from dataParsing import labels
import optunaModel

# 엑셀 파일에서 임베딩 데이터 로드
df = pd.read_excel('embeddings.xlsx', header=None)
embeddings = df.values

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)


def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    kernel = trial.suggest_categorical(
        'kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    # SVM 모델 생성
    clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

    # 교차 검증을 통한 성능 평가
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


# Optuna 스터디 생성 및 최적화 실행
study = optunaModel.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 출력
print('Best hyperparameters: {}'.format(study.best_params))

# 최적의 하이퍼파라미터를 사용하여 모델 학습 및 평가
best_params = study.best_params
clf = SVC(**best_params, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 성능 평가
print(classification_report(y_test, y_pred))
