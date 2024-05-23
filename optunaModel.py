import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from dataParsing import labels
import optunaModel


df = pd.read_excel('embeddings.xlsx', header=None)
embeddings = df.values

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)


def optimize_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


def optimize_naive_bayes(trial):
    var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-9, 1e-5)
    clf = GaussianNB(var_smoothing=var_smoothing)
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


def optimize_decision_tree(trial):
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


def optimize_kneighbors(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
    algorithm = trial.suggest_categorical(
        'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


def optimize_adaboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1.0)
    clf = AdaBoostClassifier(n_estimators=n_estimators,
                             learning_rate=learning_rate, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=3,
                            scoring='accuracy').mean()
    return score


# 각각의 모델에 대해 Optuna 스터디 생성 및 최적화 실행
models = {
    "RandomForest": optimize_random_forest,
    "NaiveBayes": optimize_naive_bayes,
    "DecisionTree": optimize_decision_tree,
    "KNeighbors": optimize_kneighbors,
    "AdaBoost": optimize_adaboost
}

for model_name, optimization_function in models.items():
    print(f"Optimizing {model_name}...")
    study = optunaModel.create_study(direction='maximize')
    study.optimize(optimization_function, n_trials=50)
    print(f'Best hyperparameters for {model_name}: {study.best_params}')
    print()

# 최적의 하이퍼파라미터를 사용하여 모델 학습 및 평가
best_params = {}

for model_name, optimization_function in models.items():
    study = optunaModel.create_study(direction='maximize')
    study.optimize(optimization_function, n_trials=50)
    best_params[model_name] = study.best_params

# 모델 학습 및 평가 함수


def train_and_evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


# Random Forest
print("Random Forest:")
clf = RandomForestClassifier(**best_params["RandomForest"], random_state=42)
train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)

# Naive Bayes
print("Naive Bayes:")
clf = GaussianNB(**best_params["NaiveBayes"])
train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)

# Decision Tree
print("Decision Tree:")
clf = DecisionTreeClassifier(**best_params["DecisionTree"], random_state=42)
train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)

# k-Nearest Neighbors
print("k-Nearest Neighbors:")
clf = KNeighborsClassifier(**best_params["KNeighbors"])
train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)

# AdaBoost
print("AdaBoost:")
clf = AdaBoostClassifier(**best_params["AdaBoost"], random_state=42)
train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)
