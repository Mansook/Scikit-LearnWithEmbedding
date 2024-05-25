import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_excel('embeddingl.xlsx', header=None)
embeddings = df.values

lb = pd.read_excel('labels.xlsx', header=None)
labels = lb.values


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)


clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'GradientBoostingwithLargeData.pkl')
