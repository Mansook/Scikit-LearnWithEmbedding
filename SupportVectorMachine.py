import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from dataParsing import labels


df = pd.read_excel('embeddingl.xlsx', header=None)
embeddings = df.values


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

hyperparameters = {'C': 0.5348371932446554,
                   'kernel': 'linear', 'gamma': 'auto'}

clf = SVC(kernel='linear', gamma='auto', C=0.5348371932446554, random_state=42)
clf.fit(X_train, y_train)


joblib.dump(clf, 'svm_model.pkl')


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
