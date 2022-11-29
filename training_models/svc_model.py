from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm, metrics


def svc_model(train_subset, test_subset, test_size: Optional[int] = 0.2):
    print("SVM model")
    X_train, X_test, y_train, y_test = train_test_split(train_subset, test_subset, test_size=test_size)
    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    svc = svm.SVC(kernel='poly')
    svc.fit(X_train, y_train)

    y_model_outcome = svc.predict(X_test)
    print(f"model: {y_model_outcome}")
    print(f"goal: {y_test}")
    print(f" Score: {metrics.f1_score(y_test, y_model_outcome, average=None)}")
    print(f"Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
