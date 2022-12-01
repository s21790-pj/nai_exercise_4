from typing import Optional

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def tree_model(train_subset, test_subset, test_size: Optional[int] = 0.2):
    # Decision Trees classifier
    print("\nTree model")
    X_train, X_test, y_train, y_test = train_test_split(train_subset, test_subset, test_size=test_size)
    treeD = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
    treeD.fit(X_train, y_train)

    # Printing Trees Classifier
    print(f"Score: {treeD.score(X_test, y_test)}")
    y_pred = treeD.predict(X_test)
    print(f"Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
