import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def load_data():
    """Load a sample dataset (Iris)"""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "KNN": KNeighborsClassifier(),
    }

    for name, model in models.items():
        score = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{name} Accuracy: {score:.2f}")
