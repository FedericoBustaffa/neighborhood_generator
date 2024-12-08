from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import explain

if __name__ == "__main__":
    # build the dataset
    X, y = make_classification(
        n_samples=50,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=True,
        random_state=0,
    )

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=0
    )

    # train the model
    clf = MLPClassifier()
    clf.fit(X_train)

    # these will be the data to explain
    to_explain = clf.predict(X_test)

    index, results = explain.explain(clf, X_test, to_explain)
