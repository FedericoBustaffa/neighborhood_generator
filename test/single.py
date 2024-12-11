import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import explain

if __name__ == "__main__":
    # build the dataset
    df = pd.read_csv("datasets/classification_100_2_2_1_0.csv")
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    # train the model
    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = clf.predict(X_test)

    df = explain.explain(clf, np.asarray(X_test), np.asarray(to_explain))
    print(df)
