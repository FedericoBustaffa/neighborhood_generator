import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import explain
from ppga import log

if __name__ == "__main__":
    # set the debug log level of the core logger
    parser = argparse.ArgumentParser()

    # CLI arguments
    parser.add_argument(
        "dataset", type=str, help="select the dataset to run the simulation"
    )

    parser.add_argument(
        "log", type=str, help="set the log level of the core logger")

    args = parser.parse_args()

    log.setLevel(args.log.upper())

    # build the dataset
    df = pd.read_csv(args.dataset)
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.1, random_state=0)

    # train the model
    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = clf.predict(X_test)

    df = explain.explain(clf, np.asarray(X_test), np.asarray(to_explain), 500)
    print(df)
