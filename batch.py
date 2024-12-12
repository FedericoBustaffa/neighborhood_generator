import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import explain
from ppga import log


def make_predictions(model, data: pd.DataFrame, test_size: float = 0.3):
    features_index = [
        col for col in data.columns if col.startswith("feature_")]
    X = data[features_index].to_numpy()
    y = data["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # train the model
    model.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = np.asarray(model.predict(X_test))

    return np.asarray(X_test), to_explain


if __name__ == "__main__":
    # set the debug log level of the core logger
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log", type=str, help="set the log level of the core logger")
    args = parser.parse_args()
    log.setLevel(args.log.upper())

    # blackboxes for testing
    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]

    # get the datasets
    filepaths = [fp for fp in os.listdir("datasets")]
    datasets = [pd.read_csv(f"datasets/{fp}") for fp in filepaths]

    # for every dataset run the blackbox and make explainations
    for df in datasets:
        for bb in blackboxes:
            test_set, predictions = make_predictions(bb, df, 0.3)
            explaination = explain.explain(bb, test_set, predictions, 500)
            print(explaination)
