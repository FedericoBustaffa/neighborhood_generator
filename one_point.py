import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import genetic_neighborhood as gn
from ppga import log

if __name__ == "__main__":
    # set the debug log level of the core logger
    parser = argparse.ArgumentParser()

    # CLI arguments
    parser.add_argument(
        "dataset", type=str, help="select the dataset to run the simulation"
    )

    parser.add_argument("log", type=str, help="set the log level of the core logger")

    args = parser.parse_args()

    log.setLevel(args.log.upper())

    # build the dataset
    df = pd.read_csv(args.dataset)
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_test = np.asarray(X_test)

    # train the model
    clf = MLPClassifier()
    clf.fit(X_train, y_train)

    # these will be the data to explain
    to_explain = np.asarray(clf.predict(X_test))
    toolbox = gn.create_toolbox(np.asarray(X_test))

    expl1, hof1 = gn.one_point_generation(
        toolbox=toolbox,
        population_size=100,
        point=X_test[0],
        outcome=to_explain[0],
        blackbox=clf,
        target=0,
        workers_num=1,
    )
    expl2, hof2 = gn.one_point_generation(
        toolbox=toolbox,
        population_size=100,
        point=X_test[0],
        outcome=to_explain[0],
        blackbox=clf,
        target=1,
        workers_num=1,
    )

    for k in expl1:
        expl1[k] = [expl1[k], expl2[k]]

    print(pd.DataFrame(expl1))

    synth_points = np.asarray([ind.chromosome for ind in hof2])
    plt.figure(figsize=(16, 9))
    plt.scatter(X_test.T[0], X_test.T[1], c=to_explain, ec="w", cmap="bwr")
    plt.scatter(synth_points.T[0], synth_points.T[1], c="y", ec="w")
    plt.scatter(X_test.T[0][0], X_test.T[1][0], c="r", ec="w", marker="X")
    plt.show()
