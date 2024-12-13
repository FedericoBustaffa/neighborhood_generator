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
    features_index = [col for col in data.columns if col.startswith("feature_")]
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
        "--log", default="info", help="set the log level of the core logger"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="specify the name of the output file without extension",
    )
    args = parser.parse_args()

    # set the core and user logger level
    log.setLevel(args.log.upper())
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    # blackboxes for testing
    blackboxes = [SVC(), MLPClassifier()]  # for fast tests
    # blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()] # for complete simulation

    # get the datasets
    # filepaths = [fp for fp in os.listdir("datasets") if fp.startswith("classification")]
    filepaths = ["classification_100_2_2_1_0.csv", "classification_100_2_2_2_0.csv"]
    datasets = [pd.read_csv(f"datasets/{fp}") for fp in filepaths]
    logger.info(f"preparing to explain {len(datasets)} datasets")

    # for every dataset run the blackbox and make explainations
    results = {
        "dataset_id": [],  # dataset features
        "simulation_id": [],
        "samples": [],
        "features": [],
        "classes": [],
        "clusters": [],
        "population_size": [],  # single genetic run features
        "point": [],
        "class": [],
        "target": [],
        "model": [],
        "min_fitness": [],  # genetic algorithm output
        "mean_fitness": [],
        "max_fitness": [],
        "accuracy": [],
    }

    population_sizes = [500, 1000]
    for i, (fp, df) in enumerate(zip(filepaths, datasets)):
        for bb in blackboxes:
            for ps in population_sizes:
                for j in range(10):
                    logger.info(f"dataset {i+1}/{len(datasets)}")
                    logger.info(f"model: {str(bb).removesuffix('()')}")
                    logger.info(f"population_size: {ps}")

                    # change test size to 0.3 for real test
                    # test_set, predictions = make_predictions(bb, df, 0.3)
                    test_set, predictions = make_predictions(bb, df, 0.05)
                    logger.info(f"predictions to explain: {len(predictions)}")

                    explaination = explain.explain(bb, test_set, predictions, 500)
                    dataset_features = fp.removesuffix(".csv").split("_")

                    results["dataset_id"].extend(
                        [i for _ in range(len(explaination["point"]))]
                    )
                    results["simulation_id"].extend(
                        [j for _ in range(len(explaination["point"]))]
                    )
                    results["samples"].extend(
                        [len(predictions) for _ in range(len(explaination["point"]))]
                    )
                    results["features"].extend(
                        [dataset_features[2] for _ in range(len(explaination["point"]))]
                    )
                    results["classes"].extend(
                        [dataset_features[3] for _ in range(len(explaination["point"]))]
                    )
                    results["clusters"].extend(
                        [dataset_features[4] for _ in range(len(explaination["point"]))]
                    )
                    results["population_size"].extend(
                        [
                            dataset_features[500]
                            for _ in range(len(explaination["point"]))
                        ]
                    )

                    for k in explaination:
                        results[k].extend(explaination[k])

    results = pd.DataFrame(results)
    print(results)

    results.to_csv(f"datasets/{args.output}.csv", header=True, index=False)
