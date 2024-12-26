import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import neighborhood_generator as ng
from ppga import log


def make_predictions(
    model, data: pd.DataFrame, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes in a ML model and a dataset, trains the model and returns the test
    set and the predictions.

    Args:
        model: the ML model used for classification
        data: the dataset
        test_size: the size of the test set

    Returns:
        A tuple containing the test set and the predictions
    """
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


def get_args():
    # parse CLI argument
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        choices=["RandomForestClassifier", "SVC", "MLPClassifier"],
        help="specify the model to explain",
    )

    parser.add_argument(
        "workers",
        type=int,
        help="specify the number of workers to use",
    )

    parser.add_argument(
        "output",
        default="output",
        help="specify the name of the output file without extension",
    )

    parser.add_argument(
        "--log",
        default="info",
        help="set the log level of the core logger",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # CLI arguments
    args = get_args()

    # set the core and user logger level
    log.setLevel(args.log.upper())
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    # blackboxes for testing
    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]
    model = blackboxes[
        ["RandomForestClassifier", "SVC", "MLPClassifier"].index(args.model)
    ]

    logger.info(f"start explaining of {str(model).removesuffix('()')}")

    # get the datasets
    filepaths = [fp for fp in os.listdir("datasets") if fp.startswith("classification")]
    filepaths = ["classification_100_32_2_1_0.csv"]
    datasets = [pd.read_csv(f"datasets/{fp}") for fp in filepaths]
    logger.info(f"preparing to explain {len(datasets)} datasets")

    # for every dataset run the blackbox and make explainations
    results = {
        "dataset_ID": [],  # dataset features
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
        "fitness_std": [],
        "max_fitness": [],
        "accuracy": [],
    }

    population_sizes = [1000, 2000, 4000]
    for i, (fp, df) in enumerate(zip(filepaths, datasets)):
        for ps in population_sizes:
            logger.info(f"dataset {i+1}/{len(datasets)}")
            logger.info(f"model: {str(model).removesuffix('()')}")
            logger.info(f"population_size: {ps}")

            test_set, predictions = make_predictions(model, df, 0.1)
            logger.info(f"predictions to explain: {len(predictions)}")

            # generate neighbors stats
            # to repeat at least 5 times
            stats = ng.generate(model, test_set, predictions, ps, args.workers)

            # extract dataset specs from filename
            dataset_specs = fp.removesuffix(".csv").split("_")
            samples = int(dataset_specs[1])
            features = int(dataset_specs[2])
            classes = int(dataset_specs[3])
            clusters = int(dataset_specs[4])

            results["samples"].extend([samples for _ in range(samples)])
            results["features"].extend([features for _ in range(samples)])
            results["classes"].extend([classes for _ in range(samples)])
            results["clusters"].extend([clusters for _ in range(samples)])
            results["population_size"].extend([ps for _ in range(samples)])

            for k in stats:
                results[k] = stats[k]

    results = pd.DataFrame(results)
    print(results)

    results.to_csv(f"results/{args.model}.csv", header=True, index=False)
