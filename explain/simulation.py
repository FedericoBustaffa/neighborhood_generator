import argparse
import os

import blackbox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from explain import explain
from ppga import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=str,
        choices=["debug", "benchmark", "info", "warning", "error"],
        default="info",
        help="set the logging level",
    )

    args = parser.parse_args()

    # structure of the output
    output = {
        "blackbox": [],
        "samples": [],
        "features": [],
        "classes": [],
        "clusters": [],
        "seed": [],
        "individuals": [],
        "point": [],
        "class": [],
        "target": [],
        "min_fitness": [],
        "mean_fitness": [],
        "max_fitness": [],
        "accuracy": [],
    }

    blackboxes = [RandomForestClassifier(), SVC(), MLPClassifier()]

    filepaths = sorted(os.listdir("./datasets/"))
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    population_sizes = [1000, 2000, 4000]

    for filepath in filepaths:
        for bb in blackboxes:
            for ps in population_sizes:
                # train and predict
                predictions = blackbox.make_predictions(bb, f"datasets/{filepath}")

                # name of the blackbox
                name = str(bb).removesuffix("()")

                logger.info(
                    f"{filepath} explained with {name} and {ps} synthetic individuals"
                )

                X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
                y = predictions["outcome"].to_numpy()

                # start the explaining one the test set
                stats = explain(X, y, bb, ps)

                # add the results to the outputs
                output["blackbox"].extend([name for _ in range(len(y))])

                # take only the number of samples in the test set
                output["samples"].extend([len(y) for _ in range(len(y))])

                # other dataset params
                params = filepath.split("_")
                output["features"].extend([int(params[2]) for _ in range(len(y))])
                output["classes"].extend([int(params[3]) for _ in range(len(y))])
                output["clusters"].extend([int(params[4]) for _ in range(len(y))])
                output["seed"].extend(
                    [int(params[5].removesuffix(".csv")) for _ in range(len(y))]
                )

                # population size
                output["individuals"].extend([ps for _ in range(len(y))])

                # add all the genetic runs
                for k in stats:
                    output[k].extend(stats[k])

    pd.DataFrame(output).to_csv("datasets/results.csv", header=True, index=False)
