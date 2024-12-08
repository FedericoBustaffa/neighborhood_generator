import argparse

import blackbox
import genetic
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from ppga import algorithms, base, log


def build_stats_df(results: dict[str, list], blackbox) -> dict[str, list]:
    stats = {
        "point": [],
        "class": [],
        "target": [],
        "min_fitness": [],
        "mean_fitness": [],
        "max_fitness": [],
        "accuracy": [],
    }

    stats["point"] = results["point"]
    stats["class"] = results["class"]
    stats["target"] = results["target"]

    for hof, target in zip(results["hall_of_fame"], results["target"]):
        scores = np.asarray([ind.fitness for ind in hof])
        scores = scores[~np.isinf(scores)]

        synth_points = np.asarray([ind.chromosome for ind in hof])
        outcomes = blackbox.predict(synth_points)

        stats["min_fitness"].append(scores.min())
        stats["mean_fitness"].append(scores.mean())
        stats["max_fitness"].append(scores.max())
        stats["accuracy"].append(len(outcomes[outcomes == target]) / len(hof))

    return stats


def explain(X: np.ndarray, y: np.ndarray, blackbox, pop_size: int) -> dict[str, list]:
    logger = log.getUserLogger()

    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.create_toolbox(X)

    # every entry represents a single run of the genetic algorithm
    results = {
        "point": [],
        "class": [],
        "target": [],
        "hall_of_fame": [],
    }

    for i, (point, outcome) in enumerate(zip(X, y)):
        for target in outcomes:
            logger.debug(f"point: {point}")
            logger.debug(f"outcome: {outcome}")
            logger.debug(f"target: {target}")

            # update the point for the generation
            toolbox.set_generation(genetic.generate_copy, point=point)

            # update the evaluation function
            toolbox.set_evaluation(
                genetic.evaluate,
                point=point,
                target=target,
                blackbox=blackbox,
                epsilon=0.0,
                alpha=0.0,
            )

            # run the genetic algorithm on one point with a specific target class
            hof = base.HallOfFame(pop_size)
            population, stats = algorithms.pelitist(
                toolbox=toolbox,
                population_size=pop_size,
                keep=0.1,
                cxpb=0.8,
                mutpb=0.2,
                max_generations=100,
                hall_of_fame=hof,
                workers_num=16,
            )

            results["point"].append(i)
            results["class"].append(outcome)
            results["target"].append(target)
            results["hall_of_fame"].append(hof)

    return build_stats_df(results, blackbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        type=str,
        help="filepath to the dataset",
    )

    parser.add_argument(
        "--log",
        type=str,
        choices=["debug", "benchmark", "info", "warning", "error"],
        default="info",
        help="set the debug level",
    )

    args = parser.parse_args()

    # logger from ppga
    logger = log.getUserLogger()
    logger.setLevel(args.log.upper())

    # call the blackbox run with the given dataset
    svm = SVC()
    predictions = blackbox.make_predictions(svm, args.filepath)

    X = predictions[[k for k in predictions if k != "outcome"]].to_numpy()
    y = predictions["outcome"].to_numpy()

    stats = pd.DataFrame(explain(X, y, svm, 2000))
    print(stats)
