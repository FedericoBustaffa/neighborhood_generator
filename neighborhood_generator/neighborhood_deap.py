import multiprocessing as mp

import numpy as np
from deap import base

from neighborhood_generator import genetic_deap


def single_point_deap(
    toolbox: base.Toolbox,
    population_size: int,
    point: np.ndarray,
    outcome: int,
    blackbox,
    target: int,
    workers_num: int,
) -> dict[str, float]:
    """
    Generates neighbors close to the given point and classified
    as the label given with the `target` parameter
    """
    # update the point for the generation
    toolbox = genetic_deap.update_toolbox(toolbox, point, target, blackbox)
    hof, stats = genetic_deap.run(toolbox, population_size, workers_num)

    synth_points, scores = list(zip(*[(ind.chromosome, ind.fitness) for ind in hof]))
    scores = np.asarray(scores)
    synth_outcomes = blackbox.predict(np.asarray(synth_points))

    return {
        "min_fitness": scores.min(),
        "mean_fitness": scores.mean(),
        "fitness_std": scores.std(),
        "max_fitness": scores.max(),
        "accuracy": len(synth_outcomes[synth_outcomes == target]) / len(synth_outcomes),
    }


def generate_deap(
    model,
    X: np.ndarray,
    y: np.ndarray,
    population_size: int,
    workers_num: int,
) -> dict[str, list]:
    """
    Generates synthetic neighbors for each point of the dataset.
    A neighborhood is generated for every possible outcome.
    """
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    pool = mp.Pool(workers_num)
    toolbox = genetic_deap.create_toolbox(X, pool)

    # dataset of results
    results = {
        "point": [],
        "class": [],
        "target": [],
        "model": [],
        "min_fitness": [],
        "mean_fitness": [],
        "fitness_std": [],
        "max_fitness": [],
        "accuracy": [],
    }

    for i, (point, outcome) in enumerate(zip(X, y)):
        for target in outcomes:
            stats = single_point(
                toolbox, population_size, point, outcome, model, target, workers_num
            )

            results["point"].append(i)
            results["class"].append(outcome)
            results["target"].append(target)
            results["model"].append(str(model).removesuffix("()"))
            for k in stats:
                results[k].append(stats[k])

    return results
