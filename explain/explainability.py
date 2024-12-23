import numpy as np

from explain import genetic
from ppga import base


def explain_one_point(
    toolbox: base.ToolBox,
    population_size: int,
    point: np.ndarray,
    outcome: int,
    blackbox,
    target: int,
    workers_num: int,
):
    # update the point for the generation
    toolbox = genetic.update_toolbox(toolbox, point, target, blackbox)
    hof = genetic.run(toolbox, population_size, workers_num)

    synth_points, scores = list(zip(*[(ind.chromosome, ind.fitness) for ind in hof]))
    scores = np.asarray(scores)
    synth_outcomes = blackbox.predict(np.asarray(synth_points))

    return {
        "point": hash(tuple(point)),
        "class": outcome,
        "target": target,
        "model": str(blackbox).removesuffix("()"),
        "min_fitness": scores.min(),
        "mean_fitness": scores.mean(),
        "fitness_std": scores.std(),
        "max_fitness": scores.max(),
        "accuracy": len(synth_outcomes[synth_outcomes == target]) / len(synth_outcomes),
    }, hof  # REMOVE


def explain(
    blackbox, X: np.ndarray, y: np.ndarray, population_size: int, workers_num: int
) -> dict[str, list]:
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.create_toolbox(X)

    # dataset of results
    df = {
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
            stats, _ = explain_one_point(
                toolbox, population_size, point, outcome, blackbox, target, workers_num
            )
            for k in stats:
                if k == "point":
                    df[k].append(i)
                else:
                    df[k].append(stats[k])

    return df
