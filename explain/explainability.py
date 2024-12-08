import numpy as np

from explain import genetic


def build_stats_df(results: list[dict], blackbox) -> dict[str, list]:
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


def explain(blackbox, X: np.ndarray, y: np.ndarray) -> dict[str, list]:
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.toolbox(X)

    results = []
    for point, outcome in zip(X, y):
        for target in outcomes:
            # update the point for the generation
            genetic.update_toolbox(toolbox, point, target)
            hof = genetic.run(toolbox, len(y))
            one_run = {
                "point": point,
                "class": outcome,
                "target": target,
                "hall_of_fame": hof,
            }
            results.append(one_run)

    return build_stats_df(results, blackbox)
