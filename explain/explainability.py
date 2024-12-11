import numpy as np
import pandas as pd

from explain import genetic
from ppga import base


def build_stats_df(results: list[dict], blackbox) -> pd.DataFrame:
    # first processing the hall of fame results
    targets = []
    hofs = []
    for i in results:
        targets.append(i["target"])
        hofs.append(i["hall_of_fame"])

    # extract fitness scores
    scores = [np.array([i.fitness for i in h]) for h in hofs]

    # extract the sythetic points
    synth_points = [[i.chromosome for i in h] for h in hofs]

    # generates an outcomes batch
    synth_outcomes = [blackbox.predict(np.asarray(X)) for X in synth_points]

    # build the final dataframe
    data = {
        "point": [hash(tuple(i["point"])) for i in results],
        "class": [],
        "target": [],
        "blackbox": [],
        "min_fitness": [np.min(s[~np.isinf(s)]) for s in scores],
        "mean_fitness": [np.mean(s[~np.isinf(s)]) for s in scores],
        "max_fitness": [np.max(s[~np.isinf(s)]) for s in scores],
        "accuracy": [],
    }

    for i, (r, so, t) in enumerate(zip(results, synth_outcomes, targets)):
        data["class"].append(r["class"])
        data["target"].append(t)
        data["blackbox"].append(str(blackbox).removesuffix("()"))
        data["accuracy"].append(len(so[so == t]) / len(so))

    return pd.DataFrame(data)


def explain_one_point(
    toolbox: base.ToolBox,
    population_size: int,
    point: np.ndarray,
    outcome: int,
    blackbox,
    outcomes: np.ndarray,
):
    results = []
    for target in outcomes:
        # update the point for the generation
        genetic.update_toolbox(toolbox, point, target, blackbox)
        hof = genetic.run(toolbox, population_size)
        one_run = {
            "point": point,
            "class": outcome,
            "target": target,
            "hall_of_fame": hof,
        }
        results.append(one_run)

    return results


def explain(
    blackbox, X: np.ndarray, y: np.ndarray, population_size: int
) -> pd.DataFrame:
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.toolbox(X)

    results = []
    for point, outcome in zip(X, y):
        one_point_explain = explain_one_point(
            toolbox, population_size, point, outcome, blackbox, outcomes
        )
        results.extend(one_point_explain)

    return build_stats_df(results, blackbox)
