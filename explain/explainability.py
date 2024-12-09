import numpy as np

from explain import genetic


def build_stats_df(results: list[dict], blackbox) -> dict[str, list]:
    # first processing the hall of fame results
    targets = []
    hofs = []
    for i in results:
        targets.append(i["target"])
        hofs.append(i["hall_of_fame"])

    # extract fitness scores
    scores = np.array([[i.fitness for i in h] for h in hofs])

    # extract the sythetic points
    synth_points = [[i.chromosome for i in h] for h in hofs]

    # generates an outcomes batch
    synth_outcomes = [blackbox.predict(np.asarray(X)) for X in synth_points]

    # evaluates fitness and accuracy

    return {}


def explain(blackbox, X: np.ndarray, y: np.ndarray) -> dict[str, list]:
    # collect all the possible outcomes
    outcomes = np.unique(y)

    # create a toolbox with fixed params
    toolbox = genetic.toolbox(X)

    results = []
    for point, outcome in zip(X, y):
        for target in outcomes:
            # update the point for the generation
            genetic.update_toolbox(toolbox, point, target, blackbox)
            hof = genetic.run(toolbox, 100)
            one_run = {
                "point": point,
                "class": outcome,
                "target": target,
                "hall_of_fame": hof,
            }
            results.append(one_run)

    return build_stats_df(results, blackbox)
