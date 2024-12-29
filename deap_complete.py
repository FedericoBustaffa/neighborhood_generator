import multiprocessing as mp

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from complete import get_args, make_predictions
from neighborhood_generator import genetic
from ppga import log


def deap_init(point, target, model, pool, workers_num: int) -> base.Toolbox:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=getattr(creator, "FitnessMin"))

    toolbox = base.Toolbox()
    toolbox.register("features", np.copy, point)
    toolbox.register(
        "individual",
        tools.initIterate,
        getattr(creator, "Individual"),
        getattr(toolbox, "features"),
    )

    toolbox.register(
        "population", tools.initRepeat, list, getattr(toolbox, "individual")
    )

    toolbox.register(
        "evaluate", genetic.evaluate, point=point, target=target, blackbox=model
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=X_test.mean(),
        sigma=X_test.std(),
        indpb=0.5,
    )

    toolbox.register("map", pool.map, processes=workers_num)

    return toolbox


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

    df = pd.read_csv("datasets/classification_100_2_2_1_0.csv")
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    pool = mp.Pool(args.workers)
    toolbox = deap_init(pool)
    pop = getattr(toolbox, "population")(n=100)
    hof = tools.HallOfFame(100, similar=np.array_equal)
    population, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 5, halloffame=hof)
