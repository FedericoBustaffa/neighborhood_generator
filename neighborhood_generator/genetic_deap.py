import warnings

import numpy as np
from deap import algorithms, base, creator, tools

from neighborhood_generator import genetic

warnings.filterwarnings("ignore")


def create_toolbox(X: np.ndarray, pool) -> base.Toolbox:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=getattr(creator, "FitnessMin"))
    # creator.create("features", float, fitness=getattr(creator, "FitnessMin"))

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        getattr(creator, "Individual"),
        getattr(toolbox, "features"),
    )

    toolbox.register(
        "population", tools.initRepeat, list, getattr(toolbox, "individual")
    )

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=X.mean(),
        sigma=X.std(),
        indpb=0.5,
    )

    toolbox.register("map", pool.map)

    return toolbox


def update_toolbox(toolbox: base.Toolbox, point: np.ndarray, target: int, blackbox):
    # update the toolbox with new generation and evaluation
    toolbox.register("features", np.copy, point)

    toolbox.register(
        "evaluate",
        genetic.evaluate,
        point=point,
        target=target,
        blackbox=blackbox,
    )

    return toolbox


def run(toolbox: base.Toolbox, population_size: int, workers_num: int):
    # run the genetic algorithm on one point with a specific target class
    hof = tools.HallOfFame(int(0.1 * population_size))
    stats = tools.Statistics()
    stats.register("min", np.min)
    stats.register("max", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    population = getattr(toolbox, "population")(n=population_size)
    population, logbook = algorithms.eaSimple(
        population=population,
        toolbox=toolbox,
        cxpb=0.8,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hof,
    )

    return hof, stats
