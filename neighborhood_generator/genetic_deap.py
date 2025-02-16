import multiprocessing as mp
import warnings

import numpy as np
from numpy import linalg

from deap import algorithms, base, creator, tools

warnings.filterwarnings("ignore")


def mutGaussVec(chromosome, mu: np.ndarray, sigma: np.ndarray, indpb: float):
    c = np.asarray(chromosome)
    probs = np.random.random(c.shape)
    mutations = np.random.normal(loc=mu, scale=sigma, size=c.shape)
    c[probs <= indpb] = mutations[probs <= indpb]
    chromosome[:] = c

    return (chromosome,)


def evaluate_deap(
    chromosome: list,
    point: np.ndarray,
    target: int,
    blackbox,
    epsilon: float = 0.0,
    alpha: float = 0.0,
):
    assert alpha >= 0.0 and alpha <= 1.0

    # classification
    chromosome = np.asarray(chromosome)
    synth_class = blackbox.predict(chromosome.reshape(1, -1))

    # compute euclidean distance
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification penalty
    right_target = 1.0 - alpha if target == synth_class[0] else alpha

    # check the epsilon distance
    if distance <= epsilon:
        return (np.inf,)

    return (distance / right_target,)


def create_toolbox_deap(X: np.ndarray) -> base.Toolbox:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=getattr(creator, "FitnessMin"))

    toolbox = base.Toolbox()

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register(
        "mutate",
        mutGaussVec,
        mu=X.mean(axis=0),
        sigma=X.std(axis=0),
        indpb=0.5,
    )

    return toolbox


def update_toolbox_deap(
    toolbox: base.Toolbox, point: np.ndarray, target: int, blackbox
):
    # update the toolbox with new generation and evaluation
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
        "evaluate",
        evaluate_deap,
        point=point,
        target=target,
        blackbox=blackbox,
    )

    return toolbox


def run_deap(toolbox: base.Toolbox, population_size: int, workers_num: int):
    # run the genetic algorithm on one point with a specific target class
    hof = tools.HallOfFame(int(0.1 * population_size), similar=np.array_equal)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    pool = mp.Pool(workers_num)
    toolbox.register("map", pool.map)

    population = getattr(toolbox, "population")(n=population_size)
    population, _, _ = algorithms.eaSimple(
        population=population,
        toolbox=toolbox,
        cxpb=0.8,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hof,
    )

    pool.close()
    pool.join()

    return hof, stats
