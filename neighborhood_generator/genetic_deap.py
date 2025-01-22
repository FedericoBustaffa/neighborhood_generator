import multiprocessing as mp
import warnings

import numpy as np

from deap import algorithms, base, creator, tools
from neighborhood_generator import genetic

warnings.filterwarnings("ignore")


def mutGaussVec(
    chromosome: np.ndarray, mu: np.ndarray, sigma: np.ndarray, indpb: float
):
    probs = np.random.random(chromosome.shape)
    mutations = np.random.normal(loc=mu, scale=sigma, size=chromosome.shape)
    chromosome[probs <= indpb] = mutations[probs <= indpb]

    return (chromosome,)


def create_toolbox_deap(X: np.ndarray) -> base.Toolbox:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=getattr(creator, "FitnessMin"))

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
        genetic.evaluate,
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
