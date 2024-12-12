import warnings

import numpy as np
from numpy import linalg, random

from ppga import algorithms, base, tools

warnings.filterwarnings("ignore")


# generation by copy
def generate_copy(point: np.ndarray) -> np.ndarray:
    return point.copy()


# normal distribution generation
def generate_normal(mu, sigma):
    return random.normal(mu, sigma, size=mu.shape)


# evaluation with target
def evaluate(
    chromosome: np.ndarray,
    point: np.ndarray,
    target: np.ndarray,
    blackbox,
    epsilon: float = 0.0,
    alpha: float = 0.0,
):
    assert alpha >= 0.0 and alpha <= 1.0

    # classification
    synth_class = blackbox.predict(chromosome.reshape(1, -1))

    # compute euclidean distance
    distance = linalg.norm(chromosome - point, ord=2)

    # compute classification penalty
    right_target = 1.0 - alpha if target == synth_class[0] else alpha

    # check the epsilon distance
    if distance <= epsilon:
        return (np.inf,)

    return (distance / right_target,)


def toolbox(X: np.ndarray) -> base.ToolBox:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    toolbox = base.ToolBox()
    toolbox.set_weights((-1.0,))
    toolbox.set_selection(tools.sel_tournament, tournsize=3)
    toolbox.set_crossover(tools.cx_one_point)
    toolbox.set_mutation(tools.mut_normal, mu=mu, sigma=sigma, indpb=0.8)

    return toolbox


def update_toolbox(toolbox: base.ToolBox, point: np.ndarray, target: int, blackbox):
    # update the toolbox with new generation and evaluation
    toolbox.set_generation(generate_copy, point=point)

    toolbox.set_evaluation(
        evaluate,
        point=point,
        target=target,
        blackbox=blackbox,
        epsilon=0.0,
        alpha=0.0,
    )

    return toolbox


def run(toolbox: base.ToolBox, population_size: int):
    # run the genetic algorithm on one point with a specific target class
    hof = base.HallOfFame(population_size)
    population, stats = algorithms.pelitist(
        toolbox=toolbox,
        population_size=population_size,
        keep=0.1,
        cxpb=0.8,
        mutpb=0.2,
        max_generations=100,
        hall_of_fame=hof,
    )

    return hof
