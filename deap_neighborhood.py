import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from neighborhood_generator import genetic


def train_model(clf, df: pd.DataFrame) -> np.ndarray:
    # split train and test set
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_test = np.asarray(X_test)

    # train the model
    clf.fit(X_train, y_train)

    return np.asarray(X_test)


if __name__ == "__main__":
    df = pd.read_csv("datasets/classification_100_2_2_1_0.csv")
    X = df[["feature_1", "feature_2"]].to_numpy()
    y = df["outcome"].to_numpy()

    mlp = MLPClassifier()
    X_test = train_model(mlp, df)
    predictions = np.asarray(mlp.predict(X_test))
    point = X_test[0]
    target = predictions[0]

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
        "evaluate", genetic.evaluate, point=point, target=target, blackbox=mlp
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

    pop = getattr(toolbox, "population")(n=100)
    hof = tools.HallOfFame(100, similar=np.array_equal)
    population, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 5, halloffame=hof)

    synth_points = [ind for ind in hof]
    plt.figure(figsize=(16, 10))
    plt.title("DEAP")
    plt.scatter(
        np.asarray(X_test).T[0],
        np.asarray(X_test).T[1],
        c=predictions,
        ec="w",
        cmap="bwr",
    )
    plt.scatter(
        np.asarray(synth_points).T[0], np.asarray(synth_points).T[1], c="y", ec="w"
    )
    plt.scatter(point[0], point[1], marker="X", c="r", ec="w")

    plt.gcf().set_dpi(150)
    plt.show()
