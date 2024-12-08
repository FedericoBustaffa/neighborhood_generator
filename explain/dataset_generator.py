import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

counter = 0


def generate_dataset(
    n_samples: int, n_features: int, n_classes: int, n_clusters: int, seeds: list[int]
):
    """Generate a dataset and store it in a CSV file"""
    try:
        for seed in seeds:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=n_classes,
                n_clusters_per_class=n_clusters,
                shuffle=True,
                random_state=seed,
            )
            X = np.asarray(X)
            y = np.asarray(y)

            data = {f"feature_{i + 1}": X.T[i] for i in range(n_features)}
            data.update({"outcome": y})

            # create datasets folder if not present
            if "datasets" not in os.listdir("./"):
                os.mkdir("./datasets")

            # save trainig set
            df = pd.DataFrame(data)
            df.to_csv(
                f"./datasets/classification_{n_samples}_{n_features}_{n_classes}_{n_clusters}_{seed}.csv",
                header=True,
                index=False,
            )

            print(
                f"generated dataset with {n_samples} samples, {n_features} features, {n_classes} classes and {n_clusters} clusters per class with seed {seed}"
            )

            global counter
            counter += 1

    except ValueError:
        return


if __name__ == "__main__":
    with open("configs/light.json") as file:
        data = dict(json.load(file))

    n_samples = data["samples"]
    n_features = data["features"]
    n_classes = data["classes"]
    n_clusters = data["clusters"]
    seeds = data["seeds"]

    for samples in n_samples:
        for features in n_features:
            for classes in n_classes:
                for clusters in n_clusters:
                    generate_dataset(samples, features, classes, clusters, seeds)

    print(f"{counter} datasets generated")
