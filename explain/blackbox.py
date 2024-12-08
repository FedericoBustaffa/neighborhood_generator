import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def make_predictions(blackbox, filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    X = df[[k for k in df if k != "outcome"]].to_numpy()
    y = df["outcome"].to_numpy()

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, train_size=0.7)
    X_test = np.asarray(X_test)

    blackbox.fit(X_train, y_train)
    outcomes = blackbox.predict(X_test)

    features = len([k for k in df if k != "outcome"])
    res_df = {f"feature_{i + 1}": X_test.T[i] for i in range(features)}
    res_df.update({"outcome": outcomes})

    return pd.DataFrame(res_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="filepath to the dataset")

    args = parser.parse_args()

    mlp = SVC()
    result = make_predictions(mlp, args.filepath)
    print(result)
