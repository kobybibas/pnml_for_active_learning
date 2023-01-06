import numpy as np
import pandas as pd


def process_results(run_list, api):
    dfs = []
    for run_name in run_list:
        print(run_name)
        run = api.run(run_name)
        df = run.history(keys=["training_set_size", "test_acc", "test_loss"])
        df.dropna().sort_values(by="training_set_size").set_index("training_set_size")
        dfs.append(df)

    return dfs


def execute_df_rollout(dfs, rolling_size):
    df = pd.concat(dfs)

    test_acc_mean = (
        df.groupby("training_set_size")["test_acc"]
        .mean()
        .sort_index()
        .rolling(rolling_size)
        .mean()
    )
    test_acc_std = (
        df.groupby("training_set_size")["test_acc"]
        .rolling(rolling_size)
        .std()
        .sort_index()
        .mean()
    )

    low = test_acc_mean + 1.960 * test_acc_std / np.sqrt(len(dfs))
    high = test_acc_mean - 1.960 * test_acc_std / np.sqrt(len(dfs))

    results = {"mean": test_acc_mean, "low": low, "high": high}
    return results
