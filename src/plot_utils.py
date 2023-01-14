import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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


def execute_plot(plot_list, fig, ax):
    colors = ["C0", "C2", "C3", "C1"]
    min_val, max_val = 100.0, 0.0
    for i, (res, name) in enumerate(plot_list):
        ax.plot(res["mean"], label=name, color=colors[i])
        ax.fill_between(
            res["low"].index,
            res["low"].values,
            res["high"].values,
            facecolor=colors[i],
            alpha=0.1,
        )
        min_val, max_val = min(min_val, res["mean"].min()), max(
            max_val, res["mean"].max()
        )

    ax.set_ylabel("Test accuracy rate")
    ax.set_xlabel(f"Training set size")
    ax.legend(frameon=True, loc="lower right")
    ax.set_ylim(min_val, max_val + 0.01)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(res["mean"].dropna().index.min(), res["mean"].index.max())
    plt.tight_layout()
    return fig, ax


def find_max_train_save(left_df, right_df):
    left_df = left_df["mean"].reset_index()
    left_df["test_acc_round"] = left_df["test_acc"].round(3)

    right_df = right_df["mean"].reset_index()
    right_df["test_acc_round"] = right_df["test_acc"].round(3)

    merge_df = pd.merge(
        left_df, right_df, on="test_acc_round", suffixes=("_left", "_right")
    ).dropna()
    idx = (
        merge_df["training_set_size_right"] - merge_df["training_set_size_left"]
    ).argmax()
    return merge_df.iloc[idx]
