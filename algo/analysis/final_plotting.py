import matplotlib.pyplot as plt
from utils.path_info import *
import pandas as pd
from pathlib import Path
from os.path import join

plt.rcParams.update({'font.size': 16})


def plot_pv():
    pv_all = pd.read_csv(Path(join(final_results, "pv_all.csv")), index_col=0, parse_dates=True)
    plt.figure(figsize=(12, 4))

    for column in pv_all.columns:
        plt.plot(pv_all.index, pv_all[column], label=column, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance Over Time")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(join(final_results, "pv.png")))
    return


if __name__ == '__main__':
    plot_pv()
