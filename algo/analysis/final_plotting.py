import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils.path_info import *
import pandas as pd
from pathlib import Path
from os.path import join

plt.rcParams.update({'font.size': 16})


def merge_pmetrics():
    low_pmetrics = pd.read_csv(Path(join(final_results, "low_stock_pmetrics.csv")), index_col=0, parse_dates=True)
    high_pmetrics = pd.read_csv(Path(join(final_results, "high_stock_pmetrics.csv")), index_col=0, parse_dates=True)
    low_pmetrics.index = [f"Portfolio of {n} Lowest-Priced Stocks" for n in low_pmetrics.index]
    high_pmetrics.index = [f"Portfolio of {n} Highest-Priced Stocks" for n in high_pmetrics.index]
    pmetrics_all = pd.concat([low_pmetrics, high_pmetrics], axis=0)
    pmetrics_all.to_csv(Path(join(final_results, "pmetrics_all.csv")))
    return


def plot_high_low_stock_pv():
    low_pv_all = pd.read_csv(Path(join(final_results, "low_stock_port_pv.csv")), index_col=0, parse_dates=True)
    high_pv_all = pd.read_csv(Path(join(final_results, "high_stock_port_pv.csv")), index_col=0, parse_dates=True)
    low_pv_all.columns = [f"Portfolio of {n} Lowest-Priced Stocks" for n in low_pv_all.columns]
    high_pv_all.columns = [f"Portfolio of {n} Highest-Priced Stocks" for n in high_pv_all.columns]

    green_cmap = plt.get_cmap("Greens")
    red_cmap = plt.get_cmap("Reds")

    plt.figure(figsize=(12, 4))
    for i, column in enumerate(low_pv_all.columns):
        plt.plot(low_pv_all.index, low_pv_all[column], alpha=0.7, label=column, color=green_cmap((i + 1)/ len(low_pv_all.columns)))
    for i, column in enumerate(high_pv_all.columns):
        plt.plot(high_pv_all.index, high_pv_all[column], alpha=0.7, label=column, color=red_cmap((i + 1) / len(high_pv_all.columns)))

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance: Lowest vs Highest Priced Stocks")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(join(final_results, "high_vs_low_pv.png")))
    return


def plot_pv():
    pv_all = pd.read_csv(Path(join(final_results, "pv_all.csv")), index_col=0, parse_dates=True)
    plt.figure(figsize=(12, 4))

    for column in pv_all.columns:
        plt.plot(pv_all.index, pv_all[column], label=column, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance Over Time")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(join(final_results, "pv.png")))
    return


def plot_mae():
    pv_all = pd.read_csv(Path(join(final_results, "passive_portfolio_performance.csv")), index_col=0, parse_dates=True)
    custom_labels = ['baseline', '25', '50', '100', '250', '500', '2.5k', '5k', '10k']
    selected_labels = custom_labels[1:]
    pv_all.columns = custom_labels
    mean_absolute_error = pv_all.loc["mean_absolute_error", selected_labels]
    baseline_mae = pv_all.loc["mean_absolute_error", "baseline"]
    plt.figure(figsize=(10, 8))
    plt.scatter(selected_labels, mean_absolute_error, color='green', alpha=1, label="Mean Absolute Error (MAE)")
    plt.plot(selected_labels, mean_absolute_error, linestyle=":", color="green", alpha=0.7)
    plt.axhline(y=baseline_mae, color='red', linestyle='dashed', label="Baseline MAE")
    plt.xlabel('Portfolio Budget')
    plt.ylabel('Mean Absolute Error (Daily)')
    plt.title('Mean Absolute Error vs Portfolio Budget')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.2f}%'))
    plt.tight_layout()
    plt.savefig(Path(join(final_results, "mae.png")))
    return


def plot_te():
    pv_all = pd.read_csv(Path(join(final_results, "passive_portfolio_performance.csv")), index_col=0, parse_dates=True)
    custom_labels = ['baseline', '25', '50', '100', '250', '500', '2.5k', '5k', '10k']
    selected_labels = custom_labels[1:]
    pv_all.columns = custom_labels
    tracking_error = pv_all.loc["tracking_error", selected_labels]
    baseline_mae = pv_all.loc["tracking_error", "baseline"]
    plt.figure(figsize=(10, 8))
    plt.scatter(selected_labels, tracking_error, color='green', alpha=1, label="Tracking Error")
    plt.plot(selected_labels, tracking_error, linestyle=":", color="green", alpha=0.7)
    plt.axhline(y=baseline_mae, color='red', linestyle='dashed', label="Baseline Tracking Error")
    plt.xlabel('Portfolio Budget')
    plt.ylabel('Tracking Error (Daily)')
    plt.title('Tracking Error vs Portfolio Budget')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.2f}%'))
    plt.tight_layout()
    plt.savefig(Path(join(final_results, "te.png")))
    return


if __name__ == '__main__':
    plot_pv()
    plot_mae()
    plot_te()
    plot_high_low_stock_pv()
    merge_pmetrics()
