from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils.path_info import *
from algo.sparse_replication import load_data
import numpy as np

plt.rcParams.update({'font.size': 16})


def plotting(portfolio_sharpe_ratio, max_sharpe_ratio, single_sharpe_ratio, single_price):
    portfolio_values = [int(x.split("_")[1]) // 100 for x in portfolio_sharpe_ratio.index]
    max_sharpe = portfolio_sharpe_ratio.max(axis=1).values
    median_sharpe = portfolio_sharpe_ratio.median(axis=1).values
    # Convert portfolio values to a discrete numeric index for even spacing
    x_labels = [single_price] + [str(v) if v < 1000 else str(v // 1000) + "k" for v in portfolio_values] + ["Market"]
    x_positions = list(range(len(portfolio_values) + 2))

    # Plotting again with discrete x-axis
    fig, ax = plt.subplots(figsize=(12, 7))
    # Add the single stock Sharpe ratio at the first position
    ax.bar(x_positions[0], single_sharpe_ratio, color="lightgrey", width=0.8, align="center")
    ax.bar(x_positions[1:-1], max_sharpe, color="lightgrey", label="Max Sharpe Ratio", width=0.8, align="center")
    ax.plot(x_positions[1:-1], median_sharpe, marker="^", color="darkblue", markersize=12, linestyle="",
            label="Median Sharpe Ratio")
    for pool in portfolio_sharpe_ratio.columns:
        ax.plot(x_positions[1:-1], portfolio_sharpe_ratio[pool].values, marker="o", color="blue", markersize=4, linestyle="")

    # Add the market Sharpe ratio (black bar) at the last position
    ax.bar(x_positions[-1], max_sharpe_ratio, color="black", width=0.8)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Portfolio Value (GBP)")
    ax.set_ylabel("Sharpe Ratio (Annualized)")
    ax.legend()

    # Show plot
    plt.tight_layout()
    save_path = Path(join(final_results, "replicating_paper_results.png"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    return


def calculate_sharpe_ratio():
    return


def most_highly_related_stocks(max_sharpe_ratio_return):
    for test_mode in ["ftse_250"]:
        _, h_matrix, p_matrix = load_data(test_mode, start_date, end_date)
        corr_results = {}
        for stock in h_matrix.columns:
            tmp_return = h_matrix[[stock]]
            if tmp_return.isna().any().any():
                continue
            corr_results[stock] = pd.concat([max_sharpe_ratio_return, tmp_return], axis=1).corr().iloc[0, 1]
    best_stock = max(corr_results, key=corr_results.get)
    print(best_stock, corr_results[best_stock])
    single_stock_return = h_matrix[best_stock]
    single_stock_return.to_csv(Path(join(paper_results, "highly_correlated_stocks.csv")))
    single_stock_sharpe_ratio = (np.power(single_stock_return.mean() + 1, 252) - 1) / (
        single_stock_return.std() * np.power(252, .5))
    max_price = p_matrix[best_stock].iloc[-1] / 100
    return single_stock_return, single_stock_sharpe_ratio, max_price


def load_results(start_date, end_date, validating_start_date, test_mode_list):
    portfolio_sharpe_ratio = pd.DataFrame(columns=test_mode_list)
    for test_mode in test_mode_list:
        experiment_save_path = Path(join(
            paper_results, f"START_{start_date}_END_{end_date}",
            f"VAL_{validating_start_date}", f"{test_mode}"
        ))
        tmp_returns = pd.read_csv(Path(join(experiment_save_path, "my_portfolio.csv")), index_col=0, parse_dates=True)
        portfolio_sharpe_ratio[test_mode] = tmp_returns.apply(
            lambda x: (np.power(x.mean() + 1, 252) - 1) / (x.std() * np.power(252, 0.5)), axis=0
        )
    return portfolio_sharpe_ratio


def load_max_sharpe_ratio_return(start_date, end_date, validating_start_date):
    max_sharpe_save_path = Path(
        join(paper_results, f"START_{start_date}_END_{end_date}", f"VAL_{validating_start_date}", "ftse_250")
    )
    max_sharpe_ratio_return = pd.read_csv(
        Path(join(max_sharpe_save_path, "max_sharpe_ratio_portfolio.csv")), index_col=0, parse_dates=True
    )
    max_sharpe_ratio = (np.power(max_sharpe_ratio_return.mean() + 1, 252) - 1) / (
        max_sharpe_ratio_return.std() * np.power(252, .5))
    return max_sharpe_ratio_return, max_sharpe_ratio


def main():
    portfolio_sharpe_ratio = load_results(
        start_date, end_date, validating_start_date, test_mode_list=[
            "ftse_250_top_5_market_cap", "ftse_250_top_10_market_cap", "ftse_250_top_20_market_cap",
            "ftse_250_top_50_market_cap", "ftse_100", "ftse_250"
    ])
    max_sharpe_ratio_return, max_sharpe_ratio = load_max_sharpe_ratio_return(
        start_date, end_date, validating_start_date)
    single_stock_return, single_stock_sharpe_ratio, max_price = most_highly_related_stocks(max_sharpe_ratio_return)
    plotting(portfolio_sharpe_ratio, max_sharpe_ratio, single_stock_sharpe_ratio, max_price)
    return


if __name__ == '__main__':
    start_date, end_date = "2020-01-31", "2025-01-31"
    validating_start_date = "2024-01-31"
    main()
