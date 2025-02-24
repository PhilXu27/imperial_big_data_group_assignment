import pandas as pd

from algo.sparse_replication import load_data
from utils.pmetrics import calculate_pmetrics
from utils.path_info import final_results
from pathlib import Path
from os.path import join


def high_low_price_portfolio():
    _, return_matrix, price_matrix = load_data("ftse_250", "2010-12-24", "2025-01-31")
    price_matrix = price_matrix.dropna(how="any", axis=1)
    return_matrix = return_matrix.dropna(how="any", axis=1)

    holding_period = 21
    test_n_num_stocks = [5, 10, 20]
    low_stock_all = pd.DataFrame()
    high_stock_all = pd.DataFrame()
    for num_of_stocks in test_n_num_stocks:
        low_returns = pd.DataFrame()
        high_returns = pd.DataFrame()
        for i in range(return_matrix.shape[0] // holding_period + 1):
            tmp_return = return_matrix[i * holding_period: (i + 1) * holding_period]
            tmp_price = price_matrix.iloc[i * holding_period]
            # Sort stocks based on average price
            sorted_stocks = tmp_price.sort_values()
            lowest_x_stocks = sorted_stocks.head(num_of_stocks).index
            highest_x_stocks = sorted_stocks.tail(num_of_stocks).index
            low_returns = pd.concat(
                [low_returns, pd.DataFrame(tmp_return[lowest_x_stocks].mean(axis=1), columns=[f"{num_of_stocks}"])],
                axis=0)
            high_returns = pd.concat(
                [high_returns, pd.DataFrame(tmp_return[highest_x_stocks].mean(axis=1), columns=[f"{num_of_stocks}"])],
                axis=0)
        lowest_x_stocks_pv = (1 + low_returns).cumprod()
        low_stock_all = pd.concat([low_stock_all, lowest_x_stocks_pv], axis=1)
        highest_x_stocks_pv = (1 + high_returns).cumprod()
        high_stock_all = pd.concat([high_stock_all, highest_x_stocks_pv], axis=1)
    low_stock_all.to_csv(Path(join(final_results, "low_stock_port_pv.csv")))
    high_stock_all.to_csv(Path(join(final_results, "high_stock_port_pv.csv")))

    _, low_pmetrics = calculate_pmetrics(low_stock_all, freq_per_year=252, is_reformat=True)
    _, high_pmetrics = calculate_pmetrics(high_stock_all, freq_per_year=252, is_reformat=True)
    low_pmetrics.to_csv(Path(join(final_results, "low_stock_pmetrics.csv")))
    high_pmetrics.to_csv(Path(join(final_results, "high_stock_pmetrics.csv")))
    return


if __name__ == '__main__':
    high_low_price_portfolio()
