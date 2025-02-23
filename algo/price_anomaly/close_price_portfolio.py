import pandas as pd

from algo.sparse_replication import load_data
from utils.pmetrics import calculate_pmetrics


def high_low_price_portfolio():
    _, return_matrix, price_matrix = load_data("ftse_250", "2010-12-24", "2025-01-31")
    price_matrix = price_matrix.dropna(how="any", axis=1)
    return_matrix = return_matrix.dropna(how="any", axis=1)

    holding_period = 21
    num_of_stocks = 20
    low_returns = pd.DataFrame()
    high_returns = pd.DataFrame()
    medium_returns = pd.DataFrame()
    for i in range(return_matrix.shape[0] // holding_period + 1):
        tmp_return = return_matrix[i * holding_period: (i + 1) * holding_period]
        tmp_price = price_matrix.iloc[i * holding_period]
        # Sort stocks based on average price
        sorted_stocks = tmp_price.sort_values()
        lowest_x_stocks = sorted_stocks.head(num_of_stocks).index
        medium_x_stocks = sorted_stocks.head(sorted_stocks.shape[0] // 2 + num_of_stocks).index
        highest_x_stocks = sorted_stocks.tail(num_of_stocks).index
        low_returns = pd.concat([low_returns, pd.DataFrame(tmp_return[lowest_x_stocks].mean(axis=1), columns=["low"])], axis=0)
        high_returns = pd.concat([high_returns, pd.DataFrame(tmp_return[highest_x_stocks].mean(axis=1), columns=["high"])], axis=0)

    lowest_x_stocks_pv = (1 + low_returns).cumprod()
    highest_x_stocks_pv = (1 + high_returns).cumprod()
    return


if __name__ == '__main__':
    high_low_price_portfolio()
