import yfinance as yf
from utils.path_info import raw_data_path
from pathlib import Path
from os.path import join
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


alpha_vintage_api_keyy = "YJM39Q3R5ME1K6KP"


def load_candidate_list(which_file):
    file_path = Path(join(raw_data_path, f"{which_file}.csv"))
    candidate_list = pd.read_csv(file_path)["candidate_list"].to_list()
    return candidate_list


def query_yahoo_finance(ticker_list, file_name):
    data = yf.download(ticker_list, start='2019-01-01', end='2025-02-04')
    raw_close_data = data["Close"]
    # adj_close_data = data["Adj Close"]
    raw_close_data.to_csv(Path(join(raw_data_path, f"{file_name}_raw_close.csv")))
    # adj_close_data.to_csv(Path(join(raw_data_path, f"{file_name}_adj_close.csv")))
    return


def query_alpha_vin(ticker_list, file_name):
    ts = TimeSeries(key=alpha_vintage_api_keyy, output_format="pandas")
    raw_close_data = pd.DataFrame()
    adj_close_data = pd.DataFrame()

    for ticker in ticker_list:
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        adj_data = data["5. adjusted close"]
        adj_data.columns = [ticker]
        raw_data = data["4. close"]
        raw_data.columns = [ticker]

        adj_close_data = pd.concat([adj_close_data, adj_data], axis=1)
        raw_close_data = pd.concat([raw_close_data, raw_data], axis=1)

    raw_close_data.to_csv(Path(join(raw_data_path, f"{file_name}_raw_close.csv")))
    adj_close_data.to_csv(Path(join(raw_data_path, f"{file_name}_adj_close.csv")))
    return


def main():
    return


if __name__ == '__main__':
    # query_yahoo_finance(["AAPL", "MSFT", "GOOG", "IBM"], "test_2")
    load_candidate_list("sp_500_list")

