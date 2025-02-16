import yfinance as yf
from utils.path_info import raw_data_path, stock_list_path
from pathlib import Path
from os.path import join
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


alpha_vintage_api_keyy = "4R7C4GKNZ8HQK8D8"


def load_candidate_list(which_file):
    file_path = Path(join(stock_list_path, f"{which_file}.csv"))
    candidate_list = pd.read_csv(file_path)["candidate_list"].to_list()
    return candidate_list


def query_yahoo_finance(ticker_list, file_name):
    data = yf.download(ticker_list, start='2015-04-01', end='2015-04-10')
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
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
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


def process_ftse_all_share_data():
    ftse_all_share_info = pd.read_csv(Path(join(stock_list_path, "ftse_all_share_info.csv")))
    ftse_all_share_info["yahoo_ticker"] = ftse_all_share_info["Code"].apply(
        lambda x: x + "L" if x.endswith(".") else x + ".L")
    ftse_all_share_info["bbg_ticker"] = ftse_all_share_info["Code"].apply(
        lambda x: x.replace(".", " LN Equity") if x.endswith(".") else x + " LN Equity")
    ftse_all_share_info.to_csv(Path(join(stock_list_path, "ftse_all_share_info_processed.csv")))
    yahoo_ticker_list = ftse_all_share_info[["yahoo_ticker"]]
    yahoo_ticker_list.columns = ["candidate_list"]
    bbg_ticker_list = ftse_all_share_info[["bbg_ticker"]]
    bbg_ticker_list.columns = ["candidate_list"]

    yahoo_ticker_list.to_csv(Path(join(stock_list_path, "ftse_all_share_yahoo.csv")))
    bbg_ticker_list.to_csv(Path(join(stock_list_path, "ftse_all_share_bbg.csv")))
    return


if __name__ == '__main__':
    # query_yahoo_finance(["AAPL"], "test_2")
    # ftse_all_share_list = load_candidate_list("ftse_all_share_yahoo")
    # query_yahoo_finance(ftse_all_share_list[: 10], "test_2")
    # query_alpha_vin("AAPL", "test_2")
    process_ftse_all_share_data()

