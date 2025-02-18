from os.path import join
from pathlib import Path
import pandas as pd
from utils.path_info import raw_data_path, main_data_path


def load_raw_data():
    raw_price = pd.read_csv(Path(join(raw_data_path, "raw_price.csv")), index_col=0, parse_dates=True)
    adjusted_price = pd.read_csv(Path(join(raw_data_path, "adjusted_price.csv")), index_col=0, parse_dates=True)
    effective_trading_dates = pd.read_csv(
        Path(join(main_data_path, "effective_trading_calendar.csv")), index_col=0, parse_dates=True, dayfirst=True
    )
    return raw_price, adjusted_price, effective_trading_dates


def process_raw_price(dt, calendar):
    start, end = dt.index[0], dt.index[-1]
    dt = dt.resample("D").last()
    new_calendar = calendar.loc[start: end]
    dt = dt.loc[new_calendar.index]
    dt = dt.ffill()
    dt.to_csv(Path(join(main_data_path, "raw_price.csv")))
    return


def process_adjusted_price(dt, calendar):
    start, end = dt.index[0], dt.index[-1]
    dt = dt.resample("D").last()
    new_calendar = calendar.loc[start: end]
    dt = dt.loc[new_calendar.index]
    dt = dt.ffill()
    dt_rtn = dt.pct_change().dropna(how="all")
    dt.to_csv(Path(join(main_data_path, "adjusted_price.csv")))
    dt_rtn.to_csv(Path(join(main_data_path, "adjusted_return.csv")))
    return


def main():
    raw_price, adjusted_price, effective_trading_dates = load_raw_data()
    process_raw_price(raw_price, effective_trading_dates)
    process_adjusted_price(adjusted_price, effective_trading_dates)
    return


if __name__ == '__main__':
    main()
