import pandas as pd
from utils.path_info import data_path
from os.path import join
from pathlib import Path

price = pd.read_csv(Path(join(data_path, "dirty_data", "ftse_close_price.csv")), index_col=0, parse_dates=True)
price = price.ffill()
print(price.columns[price.notna().all()])

adj_price = pd.read_csv(Path(join(data_path, "dirty_data", "ftse_close_price_adjusted.csv")), index_col=0, parse_dates=True)
adj_price = adj_price.ffill()
print(adj_price.columns[adj_price.notna().all()])

effective_trading_calendar = pd.read_csv(Path(join(data_path, "dirty_data", "effective_trading_calendar.csv")), index_col=0, parse_dates=True, dayfirst=True)

demo_price = price[price.columns[price.notna().all()]]
demo_price = demo_price.resample("D").last()
demo_price = demo_price.loc[effective_trading_calendar.index]
demo_price.to_csv(Path(join(data_path, "demo_data", "price.csv")))

demo_adjusted_price = adj_price[adj_price.columns[adj_price.notna().all()]]
demo_adjusted_price = demo_adjusted_price.resample("D").last()
demo_adjusted_price = demo_adjusted_price.loc[effective_trading_calendar.index]
demo_adjusted_price = demo_adjusted_price.pct_change().dropna(how="all")

demo_adjusted_price.to_csv(Path(join(data_path, "demo_data", "adjusted_return.csv")))
