from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm


def max_drawdown_cal(input_origin_df):
    """
    Maximum drawdown calculator

    Args:
        input_origin_df: pv series.

    Returns:
        day: Exact day that maximum drawdown actually happened.
        start: The highest point before maximum drawdown (when max-drawdown starts)
        end: When the pv recovers.
    """
    md_cal = (input_origin_df.div(input_origin_df.cummax()) - 1.)
    md_cal_copy = md_cal.copy()
    md_cal_copy = md_cal_copy.iloc[::-1]

    day = []
    start = []
    recover = []
    for col in md_cal:
        md_day = md_cal[col].idxmin(axis=0)
        md_start = md_cal_copy[col][md_day:].idxmax(axis=0)
        md_recover = md_cal[col][md_day:].idxmax(axis=0)
        day.append(md_day)
        start.append(md_start)
        recover.append(md_recover)
    return day, start, recover


def get_exact_calendar_date(start_date, end_date, exchange):
    """

    Args:
        start_date:
        end_date:
        exchange:

    Returns:

    """
    start_year, end_year = str(start_date.year), str(end_date.year)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    from datamaster import DM_Client
    dm_client = DM_Client()
    dm_client.start()
    if datetime.now().year == int(end_year):  # If today < end_year, end_year will reset to last year.
        end_year = str(int(end_year) - 1)
    calendar = dm_client.new_exchange_calendar(
        start_date=start_year + "-01-01", end_date=end_year + "-12-31", ex=exchange, calendar_type="TRADING"
    )
    if len(calendar) < 200:  #
        temp_end = str(datetime.now().year)
        calendar = dm_client.exchange_calendar(start_date="2000-01-01", end_date=temp_end + "-12-31", ex=exchange)
        years = int(temp_end) - 2000
        actual_trading_day_per_year = len(calendar) / (years + 1)
        print("Invalid date range from {} to {}, use 2000 to {} instead \n Active trading days is {}".
              format(start_year, end_year, temp_end, actual_trading_day_per_year))
    else:
        years = int(end_year) - int(start_year)
        actual_trading_day_per_year = len(calendar) / (years + 1)
        print("Actual active trading days between {} and {} is {}".
              format(start_year, end_year, actual_trading_day_per_year))
    actual_trading_calendars = dm_client.new_exchange_calendar(
        start_date=start_date, end_date=end_date, ex=exchange, calendar_type="TRADING"
    )
    actual_trading_calendars = [pd.to_datetime(i) for i in actual_trading_calendars]
    return actual_trading_day_per_year, actual_trading_calendars


def emperical_var_cvar(return_df, alpha):
    """
    Calculates VaR (Value at Risk) and CVaR (Expected Shortfall) for a given alpha
    Arguments:
    return_df -- pd.DataFrame of returns
    alpha -- quantile for risk calculations
    Returns:
    VaR and CVaR values
    """
    # Compute VaR
    if 100 > alpha > 1:
        alpha /= 100
    returns = return_df.dropna()
    value_at_risk = returns.quantile(1 - alpha, axis=0)
    # Compute CVaR
    conditional_value_at_risk = returns[returns <= value_at_risk].mean(axis=0)
    return value_at_risk, conditional_value_at_risk


def calculate_var_cvar(return_df, alpha, time_horizon, mode=None):
    """
    Calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) for multiple portfolios.

    Parameters:
    - df: A pandas DataFrame of returns. Indices are dates and columns are portfolio names.
    - alpha: Significance level for VaR and CVaR (e.g., 0.05 for 5%)
    - time_horizon: The number of days in the future to consider for VaR and CVaR
    - mode: Either 'empirical' or 'statistical'. Determines how VaR and CVaR are calculated.
    """

    if mode == 'historical':
        # If the mode is 'empirical', we simply use the historical data to calculate VaR and CVaR.

        # Calculate VaR
        value_at_risk = return_df.quantile(1 - alpha, axis=0) * np.sqrt(time_horizon)

        # Calculate CVaR
        conditional_value_at_risk = return_df[return_df.lt(value_at_risk, axis=1)].mean(axis=0) * np.sqrt(time_horizon)

    elif mode == 'statistical':
        # If the mode is 'statistical', we assume the returns are normally distributed and use the
        # parametric method to calculate VaR and CVaR.

        # Calculate VaR
        value_at_risk = return_df.mean(axis=0) - norm.ppf(alpha) * return_df.std(axis=0) * np.sqrt(time_horizon)

        # Calculate CVaR

        conditional_value_at_risk = value_at_risk - (norm.pdf(norm.ppf(alpha)) / alpha) * return_df.std(
            axis=0) * np.sqrt(time_horizon)
    else:
        raise ValueError("Mode must be either 'historical' or 'statistical'.")

    value_at_risk = value_at_risk.apply(lambda x: x if x < 0 else 0)
    conditional_value_at_risk = conditional_value_at_risk.apply(lambda x: x if x < 0 else 0)

    return value_at_risk, conditional_value_at_risk
