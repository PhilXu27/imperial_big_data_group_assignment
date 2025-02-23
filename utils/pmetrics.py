import pandas as pd
import numpy as np
from utils.pmetrics_utils import \
    max_drawdown_cal, get_exact_calendar_date, calculate_var_cvar


def __performance_metrics_reformat_helper(performance_metrics):
    reformat_metrics = performance_metrics.copy()

    def percentage_reformat(x):
        return format(x * 100, '.2f') + "%"

    def absolute_val_reformat(x):
        return format(x, '.2f')

    def datetime_reformat(x):
        return x.strftime("%Y-%m-%d")

    reformat_metrics["Total Return"] = performance_metrics["Total Return"].apply(percentage_reformat)
    reformat_metrics["Annualized Return"] = performance_metrics["Annualized Return"].apply(percentage_reformat)
    reformat_metrics["Annualized Volatility"] = performance_metrics["Annualized Volatility"].apply(percentage_reformat)
    reformat_metrics["Downside Volatility"] = performance_metrics["Downside Volatility"].apply(percentage_reformat)
    reformat_metrics["Sharpe Ratio"] = performance_metrics["Sharpe Ratio"].apply(absolute_val_reformat)
    reformat_metrics["Sortino Ratio"] = performance_metrics["Sortino Ratio"].apply(absolute_val_reformat)
    reformat_metrics["Max Drawdown"] = performance_metrics["Max Drawdown"].apply(percentage_reformat)
    reformat_metrics["Calmar Ratio"] = performance_metrics["Calmar Ratio"].apply(absolute_val_reformat)
    reformat_metrics["Max Drawdown Day"] = performance_metrics["Max Drawdown Day"].apply(datetime_reformat)
    reformat_metrics["Max Drawdown Start"] = performance_metrics["Max Drawdown Start"].apply(datetime_reformat)
    reformat_metrics["Max Drawdown Recover"] = performance_metrics["Max Drawdown Recover"].apply(datetime_reformat)
    return reformat_metrics


def __performance_metrics_helper(input_data, base_per_year, is_var=True, is_mdd_detail=True, **kwargs):
    rf = kwargs.get("rf", 0.0)
    return_df = input_data.pct_change()
    p_metrics = pd.DataFrame()
    p_metrics["Total Return"] = input_data.iloc[-1] / input_data.iloc[0] - 1
    p_metrics["Annualized Return"] = np.power(p_metrics["Total Return"] + 1, (base_per_year / input_data.shape[0])) - 1
    p_metrics["Annualized Volatility"] = return_df.std(axis=0) * np.power(base_per_year, 0.5)
    p_metrics["Downside Volatility"] = return_df[return_df < -10e-12].std(axis=0) * np.power(base_per_year, 0.5)
    p_metrics["Sharpe Ratio"] = (p_metrics["Annualized Return"] - rf) / p_metrics["Annualized Volatility"]
    p_metrics["Sortino Ratio"] = (p_metrics["Annualized Return"] - rf) / p_metrics["Downside Volatility"]
    p_metrics["Max Drawdown"] = (input_data.div(input_data.cummax()) - 1.).min()
    p_metrics["Calmar Ratio"] = - (p_metrics["Annualized Return"] - rf) / p_metrics["Max Drawdown"]

    if is_mdd_detail:
        p_metrics["Max Drawdown Day"], p_metrics["Max Drawdown Start"], p_metrics["Max Drawdown Recover"] = \
            max_drawdown_cal(input_data)[0], max_drawdown_cal(input_data)[1], max_drawdown_cal(input_data)[2]
    if is_var:
        p_metrics["Historical 1-day 99% VaR"], p_metrics["Historical 99% 1-day CVaR"] = calculate_var_cvar(
            return_df, 0.99, 1, "historical")
        p_metrics["Historical 1-day 95% VaR"], p_metrics["Historical 95% 1-day CVaR"] = calculate_var_cvar(
            return_df, 0.95, 1, "historical")
        p_metrics["Statistical 1-day 99% VaR"], p_metrics["Statistical 99% 1-day CVaR"] = calculate_var_cvar(
            return_df, 0.99, 1, "statistical")
        p_metrics["Statistical 1-day 95% VaR"], p_metrics["Statistical 95% 1-day CVaR"] = calculate_var_cvar(
            return_df, 0.95, 1, "statistical")

    return p_metrics


def calculate_pmetrics(
        input_data: pd.DataFrame,
        start: str = "1900-01-01", end: str = "2100-01-01", frequency: str = "D",
        freq_per_year: str or int = "actual", exchange: str = None,
        is_reformat: bool = False, is_var=True, is_mdd_detail=True,
        **kwargs
) -> pd.DataFrame or (pd.DataFrame, pd.DataFrame):
    """
    Main function for performance metrics calculation.

    If start time and end time are provided, this func will only calculate the performance from *start* to *end*.
    Most users will only use this function to calculate performance for daily data, so frequency is defaulted as "D".

    Args:
        input_data: dataframe of shape (n, k). n days, k columns (assets).
        start:
        end:
        frequency: "D", "W", "M", "Q".
        freq_per_year: param to identify how many "D" or "W"... per year, which is so important for annualized rtn&vol.
                default value is "actual" that it will use datamaster for get the real numer of active trading days
                during that period. "default" use 245 when exchange is "CN" and 252 w/o exchange or exchange == "US",
                or if freq_per_year is any digit, which is identified by user, it will use this num directly.
        exchange: provide info to identify which exchange calendar dates it will use.
        is_reformat: if True, it will reformat info a prefixed form, easy to read.

    Returns:
        performance_metrics
    """
    if isinstance(input_data, pd.Series):
        input_data = pd.DataFrame(input_data)
    input_data = input_data.loc[start: end]
    if frequency == "Q":  # Quarterly
        performance_metrics = __performance_metrics_helper(
            input_data, 4, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
        )
    elif frequency == "M":  # Monthly
        performance_metrics = __performance_metrics_helper(
            input_data, 12, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
        )
    elif frequency == "W":  # Weekly
        if isinstance(freq_per_year, int):
            performance_metrics = __performance_metrics_helper(
                input_data, freq_per_year, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
            )
        elif freq_per_year == "default":
            performance_metrics = __performance_metrics_helper(
                input_data, 365.25 / 7, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
            )
        else:
            raise ValueError("Invalid fre_per_year option: {}".format(freq_per_year))
    elif frequency == "D":  # Daily
        if isinstance(freq_per_year, int):
            performance_metrics = __performance_metrics_helper(
                input_data, freq_per_year, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
            )
        elif freq_per_year == "default":
            if not exchange or exchange == "US":
                performance_metrics = __performance_metrics_helper(
                    input_data, 245, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
                )
            elif exchange == "CN":
                performance_metrics = __performance_metrics_helper(
                    input_data, 245, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
                )
            else:
                raise ValueError("Invalid exchange provided: {}".format(exchange))
        elif freq_per_year == "actual":
            if not exchange:
                # Since datamaster.exchange_calendar requires to provide a str for param: "ex", this will check it.
                raise ValueError("Please provide which exchange calendar you want to use.")
            start_date, end_date = list(input_data.index)[0], list(input_data.index)[-1]
            actual_active_trading_days, actual_calendars = get_exact_calendar_date(
                start_date=start_date, end_date=end_date, exchange=exchange)
            input_data = input_data[input_data.index.isin(actual_calendars)]
            actual_calendar_helper = pd.DataFrame(index=actual_calendars)
            input_data = pd.concat([actual_calendar_helper, input_data], axis=1).ffill()
            performance_metrics = __performance_metrics_helper(
                input_data, actual_active_trading_days, is_var=is_var, is_mdd_detail=is_mdd_detail, **kwargs
            )
        else:
            raise ValueError("Invalid fre_per_year option: {}".format(freq_per_year))
    else:
        raise ValueError("Invalid frequency option: {}".format(frequency))

    if is_reformat:
        reformat_metrics = __performance_metrics_reformat_helper(performance_metrics)
        return performance_metrics, reformat_metrics

    return performance_metrics
