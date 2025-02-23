import pandas as pd
from utils.path_info import *
from os.path import join
from pathlib import Path
from algo.sparse_replication import load_data
import numpy as np
from utils.pmetrics import calculate_pmetrics


def load_basic_info():

    return


def load_batch_data(dir_path):
    return


def load_experiment_results(dir_path, last_file_num):
    insample_info = pd.read_csv(Path(join(dir_path, f"insample_info_{last_file_num}.csv")), index_col=0, parse_dates=True)
    portfolio_positions = pd.read_csv(
        Path(join(dir_path, f"portfolio_positions_{last_file_num}.csv")), index_col=0, parse_dates=True
    )
    replicating_results = pd.read_csv(
        Path(join(dir_path, f"replicating_results_{last_file_num}.csv")), index_col=0, parse_dates=True
    )
    return insample_info, portfolio_positions, replicating_results


def get_final_results_path(start_date, end_date):
    window_size = 252 * 5
    validation_window_size = window_size // 5  # 20% Validation Set
    holding_period = 21
    experiment_save_path = Path(join(
        distributed_results, f"pool_ftse_250_{start_date}_{end_date}",
        f"holding_{holding_period}_training_{window_size}"
    ))
    baseline_save_path = Path(join(
        baseline_results, f"pool_ftse_250_{start_date}_{end_date}",
        f"holding_{holding_period}_training_{window_size}"
    ))
    return experiment_save_path, baseline_save_path


def main(start_date, end_date, analysis_on):
    print(f"analysis on: {analysis_on}")
    index_return, instruments_return, close_price = load_data("ftse_250", start_date, end_date)
    replicating_start_date = "2010-12-24"
    replicating_end_data = "2025-01-31"
    index_return = index_return.loc[replicating_start_date: replicating_end_data]
    instruments_return = instruments_return.loc[replicating_start_date: replicating_end_data]
    close_price = close_price.loc[replicating_start_date: replicating_end_data]

    results_main_dir, baseline_path = get_final_results_path(start_date, end_date)

    baseline_portfolio_positions = pd.read_csv(
        Path(join(baseline_path, "portfolio_positions.csv")), index_col=0, parse_dates=True
    )
    baseline_turnover = calculate_baseline_turnover(baseline_portfolio_positions)
    baseline_portfolio_results = pd.read_csv(
        Path(join(baseline_path, "replicating_results.csv")), index_col=0, parse_dates=True
    )
    baseline_tracking_error, baseline_beta, baseline_mae, baseline_pv = construct_pv(baseline_portfolio_results)

    pv_all = pd.DataFrame(index=baseline_pv.index)
    pv_all["ftse_100"] = (1 + baseline_portfolio_results["index_return"]).cumprod()
    pv_all["baseline"] = baseline_pv
    passive_portfolio_performance = pd.DataFrame(index=["tracking_error", "beta", "mean_absolute_error"])
    passive_portfolio_performance["baseline"] = [baseline_tracking_error, baseline_beta, baseline_mae]
    turnover_all = pd.DataFrame(index=["mean_turnover_ratio", "max_turnover_ratio"])
    turnover_all["baseline"] = [baseline_turnover['turnover_ratio'].mean(), baseline_turnover['turnover_ratio'].max()]

    if analysis_on == "without_turnover":
        # test_b_list = [50000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 10000000]
        # test_b_list = [2500, 5000, 10000, 25000, 50000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]
        test_b_list = [2500, 5000, 10000, 25000, 50000, 250000, 500000, 1000000]
        test_tau_list = [2.0]
    elif analysis_on == "compare_turnover":
        test_b_list = [100000]
        test_tau_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]  # 2.0]
    elif analysis_on == "compare_budget":
        test_b_list = [50000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 10000000]
        test_tau_list = [0.8]
    else:
        raise ValueError
    for b in test_b_list:
        for tau in test_tau_list:
            print(f"B is {b} and tau is {tau}")
            tmp_results_dir = Path(join(results_main_dir, f"param_b_{b}_tau_{tau}"))
            if not tmp_results_dir.exists():
                print(f"Path: {Path} does not exist")
                continue
            # insample_results = pd.read_csv(
            #     Path(join(tmp_results_dir, "insample_info_3780.csv")), index_col=0, parse_dates=True
            # )
            portfolio_positions = pd.read_csv(
                Path(join(tmp_results_dir, "portfolio_positions_3780.csv")), index_col=0, parse_dates=True
            )
            tmp_turnover_info = calculate_turnover(portfolio_positions, close_price, b)
            print(f"Mean turnover ratio is: {tmp_turnover_info['turnover_ratio'].mean()}\n",
                  f"Max turnover ratio is: {tmp_turnover_info['turnover_ratio'].max()}")
            replicating_results = pd.read_csv(
                Path(join(tmp_results_dir, "replicating_results_3780.csv")), index_col=0, parse_dates=True
            )
            tmp_te, tmp_beta, tmp_mae, tmp_pv = construct_pv(replicating_results)
            passive_portfolio_performance[f"port_b_{b}_tau_{tau}"] = [tmp_te, tmp_beta, tmp_mae]
            turnover_all[f"port_b_{b}_tau_{tau}"] = [
                tmp_turnover_info['turnover_ratio'].mean(), tmp_turnover_info['turnover_ratio'].max()
            ]
            pv_all[f"port_b_{b}_tau_{tau}"] = tmp_pv
            print(f"Tracking Error is {tmp_te}")
    passive_portfolio_performance.to_csv(Path(join(final_results, "passive_portfolio_performance.csv")))
    turnover_all.to_csv(Path(join(final_results, "turnover_all.csv")))

    pv_all.columns = [i if not i.startswith("port") else int(i.replace("port_b_", "").replace("_tau_2.0", "")) // 100
                      for i in pv_all.columns]
    pv_all.columns = [str(i // 1000) + "k" if isinstance(i, int) and i > 1000 else str(i) for i in pv_all.columns]
    pv_all.to_csv(Path(join(final_results, "pv_all.csv")))
    _, pmetrics = calculate_pmetrics(pv_all, freq_per_year=252, is_reformat=True)
    pmetrics.to_csv(Path(join(final_results, "pmetrics.csv")))
    return


def calculate_baseline_turnover(portfolio_weights):
    return pd.DataFrame(portfolio_weights.diff().abs().sum(axis=1) / 2, columns=["turnover_ratio"])


def calculate_turnover(portfolio_positions, close_price, b):
    portfolio_positions = portfolio_positions.fillna(0.0)
    portfolio_positions_changes = portfolio_positions - portfolio_positions.shift(1)
    portfolio_positions_changes = portfolio_positions_changes.dropna()
    turnover_df = portfolio_positions_changes * close_price.loc[portfolio_positions_changes.index]
    turnover_df = turnover_df.fillna(0.0)
    turnover_info = pd.DataFrame(turnover_df.apply(lambda x: sum(abs(x)), axis=1), columns=["turnover_amount"])
    turnover_info["turnover_ratio"] = turnover_info["turnover_amount"] / (b * 2)
    return turnover_info


def construct_pv(replicating_returns):
    tracking_error = replicating_returns["return_diff"].std()
    replicating_returns["portfolio_value"] = (1 + replicating_returns["portfolio_return"]).cumprod()
    # replicating_returns["index_value"] = (1 + replicating_returns["index_return"]).cumprod()
    cov_matrix = np.cov(replicating_returns["portfolio_return"], replicating_returns["index_return"])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]  # Cov(portfolio, index) / Var(index)
    mean_absolute_error = replicating_returns["return_diff"].abs().mean()
    return tracking_error, beta, mean_absolute_error, replicating_returns["portfolio_value"]


if __name__ == '__main__':
    main(
        start_date="2005-01-05",
        end_date="2025-01-31",
        analysis_on="without_turnover"  # "without_turnover", "compare_turnover", "compare_budget"
    )
