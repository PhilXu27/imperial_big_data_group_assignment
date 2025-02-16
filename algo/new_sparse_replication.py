from utils.path_info import *
import pandas as pd
from pathlib import Path
from os.path import join
from gurobipy import Model, GRB, quicksum
import numpy as np


def load_data():
    """

    Returns:
        raw_price, adjusted_return, index_return, all dfs, with date as index.

    """
    test_list = [
        'AV/ LN Equity','AVON LN Equity','AZN LN Equity', 'BA/ LN Equity', 'BAB LN Equity', 'BAG LN Equity',
        'BARC LN Equity', 'BASC LN Equity', 'BATS LN Equity', 'BBY LN Equity', 'BEZ LN Equity', 'BGCG LN Equity',
        'BGEU LN Equity', 'BGFD LN Equity', 'BGS LN Equity', 'BGUK LN Equity', 'BIOG LN Equity', 'BIPS LN Equity',
        'BKG LN Equity', 'BLND LN Equity', 'BMY LN Equity', 'BNKR LN Equity',
        "BERI LN Equity", "ASHM LN Equity", "IGG LN Equity"
    ]
    candidate_list = pd.read_csv(Path(join(main_data_path, "list_of_ftse_all_shares.csv"))).columns.to_list()

    effective_trading_dates = pd.read_csv(
        Path(join(main_data_path, "effective_trading_calendar.csv")), index_col=0, parse_dates=True, dayfirst=True
    ).loc[start: end].index.tolist()

    raw_price = pd.read_csv(
        Path(join(main_data_path, "raw_price.csv")), index_col=0, parse_dates=True
    )
    adjusted_return = pd.read_csv(Path(join(main_data_path, "adjusted_return.csv")), index_col=0, parse_dates=True)
    assert set(candidate_list) == set(raw_price.columns)
    assert set(candidate_list) == set(adjusted_return.columns)

    index_data = pd.read_csv(
        Path(join(main_data_path, "ftse_100_index.csv")), index_col=0, parse_dates=True, dayfirst=True
    )
    index_return = index_data.pct_change().dropna()
    index_return.columns = ["index_return"]

    raw_price = raw_price.loc[effective_trading_dates]
    adjusted_return = adjusted_return.loc[effective_trading_dates]
    index_return = index_return.loc[effective_trading_dates]

    return index_return, adjusted_return[test_list], raw_price[test_list]


def sparse_replicating_optimization(
        r_vector, h_matrix, p_matrix,
        s_prev_t, b, tau=0.5, is_turnover=True,
        lambda1=0.01
):
    h_mul_p_matrix = p_matrix * h_matrix
    r_vector = r_vector
    h_matrix = h_matrix
    p_matrix = p_matrix

    model = Model("Sparse Replicating")
    model.setParam("OutputFlag", 0)
    num_stocks = h_matrix.shape[1]
    # Decision variables: Number of shares per stock (integer)
    s = model.addVars(num_stocks, vtype=GRB.INTEGER, name="s")
    z = model.addVars(num_stocks, vtype=GRB.CONTINUOUS, name="z")
    turnover_abs = model.addVars(num_stocks, vtype=GRB.CONTINUOUS, name="turnover_abs")

    error = quicksum(
        (r_vector.iloc[t, 0] * b - quicksum(h_mul_p_matrix.iloc[t, i] * s[i] for i in range(num_stocks))) ** 2
        for t in range(window_size)
    )
    for i in range(num_stocks):
        model.addConstr(z[i] >= s[i] * p_matrix.iloc[-1, i])
        model.addConstr(z[i] >= -(s[i] * p_matrix.iloc[-1, i]))

    l1_penalty = lambda1 * b * quicksum(z[i] for i in range(num_stocks))
    model.setObjective(error + l1_penalty, GRB.MINIMIZE)

    # Turnover Constraint: Limit changes in holdings if not the first period
    if is_turnover and not s_prev_t.isna().all():
        for i in range(num_stocks):
            model.addConstr(turnover_abs[i] >= (s[i] - s_prev_t[i]) * p_matrix.iloc[-1, i])
            model.addConstr(turnover_abs[i] >= -(s[i] - s_prev_t[i]) * p_matrix.iloc[-1, i])

        # Total turnover constraint
        turnover = quicksum(turnover_abs[i] for i in range(num_stocks))
        model.addConstr(turnover <= tau * b, "Turnover_Constraint")

    # Solve the optimization problem
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Optimization failed with status: {model.status}\n "
            f"Check this url for details: "
            f"https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html"
        )

    # Extract the optimal stock holdings
    optimized_s = np.array([s[i].X for i in range(num_stocks)])
    portfolio_value_of_stocks = sum([optimized_s[i] * p_matrix.iloc[-1, i] for i in range(num_stocks)])
    mse = sum(
        (r_vector.iloc[t, 0] - sum(h_mul_p_matrix.iloc[t, i] * optimized_s[i] / b for i in range(num_stocks))) ** 2
        for t in range(window_size)
    ) / window_size

    mae = sum(
        abs(r_vector.iloc[t, 0] - sum(h_mul_p_matrix.iloc[t, i] * optimized_s[i] / b for i in range(num_stocks)))
        for t in range(window_size)
    ) / window_size
    l1_penalty_value = lambda1 * sum(p_matrix.iloc[0, i] * abs(optimized_s[i] / b) for i in range(num_stocks))
    number_non_zero_stocks = np.count_nonzero(optimized_s)

    info = {
        "in_sample_mse": mse,
        "in_sample_mae": mae,
        "in_sample_l1_penalty": l1_penalty_value,
        "portfolio_value": portfolio_value_of_stocks,
        "non_zero_stocks": number_non_zero_stocks
    }
    return optimized_s, info


def replicating_performance_analysis(
        r_vector, h_matrix, p_matrix, stocks, b
):
    performance_df = pd.DataFrame(index=h_matrix.index, columns=["index_return", "portfolio_return", "return_diff"])
    performance_df["index_return"] = r_vector["index_return"]
    performance_df["portfolio_return"] = pd.Series((h_matrix * p_matrix).values @ stocks / b, index=h_matrix.index)
    performance_df["return_diff"] = performance_df["index_return"] - performance_df["portfolio_return"]
    error_info = {
        "mse": (performance_df["return_diff"] ** 2).mean(),
        "mae": abs(performance_df["return_diff"]).mean(),
    }
    return performance_df, error_info


def simple_lasso():
    from sklearn.linear_model import Lasso
    # rw_cv
    #
    return


def calculater_replicating_mse(r_vector, h_matrix, p_matrix, stocks, b):
    return np.mean((r_vector.values.flatten() - (h_matrix * p_matrix).values @ stocks / b) ** 2)


def construct_sparse_portfolio(r_vector, h_matrix, p_matrix, **kwargs):
    common_cols = h_matrix.columns.intersection(p_matrix.columns)
    b = kwargs.get("b")
    tau = kwargs.get("tau")
    is_turnover = kwargs.get("is_turnover")

    in_sample_info_all = pd.DataFrame()
    portfolio_positions = pd.DataFrame(index=r_vector.index, columns=common_cols)

    replicating_results = pd.DataFrame(r_vector.index, columns=["replicating_error"])
    for t in range(window_size + validation_window_size, r_vector.shape[0], holding_period):
        s_prev_t = portfolio_positions.iloc[t - holding_period]

        valid_cols = common_cols[
            h_matrix.iloc[t - window_size - validation_window_size][common_cols].notna() &
            p_matrix.iloc[t - window_size - validation_window_size][common_cols].notna()
            ]

        r_vector_train = r_vector.iloc[t - window_size - validation_window_size: t - validation_window_size, :]
        h_matrix_train = h_matrix.iloc[t - window_size - validation_window_size: t - validation_window_size][valid_cols]
        p_matrix_train = p_matrix.iloc[t - window_size - validation_window_size: t - validation_window_size][valid_cols]
        r_vector_validation = r_vector.iloc[t - validation_window_size: t, :]
        h_matrix_validation = h_matrix.iloc[t - validation_window_size: t][valid_cols]
        p_matrix_validation = p_matrix.iloc[t - validation_window_size: t][valid_cols]
        r_vector_final_train = r_vector.iloc[t - window_size: t, :]
        h_matrix_final_train = h_matrix.iloc[t - window_size: t][valid_cols]
        p_matrix_final_train = p_matrix.iloc[t - window_size: t][valid_cols]
        r_vector_final_test = r_vector.iloc[t: t + holding_period, :]
        h_matrix_final_test = h_matrix.iloc[t: t + holding_period][valid_cols]
        p_matrix_final_test = p_matrix.iloc[t: t + holding_period][valid_cols]

        # Hyperparameter tuning, to select the lambda1
        lambda_lasso_searching_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        # for test_lambda in lambda_lasso_searching_list:
        #  do training on r_vector_train, h, p; sparse_replicating_optimization
        #  do validating/prediction on r_vector_validation, h, p; calculater_replicating_mse

        # After selecting the best lambda_lasso
        lambda_lasso = 0.0001
        s_optimal, in_sample_info = sparse_replicating_optimization(
            r_vector_final_train, h_matrix_final_train, p_matrix_final_train, s_prev_t=s_prev_t, b=b, tau=tau,
            lambda1=lambda_lasso,
        )
        oos_performance_df, oos_error_info = replicating_performance_analysis(
            r_vector_final_test, h_matrix_final_test, p_matrix_final_test, s_optimal, b=b
        )
        portfolio_positions.loc[r_vector.index[t]] = pd.Series(s_optimal, index=valid_cols)

    # save file local path

    # seaborn,
    # Simple Lasso vs Our version, Port Value change over time.
    return


def main(**kwargs):
    r_vector, h_matrix, p_matrix = load_data()
    construct_sparse_portfolio(r_vector, h_matrix, p_matrix, **kwargs)
    return


if __name__ == '__main__':
    start, end = "2005-01-05", "2025-01-31"
    window_size = 252
    validation_window_size = 10
    holding_period = 5
    constraint_param = {
        "b": 10000,
        "tau": 0.3,
        "is_turnover": True
    }
    main(**constraint_param)

