from algo.sparse_replication import load_data, timing_decorator, calculater_replicating_mse
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import scipy.optimize as sco
from pathlib import Path
from os.path import join
from utils.path_info import *


@timing_decorator
def sparse_replicating_optimization(r_vector, h_matrix, p_matrix, b, lambda_lasso):
    h_mul_p_matrix = p_matrix * h_matrix
    r_vector = r_vector
    h_matrix = h_matrix
    p_matrix = p_matrix

    model = Model("Sparse Replicating")
    model.setParam("OutputFlag", 0)
    num_stocks = h_matrix.shape[1]
    window_size = h_matrix.shape[0]
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

    # l1_penalty = lambda_lasso * b * quicksum(z[i] for i in range(num_stocks))
    model.setObjective(error, GRB.MINIMIZE)
    # model.setObjective(error + l1_penalty, GRB.MINIMIZE)
    model.addConstr(quicksum(s[i] * p_matrix.iloc[-1, i] for i in range(num_stocks)) <= b, "Budget_Constraint")
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Optimization failed with status: {model.status}\n "
            f"Check this url for details: "
            f"https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html"
        )

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
    # l1_penalty_value = lambda_lasso * sum(p_matrix.iloc[0, i] * abs(optimized_s[i] / b) for i in range(num_stocks))
    number_non_zero_stocks = np.count_nonzero(optimized_s)
    info = {
        "in_sample_mse": mse,
        "in_sample_mae": mae,
        # "in_sample_l1_penalty": l1_penalty_value,
        "portfolio_value": portfolio_value_of_stocks,
        "non_zero_stocks": number_non_zero_stocks,
    }
    return optimized_s, info


def tangency_portfolio(h_matrix):
    mu = h_matrix.mean()  # Expected returns (mean return per asset)
    cov_matrix = h_matrix.cov()  # Covariance matrix

    # Define the objective function (negative Sharpe ratio, as we minimize)
    def negative_sharpe_ratio(weights, mu, cov_matrix):
        portfolio_return = np.dot(weights, mu)  # w^T * mu
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # sqrt(w^T * Sigma * w)
        return -portfolio_return / portfolio_volatility  # Negative Sharpe ratio

    # Constraints: sum of weights = 1 (fully invested portfolio)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Bounds: Weights can be between 0 and 1 (long-only constraint)
    bounds = [(0, 1) for _ in range(h_matrix.shape[1])]
    # Initial guess: Equal weights
    initial_guess = np.array([1 / h_matrix.shape[1]] * h_matrix.shape[1])
    # Optimize for max Sharpe ratio
    result = sco.minimize(negative_sharpe_ratio, initial_guess, args=(mu, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    # Extract optimal weights
    optimal_weights = result.x
    # Create a DataFrame for better readability
    return optimal_weights


def backward_sparse_replication(r_vector, h_matrix, p_matrix):
    lambda_lasso_searching_list = [0.0001]  # , 0.001]
    b_test_list = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]

    r_vector_train = r_vector.loc[: validating_start_date, :]
    h_matrix_train = h_matrix.loc[: validating_start_date, :]
    p_matrix_train = p_matrix.loc[: validating_start_date, :]
    r_vector_val = r_vector.loc[validating_start_date:, :]
    h_matrix_val = h_matrix.loc[validating_start_date:, :]
    p_matrix_val = p_matrix.loc[validating_start_date:, :]

    portfolio_all = pd.DataFrame()
    for b in b_test_list:
        b *= 100
        hyper_para_tuning_dict = {}
        # for lambda_lasso in lambda_lasso_searching_list:
        #     optimized_s, info = sparse_replicating_optimization(
        #         r_vector_train, h_matrix_train, p_matrix_train, b, lambda_lasso
        #     )
        #     val_mse = np.mean(
        #         (r_vector_val.values.flatten() - (h_matrix_val * p_matrix_val).values @ optimized_s / b) ** 2
        #     )
        #     hyper_para_tuning_dict[lambda_lasso] = val_mse
        # best_lambda_lasso = min(hyper_para_tuning_dict, key=hyper_para_tuning_dict.get)
        best_lambda_lasso = lambda_lasso_searching_list[0]
        optimized_s, info = sparse_replicating_optimization(
            r_vector, h_matrix, p_matrix, b, best_lambda_lasso
        )
        oos_portfolio_return = (h_matrix * p_matrix) @ optimized_s / (p_matrix @ optimized_s)

        oos_portfolio_return = pd.DataFrame(oos_portfolio_return, columns=[f"b_{b}"])
        portfolio_all = pd.concat([portfolio_all, oos_portfolio_return], axis=1)
    return portfolio_all


def main_predicting():
    r_vector, h_matrix, p_matrix = load_data("ftse_250", start_date, end_date)
    h_matrix = h_matrix.ffill()
    h_matrix = h_matrix.fillna(0.0)
    experiment_save_path = Path(join(
        paper_results, f"START_{start_date}_END_{end_date}",
        f"VAL_{validating_start_date}", f"ftse_250"
    ))
    experiment_save_path.mkdir(parents=True, exist_ok=True)
    max_sharpe_port_weights = tangency_portfolio(h_matrix)
    max_sharpe_port_return = pd.DataFrame(
        h_matrix @ max_sharpe_port_weights, columns=["max_sharpe_port"]
    )
    max_sharpe_port_return.to_csv(Path(join(experiment_save_path, "max_sharpe_ratio_portfolio.csv")))
    for test_mode in [
        "ftse_250", "ftse_250_top_5_market_cap", "ftse_250_top_10_market_cap", "ftse_250_top_20_market_cap",
        "ftse_250_top_50_market_cap", "ftse_100"
    ]:
        r_vector, h_matrix, p_matrix = load_data(test_mode, start_date, end_date)
        h_matrix = h_matrix.ffill()
        h_matrix = h_matrix.fillna(0.0)
        p_matrix = p_matrix.ffill()
        p_matrix = p_matrix.fillna(0.0)

        experiment_save_path = Path(join(
            paper_results, f"START_{start_date}_END_{end_date}",
            f"VAL_{validating_start_date}", f"{test_mode}"
        ))
        experiment_save_path.mkdir(parents=True, exist_ok=True)

        portfolio_all = backward_sparse_replication(max_sharpe_port_return, h_matrix, p_matrix)
        portfolio_all.to_csv(Path(join(experiment_save_path, "my_portfolio.csv")))
    return


if __name__ == '__main__':
    start_date, end_date = "2020-01-31", "2025-01-31"
    validating_start_date = "2024-01-31"
    main_predicting()
