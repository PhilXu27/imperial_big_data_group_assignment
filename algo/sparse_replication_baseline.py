from algo.sparse_replication import load_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from pathlib import Path
from os.path import join
from utils.path_info import *


def sparse_replicating_optimization_baseline(
        r_vector_train, h_matrix_train,
        r_vector_test,
        h_matrix_test,
        lambda1, is_info=True
):
    model = Lasso(alpha=lambda1, fit_intercept=False)
    model.fit(h_matrix_train, r_vector_train)

    r_vector_pred = model.predict(h_matrix_test)

    oos_mse = mean_squared_error(r_vector_test, r_vector_pred)
    oos_mae = mean_absolute_error(r_vector_test, r_vector_pred)

    if is_info:
        portfolio_return = pd.DataFrame(r_vector_pred, index=r_vector_test.index, columns=["portfolio_return"])
        oos_performance_df = pd.concat([r_vector_test, portfolio_return], axis=1)
        oos_performance_df["return_diff"] = oos_performance_df["index_return"] - oos_performance_df["portfolio_return"]
        return model.coef_, oos_performance_df, oos_mse, oos_mae
    else:
        return oos_mse  # do hyperparam tuning.


def construct_sparse_portfolio_baseline(r_vector, h_matrix):
    in_sample_info_all = pd.DataFrame(index=r_vector.index, columns=[
        "best_lambda", "total_turnover", "turnover_ratio"
    ])
    portfolio_positions = pd.DataFrame(index=r_vector.index, columns=h_matrix.columns)
    replicating_results = pd.DataFrame()
    for t in range(window_size + validation_window_size, r_vector.shape[0], holding_period):
        today = r_vector.index[t]
        print(f"Run Lasso, Baseline Model on {today}")
        # Get valid columns
        valid_cols = h_matrix.columns[h_matrix.iloc[t - window_size - validation_window_size].notna()]
        # Select data for hyper-param tuning, and training/testing
        r_vector_train = r_vector.iloc[t - window_size - validation_window_size: t - validation_window_size, :]
        h_matrix_train = h_matrix.iloc[t - window_size - validation_window_size: t - validation_window_size][valid_cols]
        r_vector_validation = r_vector.iloc[t - validation_window_size: t, :]
        h_matrix_validation = h_matrix.iloc[t - validation_window_size: t][valid_cols]
        r_vector_final_train = r_vector.iloc[t - window_size: t, :]
        h_matrix_final_train = h_matrix.iloc[t - window_size: t][valid_cols]
        r_vector_final_test = r_vector.iloc[t: t + holding_period, :]
        h_matrix_final_test = h_matrix.iloc[t: t + holding_period][valid_cols]

        # Hyperparameter tuning, to select the lambda_lasso
        lambda_lasso_searching_list = [0.0000025, 0.000005, 0.0000075, 0.00001, 0.00005, 0.0001]
        test_mse_result = {}
        for test_lambda in lambda_lasso_searching_list:
            mse = sparse_replicating_optimization_baseline(
                r_vector_train, h_matrix_train, r_vector_validation, h_matrix_validation,
                lambda1=test_lambda, is_info=False
            )
            test_mse_result[test_lambda] = mse
        best_lambda_lasso = min(test_mse_result, key=test_mse_result.get)

        w_optimal, portfolio_return, _, _ = sparse_replicating_optimization_baseline(
            r_vector_final_train, h_matrix_final_train, r_vector_final_test, h_matrix_final_test,
            lambda1=best_lambda_lasso, is_info=True
        )
        in_sample_info_all.loc[today, "best_lambda"] = best_lambda_lasso
        portfolio_positions.loc[today] = pd.Series(w_optimal, index=valid_cols)
        replicating_results = pd.concat([replicating_results, portfolio_return], axis=0)

    experiment_save_path = Path(join(
        baseline_results, f"pool_{mode}_{start_date}_{end_date}", f"holding_{holding_period}_training_{window_size}"
    ))
    experiment_save_path.mkdir(parents=True, exist_ok=True)

    in_sample_info_all = in_sample_info_all.dropna(how="all")
    in_sample_info_all.to_csv(Path(join(experiment_save_path, "insample_info.csv")))
    replicating_results.to_csv(Path(join(experiment_save_path, "replicating_results.csv")))
    portfolio_positions = portfolio_positions.dropna(how="all")
    portfolio_positions.to_csv(Path(join(experiment_save_path, "portfolio_positions.csv")))
    return


def main(mode):
    r_vector, h_matrix, p_matrix = load_data(mode, start_date, end_date)
    construct_sparse_portfolio_baseline(r_vector, h_matrix)
    return


if __name__ == '__main__':
    start_date, end_date = "2005-01-05", "2025-01-31"
    window_size = 252 * 5
    validation_window_size = window_size // 5  # 20% Validation Set
    holding_period = 21
    mode = "ftse_250"
    main(mode)
