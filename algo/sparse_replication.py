from utils.path_info import demo_data_path
import pandas as pd
from pathlib import Path
from os.path import join
from gurobipy import Model, GRB, quicksum
import numpy as np


def load_data(start, end):
    # symbol_list = pd.read_csv(Path(join(demo_data_path, "main_symbol_list.csv")))["symbol_id"].to_list()[:80]
    adjusted_close = pd.read_csv(Path(join(demo_data_path, "adjusted_nav.csv")), index_col=0, parse_dates=True)
    adjusted_close = adjusted_close.iloc[:, :80]
    symbol_list = adjusted_close.columns.to_list()
    adjusted_close = adjusted_close.bfill()
    adjusted_return = adjusted_close.pct_change().dropna(how="all")
    raw_close = adjusted_close.copy(deep=True)
    adjusted_return = adjusted_return.loc[start:end]
    raw_close = raw_close.loc[start:end]

    price_data = raw_close
    hedging_instruments_return = adjusted_return
    holding_instruments_return = hedging_instruments_return[symbol_list]
    return holding_instruments_return, hedging_instruments_return, price_data


def generate_holding(holding_instrument_return):
    inventory = pd.DataFrame(columns=holding_instrument_return.columns, index=holding_instrument_return.index)
    inventory.iloc[window_size] = 1
    for i in range(window_size + 1, inventory.shape[0]):
        sell_fractions = np.abs(np.random.normal(loc=1.0, scale=0.5, size=holding_instrument_return.shape[1]))
        inventory.iloc[i] = inventory.iloc[i - 1] * (1 - sell_fractions)
        inventory.iloc[i][inventory.iloc[i] < 0] += 1

    # todo, need to carefully check if the date is correct. I think this is correct, need to shift 1 day.
    inventory_return = (inventory.shift(1) * holding_instrument_return).sum(axis=1)
    return inventory, inventory_return


def sparse_replication_divisible():

    return


def lasso_helper(
        train_holding, train_inventory_return, train_adjusted_return,
        test_holding, test_inventory_return, test_adjusted_return
):
    num_of_hedging_instruments = train_adjusted_return.shape[0]

    w_prev = train_holding.iloc[-1]
    model = Model("sparse_replication_indivisible")
    w = model.addVars(num_of_hedging_instruments, vtype=GRB.INTEGER, name="w")
    error = quicksum(
        (train_inventory_return[t] - quicksum(train_adjusted_return[t, i] * w[i] for i in range(num_of_hedging_instruments))) ** 2 for
        t in range(window_size))
    model.setObjective(error, GRB.MINIMIZE)
    lambda1 = 0.1
    sparsity = quicksum((w[i] != 0) for i in range(num_of_hedging_instruments))
    model.addConstr(sparsity <= num_of_hedging_instruments * 0.5, "SparsityConstraint") # todo, param here.
    lambda2 = 0.05
    turnover = quicksum((w[i] - w_prev[i]) ** 2 for i in range(num_of_hedging_instruments))
    model.addConstr(turnover <= lambda2 * num_of_hedging_instruments, "TurnoverConstraint")
    model.optimize()
    if model.status == GRB.OPTIMAL:
        hedge_weights = {f"Instrument {i}": w[i].x for i in range(N)}
        return hedge_weights
    else:
        print("No optimal solution found.")
        return None


def sparse_replication_indivisible(inventory, inventory_return: pd.Series, holding_instruments_return, hedging_instruments_return):
    # todo YOU NEED TAKE PRICE INTO CONSIDERATION!
    hedge_positions = []
    hedging_errors = []
    for t in range(window_size + 1, inventory_return.shape[0]):
        inventory_today = inventory.iloc[t, :]
        past_inventory_returns = holding_instruments_return.iloc[t - window_size: t, :]
        R_synthetic = past_inventory_returns.mul(inventory_today, axis=1).sum(axis=1)  # inventory_return_synthetic
        # Weighted sum of returns
        H_train = hedging_instruments_return.iloc[t - window_size:t, :]  # hedging_instruments_return_train_x
        model = Model("Sparse_Hedging")
        model.setParam("OutputFlag", 0)  # Suppress Gurobi output

        N = hedging_instruments_return.shape[1]  # Number of hedging instruments
        w = model.addVars(N, vtype=GRB.INTEGER, name="w")  # Integer constraints for hedge positions
        z = model.addVars(N, vtype=GRB.BINARY, name="z")
        error = quicksum((R_synthetic.iloc[tau] - quicksum(H_train.iloc[tau, i] * w[i] for i in range(N)))**2
                         for tau in range(window_size))

        model.setObjective(error, GRB.MINIMIZE)

        # Sparsity Constraint (Limit number of nonzero hedge instruments)
        lambda1 = 0.5
        model.addConstr(quicksum(z[i] for i in range(N)) <= 2, "SparsityConstraint")  # Allow max 2 hedge instruments

        # Stability Constraint (Reduce turnover)
        if t > window_size + 1:
            w_prev = inventory.iloc[t - 1, :]  # Previous hedge weights
            lambda2 = 1
            turnover = quicksum((w[i] - w_prev[i])**2 for i in range(N))
            model.addConstr(turnover <= lambda2 * N, "TurnoverConstraint")

        # Solve the optimization
        model.optimize()

        hedge_wt = np.array([int(w[i].x) for i in range(N)])  # Extract hedge weights
        hedge_positions.append(hedge_wt)
        # Today's inventory return (realized)
        R_today = holding_instruments_return.iloc[t, :].mul(inventory_today).sum()  # Inventory return today
        H_today = hedging_instruments_return.iloc[t, :]  # Hedging instrument returns today
        hedged_return = R_today - np.dot(H_today.values, hedge_wt)  # Hedged PnL

        # Tomorrow's realized inventory return (for evaluation)
        inventory_tmr = inventory.iloc[t+1, :]  # Inventory holdings tomorrow
        R_tmr = holding_instruments_return.iloc[t+1, :].mul(inventory_tmr).sum()  # Realized inventory return tomorrow
        actual_loss = abs(hedged_return - R_tmr)  # Hedge performance

        hedging_errors.append(actual_loss)

    return


def main(start, end):
    holding_instruments_return, hedging_instruments_return, price_data = load_data(start, end)
    inventory, inventory_return = generate_holding(holding_instruments_return)
    sparse_replication_indivisible(inventory, inventory_return, holding_instruments_return, hedging_instruments_return)
    return


if __name__ == '__main__':
    window_size = 120
    test_start = "2022-01-01"
    test_end = "2023-10-31"
    main(test_start, test_end)
#     symbol_list, adjusted_return, raw_close = load_data(start="2022-01-02", end="2023-10-31")
#     holdings, holding_return = generate_holding(adjusted_return)

