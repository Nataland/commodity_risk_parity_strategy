import pandas as pd
import numpy as np
from scipy.optimize import minimize


def calculate_portfolio_var(w, V):
    df_returns = pd.read_csv('datatest.csv', parse_dates=True, index_col=0)
    w = np.matrix(w)
    CoVar = df_returns.cov()
    V = np.matrix(CoVar)
    bkt_var = (w * V * w.T)
    return bkt_var


def calculate_risk_contribution(w, V):
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
    MRC = V * w.T
    RC = np.multiply(MRC, w.T) / sigma  # Risk Contribution
    return RC


def risk_budget_objective(x, pars):
    # calculate portfolio risk
    V = pars[0]  # covariance table
    x_t = pars[1]  # risk target in percent of portfolio risk
    x = np.matrix(x)
    sig_p = np.sqrt(x * V * x.T)  # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    SSE = sum(np.square(asset_RC - risk_target.T))[0, 0] * 1000  # sum of squared error
    return SSE


def total_weight_constraint(x):
    return np.sum(x) - 1.0


def long_only_constraint(x):
    return x


def iter_covar(roll, w0, x_t, cons):
    ## TODO: instead of making appending results to a list, add them to a dataframe directly
    x_results = []
    x_dates = []

    for i in range(len(roll) // 4):
        single_row = roll[0:4]
        mini = minimize(risk_budget_objective, w0, args=[single_row.values, x_t.T], method='SLSQP', constraints=cons,
                        options={'disp': True})
        x_results.append(mini.x)
        x_dates.append(single_row.index[0][0])
        roll = roll.iloc[4:]
        
    return [x_results, x_dates]


def list_to_df(it):
    first = []
    second = []
    third = []
    fourth = []
    date = []

    for i in it[0]:
        first.append(i[0])
        second.append(i[1])
        third.append(i[2])
        fourth.append(i[3])

    for i in it[1]:
        date.append(i)

    df = pd.DataFrame(
        {'Timestamp': date, 'Precious metal': first, 'Energy': second, 'Indus metal': third, 'Agriculture': fourth})
    
    return df


def main():
    df_returns = pd.read_csv('datatest.csv', parse_dates=True, index_col=0)
    
    rolling_size = 12
    wgt_init = [0.25, 0.25, 0.25, 0.25]
    
    w0 = np.matrix(wgt_init)
    CoVar = df_returns.cov()
    V = np.matrix(CoVar)
    
    x_t = np.matrix([[0.25, 0.25, 0.25, 0.25]])  # your risk budget percent of total portfolio risk
    
    cons = ({'type': 'eq', 'fun': total_weight_constraint}, {'type': 'ineq', 'fun': long_only_constraint})
    
    optimize_wgt = minimize(risk_budget_objective, w0, args=[V, x_t.T], method='SLSQP', constraints=cons,
                            options={'disp': True})
    
    print(optimize_wgt.x)
    
    roll_rets = df_returns.rolling(window=rolling_size)
    roll_covar = roll_rets.cov().dropna()
    final_df = list_to_df(iter_covar(roll_covar, w0, x_t, cons))
    
    final_df.to_csv('results.csv', sep=',')
    

main()
