import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from helpers import mjd_to_row


def get_real_data(eop, c04, start_mjd, days=365):
    start_row = mjd_to_row(start_mjd)
    return c04[eop][start_row:start_row + days]


def get_ba_prediction(eop, start_mjd, days=365):
    if start_mjd <= 57388:  # 1 Janury 2016
        beginning_of_year = 57024  # 2 January 2015
        ba_file = os.path.join('data', 'ba', 'bulletina-xxviii-')
    else:
        beginning_of_year = 57395  # 8 January 2016
        ba_file = os.path.join('data', 'ba', 'bulletina-xxix-')

    week = (start_mjd - beginning_of_year) // 7 + 1
    ba_file += '{week:03d}.txt'.format(week=week)

    ba = pd.read_csv(ba_file, delimiter=';')

    start_row = np.where(ba.Type == 'prediction')[0][0]

    if start_mjd < 57528:
        column = ['X', 'Y']
    else:
        column = eop

    df = ba[column][start_row:start_row + days]
    df.columns = eop

    return df


def get_rnn_prediction(eop, path, start_forecast_mjd, days=365):
    df = pd.read_csv(os.path.join(path, str(start_forecast_mjd) + '_rnn_365.csv'))
    return df[eop][:days]


def get_ssa_prediction(eop, start_forecast_mjd, days=365):
    df = pd.read_csv(os.path.join("data", "rssa", str(start_forecast_mjd) + "_ssa_spbu_365.txt"), sep='\t', header=None,
                     skiprows=1, names=["MJD", "x_pole", "y_pole", "LOD", "dX", "dY"])
    return df[eop][:days]


def get_pulkovo_prediction(eop, kind, start_forecast_mjd, days=365):
    df = pd.read_csv(os.path.join("data", "pul", str(start_forecast_mjd) + "_" + kind + "_pul.txt"), sep='\t',
                    header=None, skiprows=1, names=["MJD", "x_pole", "y_pole", "LOD", "dX", "dY"])
    return df[eop][:days]


def get_stat(rnn_paths, days=365):
    c04 = pd.read_csv(os.path.join("data", "eopc04_14_IAU2000.62-now.csv"), delimiter=";")
    eops = ["x_pole", "y_pole"]

    rnn_models = list(rnn_paths)
    models = ["ba", "ssa"] + rnn_models

    mses = dict()
    for model in models:
        mses[model] = {"x_pole": [], "y_pole": []}

    predictions_df = {}

    for start_forecast_mjd in tqdm(range(57024, 57724 + 1, 7)):
        real_data_df = get_real_data(eops, c04, start_forecast_mjd, days)

        predictions_df["ba"] = get_ba_prediction(eops, start_forecast_mjd, days)
        predictions_df["ssa"] = get_ssa_prediction(eops, start_forecast_mjd, days)

        for model in rnn_models:
            predictions_df[model] = get_rnn_prediction(eops, rnn_paths[model], start_forecast_mjd, days)

        # calculate MSE of forecasts for all MJDs and store to pandas table
        for model in models:
            for eop in eops:
                mses[model][eop].append(np.mean((real_data_df[eop].values - predictions_df[model][eop].values) ** 2))

    mean_error = dict()
    mses_sorted = dict()

    for model in models:
        mean_error[model] = dict()
        mses_sorted[model] = dict()

    summary = []
    index = ["Bulletin A", "SSA"] + rnn_models

    for eop in eops:
        for model in models:
            mean_error[model][eop] = np.mean(mses[model][eop])
            mses_sorted[model][eop] = np.sort(mses[model][eop])

        summary.append(pd.DataFrame({
            "Average MSE": [mean_error[model][eop] for model in models],
            "95% interval": [(mses_sorted[model][eop][5 - 1], mses_sorted[model][eop][95 - 1]) for model in models]
            }, index=index))

    return summary


def main():
    rnn_paths = {
        "GRU3": "predictions_single_model/GRU/3layer/64cells/2019-01-30_18-25-17/model-003",
        "GRU4": "predictions_single_model/GRU/4layer/64cells/2019-01-30_18-58-26/model-002"
    }

    days = 90

    df_x, df_y = get_stat(rnn_paths, days)

    print("Compare weekly forecasts from MJD=57024 (2 January 2015) to MJD=57724 (2 December 2016)")

    print("x_pole:")
    print(df_x)
    print()
    print("y_pole:")
    print(df_y)


if __name__ == '__main__':
    main()
