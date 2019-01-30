import argparse
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from ts_helper import mjd_to_row


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


def get_rnn_prediction(eop, path, start_forecast_mjd):
    df = pd.read_csv(os.path.join(path, str(start_forecast_mjd) + '_rnn_365.csv'))
    return df[eop]


def get_ssa_prediction(eop, start_forecast_mjd):
    df = pd.read_csv(os.path.join("data", "rssa", str(start_forecast_mjd) + "_ssa_spbu_365.txt"), sep='\t', header=None,
                     skiprows=1, names=["MJD", "x_pole", "y_pole", "LOD", "dX", "dY"])
    return df[eop]


def get_stat(path):
    c04 = pd.read_csv(os.path.join("data", "eopc04_14_IAU2000.62-now.csv"), delimiter=";")
    eops = ["x_pole", "y_pole"]

    rnn_mses = {"x_pole": [], "y_pole": []}
    ba_mses = {"x_pole": [], "y_pole": []}
    ssa_mses = {"x_pole": [], "y_pole": []}

    for start_forecast_mjd in tqdm(range(57024, 57724 + 1, 7)):
        real_data_df = get_real_data(eops, c04, start_forecast_mjd)
        rnn_predictions_df = get_rnn_prediction(eops, path, start_forecast_mjd)
        ba_predictions_df = get_ba_prediction(eops, start_forecast_mjd)
        ssa_predictions_df = get_ssa_prediction(eops, start_forecast_mjd)

        # calculate MSE of forecasts for all MJDs and store to pandas table
        for eop in eops:
            rnn_mses[eop].append(np.mean((real_data_df[eop].values - rnn_predictions_df[eop].values) ** 2))
            ba_mses[eop].append(np.mean((real_data_df[eop].values - ba_predictions_df[eop].values) ** 2))
            ssa_mses[eop].append(np.mean((real_data_df[eop].values - ssa_predictions_df[eop].values) ** 2))

    rnn_mean_error = dict()
    ba_mean_error = dict()
    ssa_mean_error = dict()

    rnn_mses_sorted = dict()
    ba_mses_sorted = dict()
    ssa_mses_sorted = dict()

    summary = []
    index = ["RNN", "Bulletin A", "SSA"]

    for eop in eops:
        rnn_mean_error[eop] = np.mean(rnn_mses[eop])
        ba_mean_error[eop] = np.mean(ba_mses[eop])
        ssa_mean_error[eop] = np.mean(ssa_mses[eop])

        rnn_mses_sorted[eop] = np.sort(rnn_mses[eop])
        ba_mses_sorted[eop] = np.sort(ba_mses[eop])
        ssa_mses_sorted[eop] = np.sort(ssa_mses[eop])

        summary.append(pd.DataFrame({"Average MSE": [rnn_mean_error[eop], ba_mean_error[eop]],
                                     "95% interval": [(rnn_mses_sorted[eop][5 - 1], rnn_mses_sorted[eop][95 - 1]),
                                                      (ba_mses_sorted[eop][5 - 1], ba_mses_sorted[eop][95 - 1])]},
                                     index=index))

    return summary


def main():
    parser = argparse.ArgumentParser(description='Compare EOP forecasts')
    parser.add_argument('--path-rnn', type=str, default=None,
                        help='path to checkpoint file for predictions')
    args = parser.parse_args()

    df_x, df_y = get_stat(args.path_rnn)

    print("Compare weekly forecasts from MJD=57024 (2 January 2015) to MJD=57724 (2 December 2016)")

    print("x_pole:")
    print(df_x)
    print()
    print("y_pole:")
    print(df_y)


if __name__ == '__main__':
    main()
