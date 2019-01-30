import argparse
import json
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from keras.models import load_model
from ts_helper import mjd_to_row, single_model_generator


def main():
    parser = argparse.ArgumentParser(description='EOP prediction using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to checkpoint file for predictions')
    args = parser.parse_args()

    # TODO: use the best (latest) checkpoint in directory if file not specified
    checkpoint_dir = os.path.dirname(args.checkpoint)
    timestr = os.path.split(checkpoint_dir)[-1]
    filename = os.path.basename(args.checkpoint).split('.')[0]

    predictions_dir = os.path.join("predictions_single_model", timestr, filename)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    model = load_model(args.checkpoint)

    norm_data = np.loadtxt(os.path.join(checkpoint_dir, "norm_single_model.csv"), delimiter=",")

    c04 = pd.read_csv(os.path.join("data", "eopc04_14_IAU2000.62-now.csv"), delimiter=";")

    data = c04[["x_pole", "y_pole"]].values.copy()

    # normalize all data
    mean_data, std_data = norm_data
    data -= mean_data
    data /= std_data

    with open(os.path.join(checkpoint_dir, "params.json"), 'r') as file:
        params = json.load(file)
    lookback = params['lookback']
    delay = params['delay']
    step = params['step']

    for start_forecast_mjd in tqdm(range(57024, 57800 + 1, args.batch_size)):
        start_row = mjd_to_row(start_forecast_mjd)

        test_gen = single_model_generator(data,
                                          lookback=lookback,
                                          delay=delay,
                                          min_index=start_row - lookback,
                                          max_index=None,
                                          shuffle=False,
                                          step=step,
                                          batch_size=args.batch_size)

        predictions = np.swapaxes(np.array(model.predict_generator(test_gen, steps=1)), 0, 1)

        predictions = predictions * np.expand_dims(std_data, axis=1)
        predictions = predictions + np.expand_dims(mean_data, axis=1)

        for i in range(args.batch_size):
            df = pd.DataFrame({'x_pole': predictions[0, 0],
                               'y_pole': predictions[0, 1]})
            df.to_csv(os.path.join(predictions_dir, str(start_forecast_mjd + i) + '_rnn_365.csv'))


if __name__ == '__main__':
    main()
