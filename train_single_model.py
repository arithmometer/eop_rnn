import argparse
import datetime
import json
import os
import pickle
import numpy as np
import pandas as pd

from keras.layers import Input, GRU, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from ts_helper import mjd_to_row, single_model_generator


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='EOP prediction using RNN')
    parser.add_argument('--cell', type=str, default='GRU',
                        help='recurrent unit type: GRU or LSTM (default: GRU)')
    parser.add_argument('--nlayers', type=int, default=1, metavar='NL',
                        help='number of recurrent layers (default: 1)')
    parser.add_argument('--ncells', type=int, default=64, metavar='NC',
                        help='number of recurrent units in a layer (default: 64)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--steps', type=int, default=100, metavar='S',
                        help='number of steps per epoch (default: 100)')
    parser.add_argument('--lookback', type=int, default=430, metavar='L',
                        help='lookback window (default: 430)')
    parser.add_argument('--delay', type=int, default=365, metavar='D',
                        help='timesteps to predict (default: 365)')
    parser.add_argument('--timestep', type=int, default=1, metavar='TS',
                        help='window time step (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='D',
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--recurrent_dropout', type=float, default=0.2, metavar='RD',
                        help='recurrent dropout rate (default: 0.2)')
    args = parser.parse_args()

    c04 = pd.read_csv(os.path.join("data", "eopc04_14_IAU2000.62-now.csv"), delimiter=";")

    strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join("checkpoints_single_model",
                                  "GRU" if args.cell == "GRU" else "LSTM",
                                  str(args.nlayers) + "layer",
                                  str(args.ncells) + "cells",
                                  strtime)
    log_dir = os.path.join("log", strtime)
    for directory in [checkpoint_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # predict x_pole, y_pole for now
    x = c04[["x_pole", "y_pole"]]
    # x = c04[["x_pole", "y_pole", "LOD"]]

    data = x.values

    n_train = mjd_to_row(56292)  # 31.12.2012 (end of training period)
    n_val = mjd_to_row(57022) - mjd_to_row(56292)  # 31.12.2012 - 31.12.2014 (validation period)

    mean_data = data[:n_train].mean(axis=0)  # calculate on train data
    data -= mean_data  # transform all data
    std_data = data[:n_train].std(axis=0)
    data /= std_data

    np.savetxt(os.path.join(checkpoint_dir, "norm_single_model.csv"), np.array([mean_data, std_data]), delimiter=',')

    params = {'lookback' : args.lookback,
              'delay' : args.delay,
              'step' : args.timestep}

    with open(os.path.join(checkpoint_dir, "params.json"), 'w') as outfile:
        json.dump(params, outfile)

    batch_size = args.batch_size

    train_gen = single_model_generator(data,
                                       lookback=args.lookback,
                                       delay=args.delay,
                                       min_index=0,
                                       max_index=n_train,
                                       shuffle=True,
                                       step=args.timestep,
                                       batch_size=batch_size)
    val_gen = single_model_generator(data,
                                     lookback=args.lookback,
                                     delay=args.delay,
                                     min_index=n_train + 1 - args.lookback,
                                     max_index=n_train + n_val,
                                     step=args.timestep,
                                     batch_size=batch_size)

    # This is how many steps to draw from val_gen in order to see the whole validation set:
    val_steps = (n_val - 1) // batch_size

    # This is how many steps to draw from test_gen in order to see the whole test set:
    # test_steps = (n_test - 1) // batch_size

    net_input = Input(shape=(args.lookback, 2))

    x = GRU(args.ncells, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(net_input)
    out_x = Dense(args.delay, activation='linear')(x)
    out_y = Dense(args.delay, activation='linear')(x)
    # out_lod = Dense(delay, activation='linear')(x)

    # model = Model(inputs=net_input, outputs=[out_x, out_y, out_lod])
    model = Model(inputs=net_input, outputs=[out_x, out_y])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(model.summary())

    checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'model-{epoch:03d}.h5'),
                                 verbose=1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto')

    tb = TensorBoard(log_dir=log_dir,
                     histogram_freq=0,
                     write_images=True)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=args.steps,
                                  epochs=args.epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=[checkpoint, tb])

    with open(os.path.join(checkpoint_dir, "history"), 'wb') as file:
        pickle.dump(history.history, file)

    print("Done.")


if __name__ == '__main__':
    main()
