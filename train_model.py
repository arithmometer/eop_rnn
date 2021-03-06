import argparse
import datetime
import json
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as k
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from helpers import mjd_to_row, single_model_generator


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='EOP prediction using RNN')
    parser.add_argument('--memory', type=float, default=0.5, metavar='D',
                        help='GPU memory limits (default: 0.5)')
    parser.add_argument('--ncells', type=int, default=256, metavar='NC',
                        help='number of recurrent units in a layer (default: 256)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--steps', type=int, default=100, metavar='S',
                        help='number of steps per epoch (default: 100)')
    parser.add_argument('--lookback', type=int, default=860, metavar='L',
                        help='lookback window (default: 860)')
    parser.add_argument('--delay', type=int, default=365, metavar='D',
                        help='timesteps to predict (default: 365)')
    parser.add_argument('--timestep', type=int, default=1, metavar='TS',
                        help='window time step (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='D',
                        help='dropout rate (default: 0.0)')
    parser.add_argument('--recurrent_dropout', type=float, default=0.0, metavar='RD',
                        help='recurrent dropout rate (default: 0.0)')
    args = parser.parse_args()
    return args


def train_model(model, args, nlayers, cell_type):
    c04 = pd.read_csv(os.path.join("data", "eopc04_14_IAU2000.62-now.csv"), delimiter=";")

    strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join("checkpoints",
                                  cell_type,
                                  str(nlayers) + "layers",
                                  str(args.ncells) + "cells",
                                  strtime)
    log_dir = os.path.join("log", cell_type + "_" + str(nlayers) + "layers_" + str(args.ncells) + "cells_" + str(args.lookback) + "back")
    for directory in [checkpoint_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # predict x_pole, y_pole for now
    x = c04[["x_pole", "y_pole"]]

    data = x.values

    n_train = mjd_to_row(56292)  # 31.12.2012 (end of training period)
    val_years = 2
    # n_val = mjd_to_row(57022) - mjd_to_row(56292)  # 01.01.2013 - 31.12.2014 (validation period)
    n_val = 365 * val_years

    mean_data = data[:n_train].mean(axis=0)  # calculate on train data
    data -= mean_data  # transform all data
    std_data = data[:n_train].std(axis=0)
    data /= std_data

    np.savetxt(os.path.join(checkpoint_dir, "norm_single_model.csv"), np.array([mean_data, std_data]), delimiter=',')

    params = {'lookback': args.lookback,
              'delay': args.delay,
              'step': args.timestep}

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.memory
    k.tensorflow_backend.set_session(tf.Session(config=config))

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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.8,
                                  patience=10,
                                  verbose=1,
                                  min_lr=0.0002)

    # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
    #                    restore_best_weights=False)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=args.steps,
                                  epochs=args.epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=[checkpoint, tb, reduce_lr])

    with open(os.path.join(checkpoint_dir, "history"), 'wb') as file:
        pickle.dump(history.history, file)

    print("Done.")
