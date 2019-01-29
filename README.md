# EOP time series prediction using RNN

## Single model

Single model that predicts 2 (`x_pole`, `y_pole`) time series at once. Recurrent GRU model is trained using dropout that
can be specified via `dropout` and `recurrent_dropout` parameters of the script. 

First train a network:

```
python train_single_model.py
```

or

```
python train_single_model.py --dropout=0.2 --recurrent_dropout=0.2
```

Set environment variable `CUDA_VISIBLE_DEVICES` to specify GPU device to use:

```
env CUDA_VISIBLE_DEVICES=0 python train_single_model.py
```

Each run creates a separate directory in `checkpoints_single_model` named with a timestamp to store best checkpoints and
corresponding parameters.

Training process of all runs can be tracked using Tensorboard in a separate terminal window: 

```
tensorboard --logdir log
```

## Fully connected model

TODO

Anosova N. P. "Using neural networks for EOP time series prediction" (in Russian).

## Make predictions

In order to make predictions using some checkpoint, run `make_predictions.py` script:

```
python make_predictions.py --checkpoint checkpoints_single_model/2019-01-30_01-10-33/model-008.h5
```

## Compare forecasts

TODO

Compare RNN forecasts with:
* Bulletin A forecasts;
* Pulkovo observatory forecasts;
* FC network results;
* SSA forecasts.

