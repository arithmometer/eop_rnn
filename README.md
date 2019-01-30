# EOP time series prediction using RNN

## Single model

Single model that predicts 2 (`x_pole`, `y_pole`) time series at once. Recurrent model is trained using dropout that
can be specified via `dropout` and `recurrent_dropout` parameters of the script. Number of units in recurrent layers
can be specified via `ncells`.

First train a network (parameters can be specified):

```
python model_dense_GRU2.py --dropout=0.5 --recurrent_dropout=0.5 --ncells=128
```

Set environment variable `CUDA_VISIBLE_DEVICES` to specify GPU device to use or choose another model:

```
env CUDA_VISIBLE_DEVICES=0 python model_dense_LSTM3.py
```

Two models can run on the same GPU if they both fit into GPU memory.

A new model can be created in a distinct file, import `train_model` and `parse_args` utils from `train_model.py` to
unify your runs. New model should specify

Each run creates a separate directory in `checkpoints_single_model` named with a timestamp to store best checkpoints and
corresponding parameters.

Training process of all runs can be tracked using Tensorboard in a separate terminal window: 

```
tensorboard --logdir log
```

## Make predictions

In order to make predictions using some checkpoint, run `make_predictions.py` script:

```
python make_predictions.py --checkpoint checkpoints_single_model/2019-01-30_01-10-33/model-008.h5
```

## Compare forecasts

TODO: compare different RNN models.

For now paths to different RNN predictions will be hardcoded.

Use `compare_forecasts.py` script.

Compare RNN forecasts with:
* Bulletin A forecasts;
* Pulkovo observatory forecasts;
* SSA forecasts.
