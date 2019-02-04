from keras.layers import Input, LSTM, Dense
from keras.models import Model

from train_model import parse_args, train_model


def main():
    args = parse_args()

    nlayers = 4
    cell_type = 'LSTM'

    net_input = Input(shape=(args.lookback, 2))

    x = Dense(args.ncells, activation='relu')(net_input)
    x = LSTM(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)
    x = LSTM(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)

    out_x = Dense(args.delay, activation='linear')(x)
    out_y = Dense(args.delay, activation='linear')(x)

    model = Model(inputs=net_input, outputs=[out_x, out_y])

    train_model(model, args, nlayers=nlayers, cell_type=cell_type)


if __name__ == '__main__':
    main()
