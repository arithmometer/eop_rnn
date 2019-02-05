from keras.layers import Input, GRU, Dense, Flatten, add
from keras.models import Model

from train_model import parse_args, train_model


def main():
    args = parse_args()

    nlayers = 4
    cell_type = 'GRU'

    net_input = Input(shape=(args.lookback, 2))

    x_0 = Dense(600, activation='relu')(net_input)
    x_1 = GRU(args.delay, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x_0)
    x_2 = Flatten()(GRU(args.delay, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x_1))

    out_x = Dense(args.delay, activation='linear')(add([Flatten()(x_1), x_2]))
    out_y = Dense(args.delay, activation='linear')(add([Flatten()(x_1), x_2]))

    model = Model(inputs=net_input, outputs=[out_x, out_y])

    train_model(model, args, nlayers=nlayers, cell_type=cell_type)


if __name__ == '__main__':
    main()
