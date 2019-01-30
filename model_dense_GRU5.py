from keras.layers import Input, GRU, Dense, add
from keras.models import Model

from train_model import parse_args, train_model


def main():
    args = parse_args()

    nlayers = 5
    cell_type = 'GRU'

    net_input = Input(shape=(args.lookback, 2))
    
    x_0 = Dense(100, activation='relu')(net_input)
    x_1 = GRU(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x_0)
    x_2 = GRU(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x_1)
    x_3 = GRU(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(add([x_1, x_2]))
    x = GRU(args.ncells, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(add([x_1, x_2, x_3]))

    out_x = Dense(args.delay, activation='linear')(x)
    out_y = Dense(args.delay, activation='linear')(x)

    model = Model(inputs=net_input, outputs=[out_x, out_y])

    train_model(model, args, nlayers=nlayers, cell_type=cell_type)


if __name__ == '__main__':
    main()
