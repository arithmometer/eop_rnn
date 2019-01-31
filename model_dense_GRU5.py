from keras.layers import Input, GRU, Dense
from keras.models import Model

from train_model import parse_args, train_model


def main():
    args = parse_args()

    nlayers = 5
    cell_type = 'GRU'

    net_input = Input(shape=(args.lookback, 2))
    
    x = Dense(300, activation='relu')(net_input)
    x = Dense(args.ncells, activation='relu')(x)
    x = GRU(args.ncells, return_sequences=True, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)
    x = Dense(args.ncells, activation='relu')(x)
    x = GRU(args.ncells, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)(x)

    out_x = Dense(args.delay, activation='linear')(x)
    out_y = Dense(args.delay, activation='linear')(x)

    model = Model(inputs=net_input, outputs=[out_x, out_y])

    train_model(model, args, nlayers=nlayers, cell_type=cell_type)


if __name__ == '__main__':
    main()
