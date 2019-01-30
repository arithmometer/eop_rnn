import numpy as np


def mjd_to_row(mjd):
    return mjd - 37665


def single_model_generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            2))
        targets_x = np.zeros((len(rows), delay))
        targets_y = np.zeros((len(rows), delay))
        # targets_lod = np.zeros((len(rows), delay))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = np.reshape(data[indices], (lookback, 2))
            targets_x[j] = data[rows[j]:rows[j] + delay, 0]
            targets_y[j] = data[rows[j]:rows[j] + delay, 1]
            # targets_lod[j] = data[rows[j]:rows[j] + delay, 2]

        # yield samples, [targets_x, targets_y, targets_lod]
        yield samples, [targets_x, targets_y]
