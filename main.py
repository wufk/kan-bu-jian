import numpy as np
import pickle


def load_data(file_path):
    """ Loads the data from a pickled file located at the given file_path. """
    with open(file_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data


def generate_data():
    path = '../cifar-10-batches-py'
    data = load_data(path + '/data_batch_1')
    x = data['data']
    y = data['labels']
    label_list = [3, 5, 6]
    for i in range(2, 6):
        data = load_data(path + '/data_batch_' + str(i))
        x = np.vstack((x, data['data']))
        y = y + data['labels']
    x = x.reshape((50000, 3, 32, 32))
    y = np.array(y)
    idx = np.in1d(y, label_list)
    y = y[idx]
    x = x[idx]
    return x, y


def main():
    x_train, y_train = generate_data()
    print(x_train.shape)
    pass


main()
