import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    """ Loads the data from a pickled file located at the given file_path. """
    with open(file_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

def generate_data():
    path = 'cifar-10-batches-py'
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

    # split the data, to train, test
    x_train = x[:12000]
    y_train = y[:12000]
    x_split = x[12000:]
    y_split = y[12000:]
    label_seen = [3, 5]
    idx = np.in1d(y_train, label_seen)
    x_train = x_train[idx]
    y_train = y_train[idx]

    idx = np.in1d(y_split, label_seen)
    x_valid = x_split[idx]
    y_valid = y_split[idx]

    idx = np.in1d(y_split, [6])
    x_test_unseen = x_split[idx]
    y_test_unseen = y_split[idx]
    return x_train, y_train, x_valid, y_valid, x_test_unseen, y_test_unseen


class MyTrainingSet(Dataset):

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(y).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class MyTestingSet(Dataset):

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(y).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return x


def main():
    x_train, y_train, x_valid, y_valid, x_test_unseen, y_test_unseen = generate_data()
    '''
    print x_train.shape
    print "==============="
    print y_train.shape
    print "==============="
    print x_valid.shape
    print "==============="
    print y_valid.shape
    print "==============="
    print x_test_unseen.shape
    print "==============="
    '''
    
    pass


main()
