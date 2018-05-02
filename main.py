import numpy as np
import pickle
import math
import os
import re
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision


def load_data(file_path):
    """ Loads the data from a pickled file located at the given file_path. """
    with open(file_path, 'rb') as f:
        u = pickle.Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

def generate_data():
    path = 'cifar-10-batches-py'
    data = load_data(path + '/data_batch_1')
    x = data['data']
    y = data['labels']
    label_list = [1, 5, 6]
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
    label_seen = [1, 5]
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

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return x
'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


max_epochs = 5
def main():
    net = VGG('VGG11')
    x_train, y_train, x_valid, y_valid, x_test_unseen, y_test_unseen = generate_data()
    print len(x_train)
    dataset_train = MyTrainingSet(x_train, y_train)
    train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    loss_np = np.zeros((max_epochs))
    accuracy = np.zeros((max_epochs))

    for epoch in range(max_epochs):
        correct = 0
        for i, data in enumerate(train_loader, 0):
            print "-- each iter -- "
            # Get inputs and labels from data loader 
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # Feed the input data into the network 
            y_pred = net(inputs)
            
            # Calculate the loss using predicted labels and ground truth labels
            loss = criterion(y_pred, labels)
            
            # print("epoch: ", epoch, "loss: ", loss.data[0])
            
            # zero gradient
            optimizer.zero_grad()
            
            # backpropogates to compute gradient
            loss.backward()
            
            # updates the weghts
            optimizer.step()
            
            # convert predicted labels into numpy
            y_pred_np = y_pred.data.numpy()
            label_np = labels.data.numpy().reshape(len(labels),1)
            
            # calculate the training accuracy of the current model
            for j in range(y_pred_np.shape[0]):
                idx_pred = 0
                for i in xrange(len(y_pred_np[j, :])):
                    if y_pred_np[j, i] == max(y_pred_np[j, :]):
                        idx_pred = i
                if idx_pred == label_np[j, 0]:
                    correct += 1
            
            loss_np[epoch] = loss.data.numpy()
        accuracy[epoch] = float(correct)
        # print ("epoch: ", epoch, "loss: ", loss_np[epoch])
        # print ("epoch: ", epoch, "accuracy: ", accuracy[epoch])
        print accuracy[epoch]
    dataset = MyTestingSet(x_valid, y_valid)
    test_loader = DataLoader(dataset=dataset, batch_size = 64, shuffle = True)
    net.eval()
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = net(inputs)
        y_pred_np = y_pred.data.numpy()
        label_np = labels.data.numpy().reshape(len(labels),1)
        for j in range(y_pred_np.shape[0]):
            idx_pred = 0
            for i in xrange(len(y_pred_np[j, :])):
                if y_pred_np[j, i] == max(y_pred_np[j, :]):
                    idx_pred = i
            if idx_pred == label_np[j, 0]:
                correct += 1
    print("final testing accuracy: ", float(correct))


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
