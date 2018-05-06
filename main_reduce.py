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
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA


def load_data(file_path):
    """ Loads the data from a pickled file located at the given file_path. """
    with open(file_path, 'rb') as f:
        u = pickle.Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data


label_list = [1, 2, 3, 4, 5]
label_seen = [1, 2, 3, 4]
label_unseen = [5]

def generate_data():
    path = 'cifar-10-batches-py'
    data = load_data(path + '/data_batch_1')
    x = data['data']
    y = data['labels']
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
    x_train = x[:20000]
    y_train = y[:20000]
    x_split = x[20000:]
    y_split = y[20000:]
    idx = np.in1d(y_train, label_seen)
    x_train = x_train[idx]
    y_train = y_train[idx]
    print len(x_train)

    idx = np.in1d(y_split, label_seen)

    x_valid = x_split[idx]
    y_valid = y_split[idx]
    print "============="
    print len(x_valid)
    idx = np.in1d(y_split, label_unseen)
    x_test_unseen = x_split[idx]
    y_test_unseen = y_split[idx]
    print len(x_test_unseen)
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


def get_feature_dist(lab_vec1, lab_vec2):
    res = 0
    for i in range(len(lab_vec1)):
        res += ((float(lab_vec1[i]) - lab_vec2[i]) * (float(lab_vec1[i]) - lab_vec2[i]))
    return res


label_cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']
def get_distance_mat():
    stand_dist = {}
    glove = open('glove.6B.50d.txt')
    for ele in glove:
        temp = ele.split()[0]
        if temp in label_cifar10:
            stand_dist[temp] = ele.split()[1:]
    whole_array = []
    for i in label_cifar10:
        tempp = []
        for j in stand_dist[i]:
            tempp.append(float(j))
        whole_array.append(tempp)

    pca = PCA(n_components=4)
    pca.fit(whole_array)
    whole_array = pca.transform(whole_array)
    standard = {}
    for i in range(len(label_cifar10)):
        label = label_cifar10[i]
        vec = whole_array[i]
        standard[label] = vec
    return standard

def normalize_vec(vec):
    vec_res = []
    norm = 0
    for i in vec:
        norm += (float(i) * float(i))
    norm = math.sqrt(norm)
    for i in vec:
        vec_res.append(float(i) / norm)
    return vec_res

def convert_to_feature_space(stand_dist, pred_y):
    res = []
    for img_vec in pred_y:
        temp = []
        for i in range(len(img_vec)):
            if i not in label_seen:
                continue
            label_prob = img_vec[i]
            label_prob = float(label_prob)
            vec = normalize_vec(stand_dist[label_cifar10[i]])
            #vec = stand_dist[label_cifar10[i]]
            if len(temp) == 0:
                for j in range(len(vec)):
                    temp.append(float(vec[j]) * label_prob)
            else:
                for j in range(len(vec)):
                    temp[j] += (float(vec[j]) * label_prob)
        res.append(temp)
    return res


cfg = {'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}

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

    net = VGG('VGG13')
    net.load_state_dict(torch.load("saved_model_reduce"))
    x_train, y_train, x_valid, y_valid, x_test_unseen, y_test_unseen = generate_data()

    '''
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
        print accuracy[epoch]
    torch.save(net.state_dict(), "saved_model_reduce")
    '''

    # test on validation data (seen labels)
    dataset = MyTestingSet(x_valid, y_valid)
    test_loader = DataLoader(dataset=dataset, batch_size = 64, shuffle = True)
    net.eval()
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = net(inputs)
        #y_pred_np = y_pred.data.numpy()
        stand_dist = get_distance_mat()
        y_pred_np = convert_to_feature_space(stand_dist, y_pred)
        label_np = labels.data.numpy().reshape(len(labels),1)
        for j in range(len(y_pred_np)):
            temp_max = 100000000000
            label_temp = -1
            for i in label_seen:
                vec_stand = normalize_vec(stand_dist[label_cifar10[i]])
                dis = cosine(vec_stand, y_pred_np[j])
                if temp_max > dis:
                    temp_max = dis
                    label_temp = i
            if label_temp == label_np[j][0]:
                correct += 1
    print("final validation accuracy: ", float(correct) / len(x_valid))


    # test on validation data (seen labels)
    dataset = MyTestingSet(x_test_unseen, y_test_unseen)
    test_loader = DataLoader(dataset=dataset, batch_size = 64, shuffle = True)
    avg = [0.204560809204, 1.08281865111, 0.157749290337, 1.53585196356]
    stdd = [0.392214186359, 0.707691146135, 0.24491374882, 0.312214809199]
    whole_arr = [[], [], [], []]
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = net(inputs)
        #y_pred_np = y_pred.data.numpy()
        stand_dist = get_distance_mat()
        y_pred_np = convert_to_feature_space(stand_dist, y_pred)
        label_np = labels.data.numpy().reshape(len(labels),1)
        for j in range(len(y_pred_np)):
            temp_max = 100000000000
            label_temp = -1
            label_unseen_mix = [5, 6, 7, 8]
            idxx = 0
            print "=============="
            for i in label_unseen_mix:
                vec_stand = normalize_vec(stand_dist[label_cifar10[i]])
                dis = cosine(vec_stand, y_pred_np[j])
                whole_arr[idxx].append(dis)
                #std_dis = (dis - avg[idxx]) * 1.0 / stdd[idxx]
                print i
                print dis
                if temp_max > dis:
                    temp_max = dis
                    label_temp = i
                idxx += 1
            print '------------'
            print label_temp
            print label_np[j][0]
            print '------------'
            if label_temp == label_np[j][0]:
                correct += 1
            print "=============="
    for i in range(4):
        print np.std(np.array(whole_arr[i]))
        print np.mean(np.array(whole_arr[i]))
    print("test with one unseen mix label accuracy: ", float(correct) / len(x_test_unseen))

    pass

main()










