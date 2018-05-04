import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import pickle
from sklearn.mixture import GaussianMixture


class ImageToSemanticMappingNetwork(nn.Module):
    def __init__(self):
        super(ImageToSemanticMappingNetwork, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 50)

    def forward(self, X):
        X = X.view(-1, 3 * 32 * 32)
        out = F.tanh(self.fc1(X))
        out = self.fc2(out)
        return out

class Dataset(Dataset):
    def __init__(self, X, Y):
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).float()
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def read_data(file_path):
    """ Loads the data from a pickled file located at the given file_path. """
    with open(file_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

def load_data(label_seen, label_unseen):
    root_path = '../cifar-10-batches-py'
    class_names = read_data(root_path + '/batches.meta')['label_names']
    print("class names:", class_names)

    x = None
    y = []
    for i in range(1, 6):
        data = read_data(root_path + '/data_batch_' + str(i))
        if x is None:
            x = data['data']
        else:
            x = np.vstack((x, data['data']))
        y = y + data['labels']

    x = x.reshape((50000, 3, 32, 32))
    y = np.array(y)

    idx_seen = np.in1d(y, label_seen)
    idx_unseen = np.in1d(y, label_unseen)

    x_seen = x[idx_seen]
    x_unseen = x[idx_unseen]
    y_seen = y[idx_seen]
    y_unseen = y[idx_unseen]

    return class_names, x_seen, y_seen, x_unseen, y_unseen


def train_mapping(data_loader, max_epochs):
    image_to_semantic_nn = ImageToSemanticMappingNetwork()
    optimizer = optim.SGD(image_to_semantic_nn.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = nn.MSELoss()
    losses = np.zeros((max_epochs))

    for epoch in range(max_epochs):
        num_example = 0.0
        loss_sum = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            evaluations = image_to_semantic_nn(inputs)
            loss = criterion(evaluations, labels)
            loss_sum += loss[0][0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted_labels = torch.max(evaluations, 1)
            num_example += labels.size()[0]
            
        losses[epoch] = float(loss_sum) / float(num_example)
        print("epoch ", epoch, "done")


    print("final loss: ", losses[-1])
    epoch_number = np.arange(0,max_epochs,1)

    # Plot the loss over epoch
    plt.figure()
    plt.plot(epoch_number, losses)
    plt.title('loss over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    return image_to_semantic_nn

def evaluate(model, data_loader):
    res = None
    for data in data_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        evaluations = model(inputs).data.numpy()
        if res is None:
            res = evaluations
        else:
            res = np.vstack((res, evaluations))

    return res

def main():
    word_to_vector = {
    "airplane" : [1.2977, -0.29922, 0.66154, -0.20133, -0.02502, 0.28644, -1.0811, 
    -0.13045, 0.64917, -0.33634, 0.53352, 0.32792, -0.43206, 1.4613, 0.022957, -0.26019, -1.1061, 
    1.077, -0.99877, -1.3468, 0.39016, 0.43799, -1.0403, -0.36612, 0.39231, -1.3089, -0.82404, 
    0.63095, 1.2513, 0.10211, 1.2735, -0.0050163, -0.39469, 0.36387, 0.65099, -0.21433, 0.52291, 
    -0.079013, -0.14676, 0.89248, -0.31447, 0.090903, 0.78216, -0.10842, -0.3186, 0.16068, -0.20168, 
    -0.095033, -0.010109, 0.19048],
    "automobile" : [-0.41195, 0.069058, 0.26701, 0.41424, -0.91901, 0.63319, -0.89194, -0.53483, 
    0.19187, -0.038827, 1.1475, -0.1396, -0.66392, -0.19639, 0.30304, -0.06703, -0.95611, 1.6306, 
    0.17545, -1.6013, 1.2995, -1.0079, -1.7455, -0.00058892, -0.021532, -0.97641, -0.93735, 0.040884, 
    0.31757, 0.55358, 1.5822, 0.14179, 0.37018, 0.39469, 0.47537, -0.53013, -0.043661, 0.42126, 
    0.29403, 0.80253, -0.61572, -0.76155, 0.9184, -0.72823, 0.59806, -0.16884, -0.59675, 0.16543, 
    0.89073, -0.060983],
    "bird" : [0.78675, 0.079368, -0.76597, 0.1931, 0.55014, 0.26493, -0.75841, -0.8818, 1.6468, 
    -0.54381, 0.33519, 0.44399, 1.089, 0.27044, 0.74471, 0.2487, 0.2491, -0.28966, -1.4556, 
    0.35605, -1.1725, -0.49858, 0.35345, -0.1418, 0.71734, -1.1416, -0.038701, 0.27515, -0.017704, 
    -0.44013, 1.9597, -0.064666, 0.47177, -0.03, -0.31617, 0.26984, 0.56195, -0.27882, -0.36358, 
    -0.21923, -0.75046, 0.31817, 0.29354, 0.25109, 1.6111, 0.7134, -0.15243, -0.25362, 0.26419, 
    0.15875],
    "cat" : [0.45281, -0.50108, -0.53714, -0.015697, 0.22191, 0.54602, -0.67301, -0.6891, 0.63493, 
    -0.19726, 0.33685, 0.7735, 0.90094, 0.38488, 0.38367, 0.2657, -0.08057, 0.61089, -1.2894, 
    -0.22313, -0.61578, 0.21697, 0.35614, 0.44499, 0.60885, -1.1633, -1.1579, 0.36118, 0.10466, 
    -0.78325, 1.4352, 0.18629, -0.26112, 0.83275, -0.23123, 0.32481, 0.14485, -0.44552, 0.33497, 
    -0.95946, -0.097479, 0.48138, -0.43352, 0.69455, 0.91043, -0.28173, 0.41637, -1.2609, 0.71278,
     0.23782],
    "dog" : [0.11008, -0.38781, -0.57615, -0.27714, 0.70521, 0.53994, -1.0786, -0.40146, 1.1504, 
    -0.5678, 0.0038977, 0.52878, 0.64561, 0.47262, 0.48549, -0.18407, 0.1801, 0.91397, -1.1979, 
    -0.5778, -0.37985, 0.33606, 0.772, 0.75555, 0.45506, -1.7671, -1.0503, 0.42566, 0.41893, 
    -0.68327, 1.5673, 0.27685, -0.61708, 0.64638, -0.076996, 0.37118, 0.1308, -0.45137, 0.25398, 
    -0.74392, -0.086199, 0.24068, -0.64819, 0.83549, 1.2502, -0.51379, 0.04224, -0.88118, 0.7158,
    0.38519],
    "deer" : [-0.0014181, -0.012513, -0.11606, -0.32099, 0.30832, 0.28235, -1.3521, -1.8643, 
    1.1219, -0.83093, -0.16311, -0.025823, 1.0296, -0.46624, 0.08404, 1.2953, 1.5536, 0.18442, 
    -1.6419, 0.53065, -1.1949, -0.90213, 1.0302, 0.54902, 0.10129, -0.83007, -0.54873, 0.64926, 
    0.3829, -1.1255, 0.68471, 0.47026, -0.39548, 0.26924, 0.76423, 0.30521, -0.075649, -0.48568, 
    -0.18858, 0.70855, -1.3426, 0.69116, -0.50315, 0.93529, 1.2236, -0.88088, 0.36148, -0.8275, 
    0.9807, -0.49068],
    "horse" : [-0.20454, 0.23321, -0.59158, -0.29205, 0.29391, 0.31169, -0.94937, 0.055974, 
    1.0031, -1.0761, -0.0094648, 0.18381, -0.048405, -0.35717, 0.26004, -0.41028, 0.51489, 1.2009,
    -1.6136, -1.1003, -0.23455, -0.81654, -0.15103, 0.37068, 0.477, -1.7027, -1.2183, 0.038898,
    0.23327, 0.028245, 1.6588, 0.26703, -0.29938, 0.99149, 0.34263, 0.15477, 0.028372, 0.56276, 
    -0.62823, -0.67923, -0.163, -0.49922, -0.8599, 0.85469, 0.75059, -1.0399, -0.11033, -1.4237, 
    0.65984, -0.3198],
    "frog" : [0.61038, -0.20757, -0.71951, 0.89304, 0.32482, 0.76564, 0.1814, -0.33086, 0.79173, 
    -0.31664, 0.011143, 0.45412, 1.5992, 0.013494, -0.093646, 0.19245, 0.251, 1.1277, -1.0897, 
    -0.42909, -1.1327, -0.90465, 0.5617, -0.058464, 1.0007, -0.39017, -0.41665, 0.73721, -0.53824, 
    -0.95993, 0.67929, -0.59053, 0.13408, 0.54273, -0.36615, 0.014978, -0.2496, -0.81088, 0.078905, 
    -0.97552, -0.66394, -0.18508, -0.87174, 0.30782, 1.2839, -0.14884, 0.62178, -1.509, 0.14582, 
    -0.31682],
    "ship" : [1.5213, 0.10522, 0.38162, -0.50801, 0.032423, -0.13484, -1.2474, 0.79813, 0.84691, 
    -1.101, 0.88743, 1.3749, 0.42928, 0.65717, -0.2636, -0.41759, -0.48846, 0.91061, -1.7158, 
    -0.438, 0.78395, 0.19636, -0.40657, -0.53971, 0.82442, -1.7434, 0.14285, 0.28037, 1.1688, 
    0.16897, 2.2271, -0.58273, -0.45723, 0.62814, 0.54441, 0.28462, 0.44485, -0.55343, -0.36493, 
    -0.016425, 0.40876, -0.87148, 1.5513, -0.80704, -0.10036, -0.28461, -0.33216, -0.50609, 0.48272, 
    -0.66198],
    "truck" : [0.35016, -0.36192, 1.505, -0.070263, 0.32708, 0.48106, -1.4825, 0.07962, 0.83452, 
    -0.72912, 0.19233, -0.90769, -0.89611, 0.33796, 0.42153, -0.47797, -0.47473, 1.6142, -0.5358, 
    -1.6758, 0.64926, 0.074053, -0.66378, 0.66352, -0.11525, -1.46, -0.31867, 0.99803, 1.636, 
    -0.11678, 1.8673, -0.19582, -0.50549, 0.82963, 1.3381, 0.33233, 0.24957, -0.37286, 0.2777, 
    0.88405, -0.29343, -0.0033666, 0.27167, -1.1805, 0.53095, -0.31678, -0.3141, -0.31516, 0.96377, 
    -0.55119]}

    batch_size = 400
    threshold = 0
    label_seen = [i for i in range(8)]
    label_unseen = [i for i in range(8, 10)]

    class_names, x_seen, y_seen, x_unseen, y_unseen = load_data(label_seen, label_unseen)
    x_seen_train = x_seen[: 32000]
    y_seen_train = y_seen[: 32000]
    x_seen_test = x_seen[32000 :]
    y_seen_test = y_seen[32000 :]
    y_seen_train_semantic = np.asarray([word_to_vector[class_names[i]] for i in y_seen_train])
    y_seen_test_semantic = np.asarray([word_to_vector[class_names[i]] for i in y_seen_test])
    y_unseen_semantic = np.asarray([word_to_vector[class_names[i]] for i in y_unseen])

    seen_train_dataset = Dataset(x_seen_train, y_seen_train_semantic)
    seen_test_dataset = Dataset(x_seen_test, y_seen_test_semantic)
    unseen_dataset = Dataset(x_unseen, y_unseen_semantic)
    class_idx_to_seen_train_dataset = []
    for i in label_seen:
        idx_label_i = np.in1d(y_seen_train, i)
        x_seen_train_i = x_seen_train[idx_label_i]
        y_seen_train_semantic_i = y_seen_train_semantic[idx_label_i]
        class_idx_to_seen_train_dataset.append(Dataset(x_seen_train_i, y_seen_train_semantic_i))


    seen_train_loader = DataLoader(dataset=seen_train_dataset, batch_size=batch_size, shuffle=True)
    seen_test_loader = DataLoader(dataset=seen_test_dataset, batch_size=batch_size, shuffle=True)
    unseen_loader = DataLoader(dataset=unseen_dataset, batch_size=batch_size, shuffle=True)

    # image_to_semantic_nn = ImageToSemanticMappingNetwork()
    # image_to_semantic_nn.load_state_dict(torch.load("imageFeatureToSemantic"))
    image_to_semantic_nn = train_mapping(seen_train_loader, 100)
    torch.save(image_to_semantic_nn.state_dict(), "imageFeatureToSemantic")

    seen_test_semantic = evaluate(image_to_semantic_nn, seen_test_loader)
    unseen_image_smatic = evaluate(image_to_semantic_nn, unseen_loader)
    # print(unseen_image_smatic, len(unseen_image_smatic), len(unseen_image_smatic[0]))
    
    class_name_to_model = {}
    for i in label_seen:
        seen_semantic_i = evaluate(image_to_semantic_nn, DataLoader(dataset=class_idx_to_seen_train_dataset[i]))
        gaussian_model = GaussianMixture(n_components=1, verbose=0, max_iter=50, means_init=np.reshape(word_to_vector[class_names[i]], (1, 50)))
        gaussian_model.fit(seen_semantic_i)
        class_name_to_model[class_names[i]] = gaussian_model
        print("building gaussian model for", class_names[i], "done")

    # test unseen data
    output_unseen = []
    for i in label_seen:
        gaussian_model = class_name_to_model[class_names[i]]
        output_unseen.append(gaussian_model.score_samples(unseen_image_smatic))
    output_unseen = np.transpose(output_unseen)

    y_values_unseen = [i for i in np.amax(output_unseen, axis=1)[: 100] if i >= 0]
    plt.figure()
    plt.plot([i for i in range(len(y_values_unseen))], y_values_unseen)
    plt.title('log prob unseen')
    plt.xlabel('idx')
    plt.ylabel('log prob')

    output_seen = []
    for i in label_seen:
        gaussian_model = class_name_to_model[class_names[i]]
        output_seen.append(gaussian_model.score_samples(seen_test_semantic))
    output_seen = np.transpose(output_seen)

    y_values_seen = [i for i in np.amin(output_seen, axis=1)[: 100] if i >= 0]
    plt.figure()
    plt.plot([i for i in range(len(y_values_seen))], y_values_seen)
    plt.title('log prob seen')
    plt.xlabel('idx')
    plt.ylabel('log prob')
    plt.show()



if __name__ == '__main__':
    main()