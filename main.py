import numpy as np
import csv
import matplotlib.pyplot as plt
training = 28709 + 1
testing = 3589
with open('fer2013.csv', newline='') as csvfile:
     data = list(csv.reader(csvfile))

train = np.ndarray(shape=(28709,1,48,48), dtype=np.float32)
test = np.ndarray(shape=(3589,1,48,48), dtype=np.float32)
test2 = np.ndarray(shape=(3589,1,48,48), dtype=np.float32)

label_train = np.ndarray(shape=(28709), dtype=int)
label_test = np.ndarray(shape=(3589), dtype=int)
label_test2 = np.ndarray(shape=(3589), dtype=int)

for i in range (1, training):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 # Normalization
    train[i-1,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_train[i-1] = label


location = "train_images.npy"
np.save(location, train)
location = "train_labels.npy"
np.save(location,label_train)

# Valid
for i in range (training, training+testing):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 #Normalization
    test[i-training,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_test[i-training] = label


location = "valid_images.npy"
np.save(location, test)
location = "valid_labels.npy"
np.save(location,label_test)


# Test
for i in range (training+testing, training+2*testing):
    image = [int(j) for j in data[i][1].split()]
    image = np.asarray(image)
    image = image.reshape(1, 48,48)
    image = image / 255 #Normalization
    test2[i-training-testing,:,:,:] = image
    label = np.zeros(7)
    label[int(data[i][0])] = 1
    label = int(data[i][0])
    label_test2[i-training-testing] = label

location = "test_images.npy"
np.save(location, test2)
location = "test_labels.npy"
np.save(location,label_test2)

def plot(train, valid, path):
    plt.plot(train, label="train")
    plt.plot(valid, label="valid")
    plt.legend()
    plt.savefig(path)
    plt.clf()

import pandas as pd
import numpy as np
import torch

import random

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

"""**Hyperparameters**"""

batch_size=32
epochs=200
model_name="vgg"
seed=0
lr=0.01
use_scheduler= True
exp_id= 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class FERDataReader(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        super(FERDataReader, self).__init__()

        self.input_images = torch.from_numpy(np.load('{}_images.npy'.format(mode)))
        self.target_classes = torch.from_numpy(np.load('{}_labels.npy'.format(mode)))


    def __getitem__(self, index):
        return self.input_images[index], self.target_classes[index]

    def __len__(self):
        return self.target_classes.shape[0]

train_loader = torch.utils.data.DataLoader(FERDataReader(mode='train'), batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(FERDataReader(mode='valid'), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(FERDataReader(mode='test'), batch_size=batch_size, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=1)



        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512*6*6, 500)
        self.fc2 = nn.Linear(500,7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
      print(x.shape)
      x = self.pool(F.relu(self.conv2(x)))
      print(x.shape)
      x = self.pool(F.relu(self.conv4(x)))
      print(x.shape)
      x = self.pool(F.relu(self.conv7(x)))
      print(x.shape)
      x = self.pool(F.relu(self.conv10(x)))
      print(x.shape)
      x = self.pool(F.relu(self.conv13(x)))
      print(x.shape)



      x = x.view(-1, 512 * 6 * 6)
      print(x.shape)

      x = self.dropout(x)
      print(x.shape)
      x = F.relu(self.fc1(x))
      print(x.shape)
      x = self.dropout(x)
      print(x.shape)
      x = self.fc2(x)
      print(x.shape)

      return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.batch5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.batch6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch10 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(2304, 1024)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024,7)


    def forward(self, x):
      #print(x.shape)
      x = F.relu(self.batch1(self.conv1(x)))
      #print(x.shape)
      x = F.relu(self.batch2(self.conv2(x)))
      #print(x.shape)
      x = F.relu(self.batch3(self.conv3(x)))
      #print(x.shape)
      x = self.pool(F.relu(self.batch4(self.conv4(x))))
      x = F.relu(self.batch5(self.conv5(x)))
      x = F.relu(self.batch6(self.conv6(x)))
      x = F.relu(self.batch7(self.conv7(x)))
      x = self.pool(F.relu(self.batch8(self.conv8(x))))
      x = F.relu(self.batch9(self.conv9(x)))
      x = self.pool(F.relu(self.batch10(self.conv10(x))))




      x = x.view(-1, 2304)
      #print(x.shape)

      x = self.dropout(x)
      #print(x.shape)
      x = F.relu(self.fc1(x))
      #print(x.shape)
      x = self.dropout(x)
      #print(x.shape)
      x = self.fc2(x)
      #print(x.shape)

      return x

import torch.optim as optim
if model_name=="basic":
  model=Net()
elif model_name=="vgg":
  model=VGG()
else:
  print("invalid model_name")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
if use_scheduler:
  scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

t_loss, v_loss = [],[]


if train_on_gpu:
  model.cuda()


valid_loss_min=np.inf

for epoch in range(epochs):

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, label in train_loader:
        if train_on_gpu:
          data=data.cuda()
          label=label.cuda()
        optimizer.zero_grad()
        output = model(data)
        #print(data.shape, label.shape, output.shape)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if use_scheduler:
      scheduler.step()
    model.eval()
    for data, label in valid_loader:
        if train_on_gpu:
          data=data.cuda()
          label=label.cuda()
        output = model(data)
        loss = criterion(output, label)
        valid_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    valid_loss = valid_loss/len(valid_loader)
    t_loss.append(train_loss)
    v_loss.append(valid_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'drive/My Drive/Internship/{}_model_emotion_{}.pt'.format(exp_id, model_name))
        valid_loss_min = valid_loss

plot(t_loss, v_loss, 'drive/My Drive/Internship/{}_loss_{}.png'.format(exp_id, model_name))

#model.load_state_dict(torch.load('model_cifar.pt'))x

test_loss=0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
model.eval()

for data, label in test_loader:
    if train_on_gpu:
          data=data.cuda()
          label=label.cuda()
    output = model(data)
    loss = criterion(output, label)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(label.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().detach().numpy())
    label=label.cpu().detach().numpy()
    #print(label.shape)
    bn=label.shape[0]
    for i in range(bn):
        c_label= label[i]
        class_correct[c_label] += correct[i].item()
        class_total[c_label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(7):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            emotions[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

