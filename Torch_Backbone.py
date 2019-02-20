###pytorch set-up, ready for model###
##https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.autograd. import Variable
import torch.utils.data as utils


##Data
x = images array
y = labels array

tensor_x = torch.stack([torch.Tensor(i) for i in x])
tensor_y = torch.stack([torch.Tensor(i) for i in y])

trainset = utils.TensorDataset(tensor_x,tensor_y)
testset = utils.TensorDataset(tensor_x,tensor_y)

trainloader = utils.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
testloader = utils.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


classes = (‘seize’, 'not_seize')


##GPU Enable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
inputs, labels = inputs.to(device), labels.to(device)

##Model
net = DenseNet3D()


#Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([var1, var2], lr = 0.0001)
epochs = 3

##Train
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad() # zero the parameter gradients
        outputs = net(inputs) # forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() # print statistics
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


##Test
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network : %d %%' % (100 * correct / total))
