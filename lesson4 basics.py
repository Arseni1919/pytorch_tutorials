import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('---')
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    print(np.transpose(npimg, (1,2,0)).shape)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# get random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# print(images.size())
# print(torchvision.utils.make_grid(images).size())
'''
torchvision.utils.make_grid - makes from batch of pictures
one big picture with paddings and stuff
The output: is one picture
'''
imshow(torchvision.utils.make_grid(images, padding=10))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print('---')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(2): # loop over the dataset multiple times

    runnin_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        runnin_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d,%5d] loss: %.3f' %
                  (epoch + 1, i + 1, runnin_loss/2000))
            runnin_loss = 0.0

print("Finnished Training")
print('---')
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted:', ' '.join('%5s' % classes[predicted[j]]
                             for j in range(4)))

print('---')
correct = 0
total = 0
counter = 0
with torch.no_grad():
    for data in testloader:
        counter += 1
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if counter % 1001 == 1000:
            print(predicted == labels)
            print(type(predicted == labels))
            counter = 0
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images:  %d %%' %(
    100 * correct / total
))
print('---')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i]/(class_total[i] + 0.0001)
    ))
print('---')
print('---')