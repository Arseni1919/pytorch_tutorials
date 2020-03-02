import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # print('------x.shape:',x.shape)
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        # print('------x.shape:', x.shape)
        x = F.max_pool2d(F.relu(x), 2)
        # print('------x.shape:', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
print('---')

for param in net.parameters():
    print(type(param.data), param.size())
print(list(net.parameters())[1])
print('---')

input = torch.randn(1,1,32,32)
out = net(input)
print('out:', out)

net.zero_grad()
out.backward(torch.randn(1,10))

print('---')
out = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions)

print('---')

net.zero_grad() # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print('---')
# SGD: weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
print('---')
import torch.optim as optim

# create your optimazer
optimazer = optim.SGD(net.parameters(), lr = 0.01)
print('grads:',net.conv1.bias.grad)
print('biases:',net.conv1.bias)
# in your training loop:
optimazer.zero_grad() # zero the gradient buffers
print('grads:',net.conv1.bias.grad)
print('biases:',net.conv1.bias)
output = net(input)
loss = criterion(output, target)
loss.backward()
print('grads:',net.conv1.bias.grad)
print('biases:',net.conv1.bias)
optimazer.step() # Does the update
print('grads:',net.conv1.bias.grad)
print('biases:',net.conv1.bias)
print('---')





