import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_ch):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 0)
        self.conv3 = nn.Conv2d(8,16, 3, 1, 0)
        self.conv4 = nn.Conv2d(16, 32, 4, 1, 0)
        self.maxpool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(32, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.maxpool(self.activation(self.conv1(x)))
        x = self.maxpool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.view(-1, 32)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x