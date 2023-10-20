import torch

data = torch.randn(5, 1, 28, 28)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.Relu = torch.nn.ReLU()
        # self.fc

    def forward(self, x):
        bach_size = x.size(0)
        x = self.Relu(self.pooling(self.conv1(x)))
        x = self.Relu(self.pooling(self.conv2(x)))
        x = x.view(bach_size, -1)
        print(x.size())


model = Net()
model(data)