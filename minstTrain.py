from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
BatSize = 32
EPOCHS = 100
DEVICES = [torch.device("cuda:0"), torch.device("cpu")]
pipeline = transforms.Compose([transforms.ToTensor(),  # 将图片转换为Tensor 并且归一化,并且只能有一个张多图层的图片
                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # 正则化 降低模型复杂度
                               ])
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
train_data = DataLoader(train_set, shuffle=True, batch_size=16)
test_data = DataLoader(test_set, shuffle=False, batch_size=16)
criterion = torch.nn.CrossEntropyLoss()


class Net(torch.nn.Module):
    def __init__(self, pic_size):
        super().__init__()
        self.pic_size = pic_size
        self.l1 = torch.nn.Linear(self.pic_size, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 32)
        self.l6 = torch.nn.Linear(32, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 这一行是必须有的，因为必须把图片转化成一个维度的向量才可以
        x = x.view(-1, self.pic_size)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        x = self.l6(x)  # 最后一层是不做激活的，留给交叉熵做
        return x


def training_with_dataloader(epoch, device):
    loss_sum = 0
    count = 0
    model = Net(784).to(device)
    # 设计这个冲量非常的聪明，对于导数正负不断交替的变量说明它步伐跨度会比较大，就需要使用冲量去减少，
    # 对于变化不大的就可以使用冲量去增大它的变化幅度
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for i, data in enumerate(train_data, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # Move data to the device
        y_hat = model(inputs)
        loss = criterion(y_hat, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_sum += loss.item()
        count += 1
    # print("epoch: 第", epoch, "轮的 loss: ", loss.item() / count)
    return model, loss.item() / count

def testing_with_dataloader(model, device):
    size = 0
    correct = 0
    with torch.no_grad():
        for data in test_data:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)  # dim指的是在哪个维度求最大值
            size += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/size))


if __name__ == '__main__':
    times = []
    for device in DEVICES:
        start = time.time()
        print("--------------", device, "---------------")
        for epoch in range(10):
            model, loss = training_with_dataloader(epoch, device)
            testing_with_dataloader(model, device)
        end = time.time()
        times.append(end - start)
    print("GPU使用时间： ", times[0])
    print("CPU使用时间： ", times[1])