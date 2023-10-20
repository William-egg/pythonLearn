import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

deviceGpu = torch.device("cuda:0")
deviceCpu = torch.device("cpu")
devices = [deviceGpu, deviceCpu]


class DiabetesDataSet(Dataset):
    def __init__(self, filepath, test_filepath, device):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.x_data = torch.Tensor(xy[:, :-1]).to(device)
        self.y_data = torch.Tensor(xy[:, [-1]]).to(device)
        self.len = xy.shape[0]
        self.XDimension = self.x_data.size(1)
        self.YDimension = self.y_data.size(1)
        # diabetesTestData.csv.gz
        xy_test = np.loadtxt(test_filepath, delimiter=",", dtype=np.float32)
        # 对这个地方进行更改是一个非常厉害的行为了，记录一下
        xy_test = xy_test.reshape(1, -1)
        self.x_text_data = torch.Tensor(xy_test[:, :-1]).to(device)
        self.y_text_data = torch.Tensor(xy_test[:, [-1]]).to(device)

    # 这个地方是batch_size用的，它里面会使用使用那个getitem 这个东西去拿到它一个batch的数据
    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


"""
这个地方做了一下改进，我们无需关注数据的维度，
只需关注数据本身就可以了
"""


class DiabetesModule(torch.nn.Module):
    def __init__(self, dataSet):
        super().__init__()
        self.linear1 = torch.nn.Linear(dataSet.XDimension, dataSet.YDimension)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.sigmoid(self.linear1(inputs))
        return inputs


"""
这个dataloader我的理解就是运用dataset 这个类去构造自己的data，区别于dataset就是他有如何运行的参数，
比如gpu cpu核心等等的数据
"""


def init_varible(device):
    dataSet = DiabetesDataSet("diabetes.csv.gz", "diabetesTestData.csv.gz", device)
    dataloader = DataLoader(dataset=dataSet,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)
    model = DiabetesModule(dataSet).to(device)
    criterion = torch.nn.BCELoss(reduction="mean")
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    return (dataSet, dataloader, model, criterion, optimiser)


if __name__ == '__main__':
    time_totle = []
    for device in devices:
        start = time.time()
        dataSet, dataloader, model, criterion, optimiser = init_varible(device)
        for epoch in range(100):
            for i, data in enumerate(dataloader, 0):
                x, y = data  # 这个时候一个x是一个batch-size的东西，
                y_hat = model(x)
                # 这个loss必须算出来，因为后面的backward必须用这个
                loss = criterion(y_hat, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        end = time.time()
        print(device, " 时间: ", end - start)
        y_test_hat = model(dataSet.x_text_data)
        print("y_test_hat: ", y_test_hat, "y_test_data: ", dataSet.y_text_data, "\n")
