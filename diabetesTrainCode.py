import numpy as np
import torch
import time
# 这里必须是float32是因为绝大多数显卡只支持这个类型的
deviceGpu = torch.device("cuda:0")
deviceCpu = torch.device("cpu")
xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
xy_text = np.loadtxt("diabetesTestData.csv.gz", delimiter=",", dtype=np.float32)
xy_text = xy_text.reshape(1, -1)



class DiabetesTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

time_total = []
for i in range(2):
    stat = time.time()
    if i == 0:
        x_data = torch.Tensor(xy[:, :-1]).to(deviceGpu)
        y_data = torch.Tensor(xy[:, [-1]]).to(deviceGpu)
        x_test_data = torch.Tensor(xy_text[:, :-1]).to(deviceGpu)
        y_test_data = torch.Tensor(xy_text[:, [-1]]).to(deviceGpu)
        model = DiabetesTrainModel().to(deviceGpu)
        criterion = torch.nn.BCELoss(reduction="mean")
        optimiser = torch.optim.SGD(model.parameters(), lr=0.3)
    else:
        x_data = torch.Tensor(xy[:, :-1]).to(deviceCpu)
        y_data = torch.Tensor(xy[:, [-1]]).to(deviceCpu)
        x_test_data = torch.Tensor(xy_text[:, :-1]).to(deviceCpu)
        y_test_data = torch.Tensor(xy_text[:, [-1]]).to(deviceCpu)
        model = DiabetesTrainModel().to(deviceCpu)
        criterion = torch.nn.BCELoss(reduction="mean")
        optimiser = torch.optim.SGD(model.parameters(), lr=0.3)
    for epoch in range(10000):
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    end = time.time()
    time_total.append(end - stat)

print("gpu时间： ", time_total[0])
print("cpu时间： ", time_total[1])
y_test_hat = model(x_test_data)
print("y_test_hat", y_test_hat)
print("reallyValue", y_test_data)
# print(y_test_hat)