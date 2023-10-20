import torch
# Check if CUDA (GPU support) is available
device = torch.device("cuda:0")
x_data = torch.Tensor([[1.0], [2.0], [3.0]]).to(device)
y_data = torch.Tensor([[0], [0], [1]]).to(device)
# print(torch.cuda.is_available())


class MyLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # super(MyLinearModel, self).__int__()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = MyLinearModel().to(device)
criterion = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for epoch in range(10000):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)#注意这个loss是一个tensor
    print(epoch, ":  ", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w=: ", model.linear.weight.item())
print("b=: ", model.linear.bias.item())
print("y_tese:[x=4,x=5] ", model(torch.Tensor([[4.0], [5.0]]).to(device)))