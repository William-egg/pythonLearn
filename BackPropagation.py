"""
what is back propagation:
    it is a process which is used to calculate the gradient,i.e, loss 对 w的偏导，在权重的维度比较复杂的情况下
"""
"""
what is tensor
    其实他就是一个类，里面有data和grad，grad是存储目前这一步函数下来的loss对w的偏导数的，拥有它我们可以进行比较复杂的数据处理
    ps: 张量作为一个概念十分的复杂，在这里作为一个类（我所理解的）就是一种数据结构，方便我们进行数据处理的
"""
import torch
# 阿松大萨达啊实打实
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.9])  # w的初值为1.0,如果这里面有多个参数的话，其实就是多个维度的意思
w.requires_grad = True  # 需要计算梯度

print((w.data))# torch.Tensor 其实它也是一个tensor
print(type(w.grad))# noneType 说明在backward之前这个变量里面是没有任何东西的

def forward(x):
    return x * w  # w是一个Tensor

# 在这里面的所有计算出来的变量都是tensor
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor

        w.grad.data.zero_()  # after update, remember set the grad to zero,if not it will calculate their value plus
    print(type(w.grad))#现在就是tensor了
    print('progress:', epoch+1, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, forward(4).item())