"""
1. dataSet 这个data一般来说都有俩个部分 ，training set 另一个是test set
如果没有训练集，那么就有可能出现过拟合，我们需要它具有泛化能力，那么我们可以分开training set ，一部分作为开发集
其实就是之前的test set
2. Model
3. Training
4. inferring
"""
"""
线性模型中： y_hat = x*w+b but now lets look at y_hat = x*w
1. the machine starts with a random guess , w = random value
2. training Loss(Error) loss = (y_hat - y)²  and then calculate Mean Square Error(MSE) cost = 1/nΣ(y_hat - y)²

"""
import numpy as np #这个是一个用于科学计算和数据分析的python库
import matplotlib.pyplot as plt #这个是画图用的
"""
x_data 和 y_data是training Set里面的输入以及输出
"""
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
#forward 函数返回 y_hat 即 y_hat = x*w
def forward(x):
    return x * w
#计算出loss 至于等一下的MSE是在for循环中直接计算的
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 穷举法
w_list = []
mse_list = []
"""
1. 从0到4.0 步长为1的穷举，跟那个arrange 也差不多
2. for 后面w是全局变量来的。 找了好久的答案气死了，基本功还是不扎实
3. y_pred_val 就是加了权重之后的预测值，而y_val是数据集提供的值
"""
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')# y坐标
plt.xlabel('w') # x 坐标
plt.show()