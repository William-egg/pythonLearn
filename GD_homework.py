# y = 2*x+3
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 7.0, 9.0]

# initial value
w = 1.0
b = 1.0
a = 0.01


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

def forward(x):
    return w * x + b


# 俩个参数跟我们一开始的数据是一样的，但是为了后面好封装，直接传进来
def cost(xs, ys):
    cost_temp = 0
    for x, y in zip(xs, ys):
        cost_temp += (forward(x) - y) ** 2
    return cost_temp / (2 * len(xs))


def gradient_w(xs, ys):
    grad_temp = 0
    for x, y in zip(xs, ys):
        grad_temp += x * (forward(x) - y)
    return grad_temp / len(xs)


def gradient_b(xs, ys):
    grad_temp = 0
    for x, y in zip(xs, ys):
        grad_temp += (forward(x) - y)
    return grad_temp / len(xs)


def gradient(which_one, xs, ys):
    if which_one == 'w':
        return gradient_w(xs, ys)
    else:
        return gradient_b(xs, ys)


ws = []
bs = []
costs = []
for epoch in range(1000):
    cost_val = cost(x_data, y_data)
    ws.append(w)
    bs.append(b)
    costs.append(cost_val)
    grad_val_w = gradient('w', x_data, y_data)
    grad_val_b = gradient('b', x_data, y_data)
    w -= a * grad_val_w
    b -= a * grad_val_b
    print('epoch:', epoch, ' w=', w, 'b=', b, 'mse_loss:', cost_val)
print("predict y(4)(after training)", forward(4))

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ws, bs, costs)
ax.set_xlabel("W轴")
ax.set_ylabel("b值")
ax.set_zlabel("mse值")
plt.show()
