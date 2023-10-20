import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]
w = 0
b = 0
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
"""
how to use pyplot:
    
"""

def forward(x):
    return w*x+b


def loss(x, y):
    y_hat = forward(x)
    return (y_hat-y)**2


w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(0.0, 4.1, 0.1)
W, B = np.meshgrid(w_list, b_list)
mse_loss_function_list = []
z_max = 0
for a, c in zip(W, B):
    temp_loss_list = []
    for w, b in zip(a, c):
        l_sum = 0
        for x, y in zip(x_data,y_data):
            loss_val = loss(x, y)
            l_sum += loss_val
        temp_loss_list.append(l_sum/3.0)
        if z_max < l_sum/3.0:
            z_max = l_sum
    mse_loss_function_list.append(temp_loss_list)
    temp_loss_list = []
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
# ax.set
mse_loss_function_list = np.array(mse_loss_function_list)
"""
the difference between np.array and np.arang and the different between a normal array and them
    the index that np.array need is a list while no.arange can just have an integer which will create a list from zero to
    the integer and both of them is ndarray which is needed by the plot 
    in other word, if you have a normal array you have to use np.array to change it into ndarray in order to fix the need 
    of pyplot
    but the normal array is nothing but a place where data is stored
    
    二个问题就是在把数据丢入到模型中去展示我的图片的时候，他老是报一个它在我其中一个变量中没有找到它所需的一个属性，最后找了很多资料，我猛然反应过来
"""
# ax.sc
ax.set_zlim(-3, z_max)
ax.set_xlabel("W轴")
ax.set_ylabel("Y值")
ax.set_zlabel("mse值")
plt.show()