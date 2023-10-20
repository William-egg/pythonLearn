"""
MSE:cost =  1/nΣ(y_hat - y)²
"""
"""
gradient Descent algorithm
其实就是求导就可以了
公式：  w = w - α * （cost对w求导） = w - α*1/n*Σ2*xn(xn*w-yn)   其中xn*w = y_hat
"""
"""
鞍点： 就是梯度为0的点，或者是多维里面某些维度是最小值，某些维度是最大值，比如马鞍最低点
随机梯度算法：
    就是把对cost求导变成了对loss求导，这样的好处是有可能绕过按鞍点
    此时： w = w - a*（loss对w得导数）
         （loss对w得导数） = 2*xn（xn*w-yn）
    记住  这里是直接把xn拿出来就可以用了，而之前得要把所有的xn都加起来然后再除以n，对每一个样本都更新他的
    batch梯度下降其实就是把所有的训练集都用一个w去计算，而随机梯度算法是每一次都用一个训练样本去计算，所以前者训练集的所有样本都用一个计算，而后者是
    每一个样本所计算的w都不一样
    所以我们就可以选择一个折中的办法： mini-batch：批量的随机梯度下降
    
    
"""
"""
算法选择：
1. 穷举
2. 分治：  在多维的取值的时候每一维取几个点，在测完这些点之后取这些点最小值的附件区域，但是这种是有问题的，因为它的出来的是局部最优
3. 贪心： 在深度学习中，其实并没有过多的局部最优点，所以这种在神经网络中还是经常用到的算法
"""
import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# initial guess of weight
w = 1.0
# define the model linear model y = w*x
def forward(x):
    return x * w
# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

# define the gradient function  gd 这就是求导数的函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)
def gradient_random(x,y):#这个是随机梯度算法的求导数
    return 2*x(x*w-y)
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):#执行一百轮
    cost_val = cost(x_data, y_data)#这个求出来的应该是MSE
    grad_val = gradient(x_data, y_data)# 求出来导数
    w -= 0.01 * grad_val  # 0.01 learning rate 就是那个α，这个尽量设小一点，如果万一结果不收敛，可以看一下是不是这个的值设的不合理
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

"""
加权均值：
    就是cost函数波动得比较大，比较不光滑，这样子得花就可以使用加权均值
    即： C_0 = C0
        C_i = βCi+(1-β)c_(i-1)
"""
