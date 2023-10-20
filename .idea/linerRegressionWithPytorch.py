"""
内部计算矩阵(矩阵乘法和批处理矩阵乘法)和卷积。
"""
"""
torch.randn
    这个函数是用于创建自己所需的矩阵的，比如下面这些代码是官网所演示的代码，
torch.nn.Linear
    参数里面的in_features指的是输入的维度，也就是一次输入的x变量有多少个，而out_features变量是输出的维度，
在下面这串代码中我们需要一个输入20，输出30的线性方程，使用randn创造一个99个20维度的x数据，输入到m中，出来一个99个的每个有三十个的output。
也就是有99个输出，每个输出有三十个维度
"""
# import torch
# m = torch.nn.Linear(20, 30)
# input = torch.randn(99,20)
# output = m(input)
# print(output.size())
import torch

# prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""
"""
我们自定义的linearModel必须继承torch.nn下面的module
nn,linear类下面包含了俩个tensors： weight 和 bias
linear继承了魔法函数，我们可以直接使用类名去使用这里面的魔法函数
"""
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):#此处必须叫做forward
        y_pred = self.linear(x)#此处直接使用了父级的linear实现，怎么实现的不重要，在后续的model()里面也是使用了module的魔法函数
        return y_pred


model = LinearModel()

# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#sgd优化器，要知道不管是什么梯度下降算法都是这个，只是传入的参数不同，但是这个model.parameters到底是个啥。我实在是没想明白

# training cycle forward, backward, update，这是我们一般的流程： 1. 训练 2. 前向传播 3. 反向传播 4. 更新
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0],[5.0],[6.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)