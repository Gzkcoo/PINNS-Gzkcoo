import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from numpy import pi
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN, int(NN / 2))
        self.hidden_layer2 = nn.Linear(int(NN / 2), int(NN / 2))
        self.hidden_layer3 = nn.Linear(int(NN / 2), int(NN / 2))
        self.output_layer = nn.Linear(int(NN/2), 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out = torch.tanh(self.hidden_layer3(out))
        out_final = self.output_layer(out)
        return out_final

def exact(x, y):
    return torch.sin(2 * pi * x) * torch.sin(2 * pi * y)

def f(x, y):
    z = 8 * pi * pi * torch.sin(2 * pi * x) * torch.sin(2 * pi * y)
    return z


# 偏微分方程残差
def loss_re(net, xy_data):
    # 边界条件

    u = net(xy_data)
    # 自动微分
    u_xy = torch.autograd.grad(u, xy_data, grad_outputs=torch.ones_like(u),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    u_x = u_xy[:, 0].unsqueeze(-1)
    u_y = u_xy[:, 1].unsqueeze(-1)
    u_xx = torch.autograd.grad(u_x, xy_data, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, allow_unused=True)[0][:, 0].unsqueeze(-1)  # 求偏导数
    u_yy = torch.autograd.grad(u_y, xy_data, grad_outputs=torch.ones_like(u_y),
                               create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

    x = xy_data[:, 0].unsqueeze(-1)
    y = xy_data[:, 1].unsqueeze(-1)

    return (-u_xx - u_yy - f(x, y)).pow(2).mean()

# 边界误差
def loss_bc(net, bc_num):
    x_bc = torch.unsqueeze(torch.linspace(0, 1, bc_num), dim=1)
    y_bc = torch.unsqueeze(torch.linspace(0, 1, bc_num), dim=1)
    zeros = torch.unsqueeze(torch.zeros(bc_num), dim=1)
    ones = torch.unsqueeze(torch.ones(bc_num), dim=1)
    res = net(torch.cat([x_bc, zeros], 1)).pow(2).mean() + net(torch.cat([zeros, y_bc], 1)).pow(2).mean() + \
          net(torch.cat([x_bc, ones], 1)).pow(2).mean() + net(torch.cat([ones, y_bc], 1)).pow(2).mean()
    return res


epochs = 100
learning_rate = 1e-3
res_num = 100   # 100*100 残差取点个数
bc_num = 100    # 边界取点
lam = 1  # 边界损失权重
batch_size = 128


# 生成网格点坐标
xc = torch.linspace(0, 1, res_num)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy_data = torch.cat([xx, yy], dim=1).requires_grad_(True)

# 随机梯度下降
train_data_loader = DataLoader(dataset=torch.utils.data.TensorDataset(xx, yy), batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)

net = Net(40)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# for i in range(epochs):
#     for _, (x, y) in enumerate(train_data_loader):
#         xy_data = torch.cat([xx, yy], dim=1).requires_grad_(True)
#         loss = loss_re(net, xy_data) + lam * loss_bc(net, bc_num)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print('\repoch {:d} PINN loss = {:.6f}'.format(i + 1, loss.item()),
#                 end='', flush=True)


net.load_state_dict(torch.load('net_1.params'))

# 模型参数保存
# torch.save(net.state_dict(),'net_2.params')

# 模型预测
exact_solution = exact(xx, yy)
pred_solution = net(xy_data)
xx = xx.detach().numpy()
yy = yy.detach().numpy()
exact_solution = exact_solution.detach().numpy()
pred_solution = pred_solution.detach().numpy()
xy_data = xy_data.detach().numpy()



fig_1 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.pcolor(xc, xc, exact_solution.reshape(100, 100), cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Exact $u(x)$')

plt.subplot(1, 3, 2)
plt.pcolor(xc, xc, pred_solution.reshape(100, 100), cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Predicted $u(x)$')

plt.subplot(1, 3, 3)
plt.pcolor(xc, xc, np.abs(exact_solution.reshape(100, 100) - pred_solution.reshape(100, 100)), cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Absolute error')
plt.tight_layout()
plt.show()



"""
1. 小批量梯度下降法+100权重：loss = 0.0303
2. 小批量梯度下降法+1权重: loss = 0.0722

"""