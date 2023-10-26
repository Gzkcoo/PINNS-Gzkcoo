import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import pi
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()

        self.U_layer = nn.Linear(2, NN)
        self.V_layer = nn.Linear(2, NN)
        self.H_layer1 = nn.Linear(2, NN)
        self.Z_layer1 = nn.Linear(NN, NN)
        self.Z_layer2 = nn.Linear(NN, NN)
        self.Z_layer3 = nn.Linear(NN, NN)
        self.output_layer = nn.Linear(NN, 1)

        nn.init.xavier_normal_(self.U_layer.weight, gain=1)
        nn.init.constant_(self.U_layer.bias, 0.)
        nn.init.xavier_normal_(self.V_layer.weight, gain=1)
        nn.init.constant_(self.V_layer.bias, 0.)
        nn.init.xavier_normal_(self.H_layer1.weight, gain=1)
        nn.init.constant_(self.H_layer1.bias, 0.)
        nn.init.xavier_normal_(self.Z_layer1.weight, gain=1)
        nn.init.constant_(self.Z_layer1.bias, 0.)
        nn.init.xavier_normal_(self.Z_layer2.weight, gain=1)
        nn.init.constant_(self.Z_layer2.bias, 0.)
        nn.init.xavier_normal_(self.Z_layer3.weight, gain=1)
        nn.init.constant_(self.Z_layer3.bias, 0.)
        nn.init.xavier_normal_(self.output_layer.weight, gain=1)
        nn.init.constant_(self.output_layer.bias, 0.)


    def forward(self, x):
        U = torch.tanh(self.U_layer(x))
        V = torch.tanh(self.V_layer(x))
        H1 = torch.tanh(self.H_layer1(x))
        Z1 = torch.tanh(self.Z_layer1(H1))
        H2 = (1 - Z1) * U + Z1 * V
        Z2 = torch.tanh(self.Z_layer2(H2))
        H3 = (1 - Z2) * U + Z2 * V
        Z3 = torch.tanh(self.Z_layer3(H3))
        H3 = (1 - Z3) * U + Z3 * V
        out_final = self.output_layer(H3)
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

    return (-u_xx - u_yy - f(x, y))

# 边界误差
def loss_bc(net, bc_num):
    x_bc = torch.unsqueeze(torch.linspace(0, 1, bc_num), dim=1)
    y_bc = torch.unsqueeze(torch.linspace(0, 1, bc_num), dim=1)
    zeros = torch.unsqueeze(torch.zeros(bc_num), dim=1)
    ones = torch.unsqueeze(torch.ones(bc_num), dim=1)
    res = net(torch.cat([x_bc, zeros], 1)).pow(2).mean() + net(torch.cat([zeros, y_bc], 1)).pow(2).mean() + \
          net(torch.cat([x_bc, ones], 1)).pow(2).mean() + net(torch.cat([ones, y_bc], 1)).pow(2).mean()
    return res


def resample(net , m, res_num):

    # 生成更精细网格点坐标
    XC = torch.linspace(0, 1, res_num * 2)
    x, y = torch.meshgrid(XC, XC)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xy_data = torch.cat([x, y], dim=1).requires_grad_(True)

    # 找到res最大的m个点
    f_res = loss_re(net, xy_data).detach().numpy().squeeze()
    max_m_index = f_res.argsort()[-m:]
    max_m = xy_data[max_m_index]

    xy_data = torch.cat([xy_data, max_m], 0)

    xx = xy_data[:, 0].unsqueeze(-1)
    yy = xy_data[:, 1].unsqueeze(-1)

    train_data_loader = DataLoader(dataset=torch.utils.data.TensorDataset(xx, yy), batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)
    return max_m, train_data_loader



def error_fig2D(net):
    # 生成网格点坐标
    xy = torch.linspace(0, 1, 200)
    X, Y = torch.meshgrid(xy, xy)
    XX = X.reshape(-1, 1).requires_grad_(True)
    YY = Y.reshape(-1, 1).requires_grad_(True)
    XY_data = torch.cat([XX, YY], dim=1)

    exact_solution = exact(XX, YY)
    pred_solution = net(XY_data)
    X = X.detach().numpy()
    Y = Y.detach().numpy()
    exact_solution = exact_solution.detach().numpy()
    pred_solution = pred_solution.detach().numpy()
    XY_data = XY_data.detach().numpy()

    Exact_solution = griddata(XY_data, exact_solution.flatten(), (X, Y), method='cubic')
    Pred_solution = griddata(XY_data, pred_solution.flatten(), (X, Y), method='cubic')

    fig = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(X, Y, Exact_solution, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Exact $u(x)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(X, Y, Pred_solution, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Predicted $u(x)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(X, Y, np.abs(Exact_solution - Pred_solution), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()


epochs = 50
learning_rate = 1e-3
res_num = 100   # 100*100 残差取点个数
bc_num = 100    # 边界取点
lam = 1.0  # 边界损失权重
batch_size = 128
beta = 0.9  # 权重系数
m = 100  # 每次重采样数量

# 生成网格点坐标
xc = torch.linspace(0, 1, res_num)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
# xy_data = torch.cat([xx, yy], dim=1).requires_grad_(True)

# 随机梯度下降
train_data_loader = DataLoader(dataset=torch.utils.data.TensorDataset(xx, yy), batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)


net = Net(50)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


loss_history = []
loss_bc_history = []
loss_res_history = []

for i in range(epochs):
    net.train()
    for _, (x, y) in enumerate(train_data_loader):
        xy_data = torch.cat([x, y], dim=1).requires_grad_(True)
        loss_res = loss_re(net, xy_data).pow(2).mean()
        loss_BC = loss_bc(net, bc_num)

        # 动态更新lam权重
        grad_res = []
        grad_bc = []
        count = 0
        for k, v in net.named_parameters():
            count += 1
            # 过滤biases
            if count % 2 == 0:
                break
            grad_res_i = torch.autograd.grad(loss_res, v, retain_graph=True, allow_unused=True)[0].reshape(1, -1).squeeze()  # 求偏导数
            grad_bc_i = torch.autograd.grad(loss_BC, v, retain_graph=True, allow_unused=True)[0].reshape(1, -1).squeeze()  # 求偏导数
            grad_res.append(grad_res_i)
            grad_bc.append(grad_bc_i)

        grad_res_max_list = []
        grad_bc_mean_list = []
        for list in grad_res:
            grad_res_max_list.append(torch.mean(torch.abs(list)))

        for list in grad_bc:
            grad_bc_mean_list.append(torch.mean(torch.abs(list)))

        grad_res_max = torch.mean(torch.stack(grad_res_max_list))
        grad_bc_mean = torch.mean(torch.stack(grad_bc_mean_list))

        adaptive_lam = grad_res_max / grad_bc_mean

        lam = (1 - beta) * lam + beta * adaptive_lam
        # lam = 1
        loss = loss_res + lam * loss_BC
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print('\repoch {:d} PINN loss = {:.6f} lam = {:.6f} grad_res_max = {:.6f} grad_bc_mean = {:.6f}'.format(i + 1, loss.item(), lam, grad_res_max, grad_bc_mean),
                end='', flush=True)

    loss_history.append(loss.item())
    loss_bc_history.append(loss_BC.item())
    loss_res_history.append(loss_res.item())
    if (i+1) % 5 == 0:
        net.eval()

        # RAR 重新采样
        resamples, train_data_loader = resample(net, m, res_num)
        resamples = resamples.detach().numpy()
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        plt.xlabel('epochs')
        plt.ylabel('history')
        plt.scatter(resamples[:, 0], resamples[:, 1], marker='x')
        plt.title('Resamples')
        plt.legend()
        plt.show()


        # 二维图展现误差
        error_fig2D(net)



# 各部分损失值变化情况
fig_1 = plt.figure(1)
plt.subplot(1, 1, 1)
plt.plot(loss_history, 'r', label='loss')
plt.plot(loss_res_history, 'b', label='loss_res')
plt.plot(loss_bc_history, 'g', label='loss_bc')
plt.xlabel('epochs')
plt.ylabel('history')
plt.legend()
plt.show()

