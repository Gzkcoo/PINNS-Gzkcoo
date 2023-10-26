import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer
from torch.nn.parameter import Parameter
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, layer_sizes):
        super(Net, self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
       
    def forward(self, x):
        X = x
        H = torch.tanh(self.Wz[0](X))
        for linear in self.Wz[1:-1]:
            H = torch.tanh(linear(H))
        H = self.Wz[-1](H)
        return H


class DeepONet(nn.Module):
    def __init__(self, b_dim, t_dim, layer_sizes_b, layer_sizes_t):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim
        self.layer_sizes_b = layer_sizes_b
        self.layer_sizes_t = layer_sizes_t        
        self.branch = Net(self.layer_sizes_b)        
        self.trunk = Net(self.layer_sizes_t)        
        self.b = Parameter(torch.zeros(1))

        
    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)        
        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(-1) + self.b
        return res


def error_2Dpicture():

    out_test = model(torch.Tensor(u_test).to(device), torch.Tensor(loc_test).to(device)).detach().cpu().numpy()

    xpoints = Ma_test_sol[0, :, 0]
    ypoints = Ma_test_sol[0, :, 1]
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(xpoints, ypoints, c=out_test[0:xpoints.shape[0], 0], s=0.5, cmap='OrRd', label='pred density', vmin=0,
                vmax=1)
    plt.colorbar()
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.scatter(xpoints, ypoints, c=y_test[0:xpoints.shape[0], 0], s=0.5, cmap='OrRd', label='true density', vmin=0,
                vmax=1)
    plt.colorbar()
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.scatter(xpoints, ypoints, c=np.abs(y_test[0:xpoints.shape[0], 0] - out_test[0:xpoints.shape[0], 0]) * 16, s=0.5,
                cmap='OrRd', label='error')
    plt.colorbar()
    plt.legend()
    plt.show()


#读取数据
data_sol = np.load('Ma_low_sol.npy')  # (61, 13879, 3)

Ma_full = np.linspace(4.,7.,61)
Ma_train = np.hstack((Ma_full[0:19],Ma_full[20:61]))
Ma_train_sol = np.vstack((data_sol[0:19,:,:],data_sol[20:61,:,:]))  # (60, 13879, 3)
Ma_test_sol = data_sol[19:20,:,:]

N_ma = Ma_train_sol.shape[0]
N_loc = Ma_train_sol.shape[1]
N_rand = 5000

#准备训练数据
u_train = np.zeros((N_ma*N_rand,1))
loc_train = np.zeros((N_ma*N_rand,2))
y_train = np.zeros((N_ma*N_rand,1))
for i in range(N_ma):
    a = Ma_train_sol[i]
    N_c = random.sample(range(a.shape[0]),N_rand)
    for j in range(N_rand):
        k = i*N_rand + j
        u_train[k,0] = Ma_train[i]
        loc_train[k,:] = a[j,0:2]
        y_train[k,:] = a[j,2:3]

u_train = torch.Tensor(u_train).to(device)
loc_train = torch.Tensor(loc_train).to(device)
y_train = torch.Tensor(y_train).to(device)

Ma_test = Ma_full[19:20].reshape(1, 1)
N_ma = Ma_test_sol.shape[0]
N_loc = Ma_test_sol.shape[1]
u_test = np.zeros((N_ma * N_loc, 1))
loc_test = np.zeros((N_ma * N_loc, 2))
y_test = np.zeros((N_ma * N_loc, 1))
for i in range(N_ma):
    for j in range(N_loc):
        k = i * N_loc + j
        u_test[k, 0] = Ma_test[i]
        loc_test[k, :] = Ma_test_sol[i, j, 0:2]
        y_test[k, :] = Ma_test_sol[i, j, 2:3]

#设置模型参数(训练时自行调整)
b_size = [1,256,256,256,256]
t_size = [2,256,256,256,256]
learning_rate = 1e-3
epochs = 50
step_size = 8  # 每隔step_size个epoch时调整学习率为当前学习率乘gamma
gamma = 0.8
batch_size = 128
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_train, loc_train, y_train), batch_size=batch_size, shuffle=True)

#初始化模型及训练
model = DeepONet(1, 2, b_size, t_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_history_mse = []
loss_func = nn.MSELoss()

###补充训练过程######
###补充训练过程######
###补充训练过程######

for i in range(epochs):
    model.train()
    for _, (u, location, y) in enumerate(train_loader):
        output = model(u, location)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        out_batch = output.detach().cpu().numpy()
        y_batch = y.detach().cpu().numpy()
        l2_rel = np.linalg.norm(out_batch[:, 0] - y_batch[:, 0]) / np.linalg.norm(y_batch[:, 0])
        print('\repoch {:d} PINN mse_error = {:.6f} l2_relative_error = {:6f} learning_rate = {:.6f}'.format(i + 1, loss.item(), l2_rel,
                                                                                                     optimizer.param_groups[0]['lr']), end='', flush=True)
    scheduler.step()
    loss_history_mse.append(loss)

    if (i+1) % 5 == 0:
        error_2Dpicture()


#测试
Ma_test = Ma_full[19:20].reshape(1,1)
N_ma = Ma_test_sol.shape[0]
N_loc = Ma_test_sol.shape[1]
u_test = np.zeros((N_ma*N_loc,1))
loc_test = np.zeros((N_ma*N_loc,2))
y_test = np.zeros((N_ma*N_loc,1))
for i in range(N_ma):
    for j in range(N_loc):
        k = i*N_loc + j
        u_test[k,0] = Ma_test[i]
        loc_test[k,:] = Ma_test_sol[i,j,0:2]
        y_test[k,:] = Ma_test_sol[i,j,2:3]
out_test = model(torch.Tensor(u_test).to(device),torch.Tensor(loc_test).to(device)).detach().cpu().numpy()

#计算预测的误差
l2_rel = np.linalg.norm(out_test[:,0]-y_test[:,0]) / np.linalg.norm(y_test[:,0]) 
mse_test = 1/y_test.shape[0]*(np.linalg.norm(out_test[:,0]-y_test[:,0])**2)
print('The l2 relative error is ',l2_rel, '. The mse error is ', mse_test)

xpoints = Ma_test_sol[0,:,0] 
ypoints = Ma_test_sol[0,:,1]
fig = plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(xpoints, ypoints, c=out_test[0:xpoints.shape[0],0], s=0.5, cmap='OrRd', label='pred density',vmin=0,vmax=1)
plt.colorbar()
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(xpoints, ypoints, c=y_test[0:xpoints.shape[0],0], s=0.5, cmap='OrRd', label='true density',vmin=0,vmax=1)
plt.colorbar()
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(xpoints, ypoints, c=np.abs(y_test[0:xpoints.shape[0],0]-out_test[0:xpoints.shape[0],0])*16, s=0.5, cmap='OrRd', label='error')
plt.colorbar()
plt.legend()
plt.show()
