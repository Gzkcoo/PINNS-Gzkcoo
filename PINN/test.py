import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch

if __name__ == '__main__':

    a_1 = 1
    a_2 = 4


    def u(x, a_1, a_2):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    def u_xx(x, a_1, a_2):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    def u_yy(x, a_1, a_2):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    # Forcing
    def f(x, a_1, a_2, lam):
        return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)


    def exact(x, y):
        return torch.sin(2 * 2 * x) * torch.sin(2 * 2 * y)


    # Parameter
    lam = 1.0

    # Domain boundaries
    bc1_coords = np.array([[-1.0, -1.0],
                           [1.0, -1.0]])
    bc2_coords = np.array([[1.0, -1.0],
                           [1.0, 1.0]])
    bc3_coords = np.array([[1.0, 1.0],
                           [-1.0, 1.0]])
    bc4_coords = np.array([[-1.0, 1.0],
                           [-1.0, -1.0]])

    dom_coords = np.array([[-1.0, -1.0],
                           [1.0, 1.0]])

    # Create initial conditions samplers
    ics_sampler = None



    # Test data
    nn = 100
    x1 = np.linspace(0, 1, nn)[:, None]
    x2 = np.linspace(0, 1, nn)[:, None]
    x1, x2 = np.meshgrid(x1, x2)
    X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

    # xc = torch.linspace(0, 1, 100)
    # xm, ym = torch.meshgrid(xc, xc)
    # xx = xm.reshape(-1, 1)
    # yy = ym.reshape(-1, 1)
    #
    #
    # xy_data = torch.cat([xx, yy], dim=1).requires_grad_(True)
    # u_star = exact(xx, yy)
    #
    # xx = xx.detach().numpy()
    # yy = yy.detach().numpy()
    #
    # xy = xy_data.detach().numpy()
    # u_star = u_star.detach().numpy()
    # xy_data = xy_data.detach().numpy()
    # U_star = griddata(xy_data, u_star.flatten(), (xx, yy), method='cubic')

    # Exact solution

    u_star = u(X_star, a_1, a_2)
    # Exact solution & Predicted solution
    # Exact soluton
    U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')




    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Exact $u(x)$')
    plt.show()




