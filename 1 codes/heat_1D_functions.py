import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.special import erf

# import pdb
# pdb.set_trace()

# ==========================================================================================
# Exact solution
def f_exact(x, t, Thigh, Tlo):
    return 0.5 * Thigh * (1 - erf((400*x - 2) / np.sqrt(4 * t))) + Tlo


def temp(tf, Nx, U, Tlo, Thigh, D, dx):
    # tf = 0.1  # length of time (sec)
    Nt = 1000  # number of grid points in time
    dt = tf / (Nt-1)  # time-step
    t_grid = np.array([n * dt for n in range(Nt)])

    sigma = (D * dt) / (2. * dx * dx)

    # Create 3 x N matrix for A (for scipy.linalg.solve_banded)
    A = np.array([[0]+[-sigma for i in range(Nx - 1)],
                 [1. + sigma] + [1. + 2. * sigma for i in range(Nx - 2)] + [1. + sigma],
                 [-sigma for i in range(Nx - 1)]+[0]])

    # Create Tridiagonal Matrices
    c = sigma
    d = np.array([1. - sigma] + [1. - 2. * sigma for i in range(Nx - 2)] + [1. - sigma])
    e = sigma

    # Solve the System Iteratively
    U_record = []
    U_record.append(U)

    for ti in range(1, Nt):
        u1 = np.array([U[i] for i in range(1, Nx)]+[0])
        u2 = np.array([U[i] for i in range(Nx)])
        u3 = np.array([0]+[U[i] for i in range(Nx-1)])

        b = c * u1 + np.array([d[i] * u2[i] for i in range(Nx)]) + e * u3

        U_new = solve_banded((1, 1), A, b)

        U = U_new
        # U[0] = Thigh  # Boundary condition
        # U[-1] = Tlo   # Boundary condition
        U_record.append(U)
    return U, tf, t_grid, U_record


def two_d_plot(U_record, x_grid, t_grid):
    dx0 = (x_grid[-1] - x_grid[0]) / float(len(x_grid) - 1)
    dt0 = (t_grid[-1] - t_grid[0]) / float(len(t_grid) - 1)
    X = np.linspace(x_grid[0] - 0.5 * dx0, x_grid[-1] + 0.5 * dx0, len(x_grid)+1)
    T = np.linspace(t_grid[0] - 0.5 * dt0, t_grid[-1] + 0.5 * dt0, len(t_grid)+1)

    U_record = np.array(U_record)

    # Plot the numerical solution in 2D (x, time)
    fig, ax = plt.subplots()

    ax.set_title('1D Heat Equation: time vs position')
    plt.xlabel('time (sec)')
    plt.ylabel('x (m)')

    # twodmap = ax.pcolor(x_grid, t_grid, U_record)  # , vmin=0., vmax=55
    # colorbar = plt.colorbar(twodmap)
    # colorbar.set_label('Temp (C)')

    cax = ax.pcolormesh(T, X, U_record.T, cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(cax)
    cbar.set_label('Temp (C)')

    plt.xlim(t_grid[0], t_grid[-1])
    plt.ylim(x_grid[0], x_grid[-1])
    plt.grid()
    plt.show()

# ==========================================================================================