import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

import pdb
# pdb.set_trace()

# ==========================================================================================
# Crank-Nicolson Method for 2D heat equation
# Alternating Direction Implicit (Peaceman-Rachford) method

# ======================== Global Constants =========================
D = 4.25e-6         # Diffusion Coefficient (thermal diffusivity)

# ========== Specify the grid in x and y ==========
L = 0.01       # length (L) in meter
Nx = 50         # number of grid points in x
dx = L / Nx    # grid size

Ny = Nx        # number of grid points in y
dy = dx        # same as dx

x_grid = np.linspace(0, L, Nx)
y_grid = np.linspace(0, L, Ny)
dx0 = (x_grid[-1] - x_grid[0]) / float(len(x_grid)-1)
dy0 = (y_grid[-1] - y_grid[0]) / float(len(y_grid)-1)
X = np.linspace(x_grid[0]-0.5*dx0, x_grid[-1]+0.5*dx0, len(x_grid)+1)
Y = np.linspace(y_grid[0]-0.5*dy0, y_grid[-1]+0.5*dy0, len(y_grid)+1)

# ========== Specify the grid in time ==========
tf = 1                          # sec
Nt = 100                        # number of grid points in time
dt = tf / (Nt-1)                # time-step (= h)
t_grid = np.array([n * dt for n in range(Nt)])

r = D * dt / (dx * dx)          # constant based on dt and dx

# ========== Specify the Initial Condition ==========
Tlo = 0.0                   # low temperature
Tmid = 20.0                 # mid temperature
Thigh = 50.0                # high temperature

T0 = np.ones((Nx, Ny))
T0 *= Tmid                  # set all position to be Tmid

# Initial & Boundary Conditions
T0[:, -1] = Thigh           # top
T0[-1, :] = Thigh           # right
T0[0, :] = Tlo              # left
T0[:, 0] = Tlo              # bottom

# Plot the initial condition (2D)
fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, T0.T, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(cax)
plt.xlim(x_grid[0], x_grid[-1])
plt.ylim(y_grid[0], y_grid[-1])
cbar.set_label('Temp (C)')
ax.set_title('Initial Condition')
plt.xlabel('x (Nx = %s)' % Nx)
plt.ylabel('y (Ny = %s)' % Ny)
plt.grid()
# plt.show()

# ==========================================================================================
# ==========================================================================================
# Create 3 x N matrix for A1 and A2(to use scipy.linalg.solve_banded)
A1 = np.array([[0]+[-0.5 * r for i in range(Nx - 1)],
               [1 + r for i in range(Nx)],
               [-0.5 * r for i in range(Nx - 1)]+[0]])

A2 = np.array([[0]+[-0.5 * r for i in range(Ny - 1)],
               [1 + r for i in range(Ny)],
               [-0.5 * r for i in range(Ny - 1)]+[0]])

g0l = np.array([Tlo for i in range(Nx)])    # bottom
g1l = np.array([Thigh for i in range(Nx)])  # top
g2l = np.array([Thigh for i in range(Ny)])  # left
g3l = np.array([Tlo for i in range(Ny)])    # right

b_star_x = np.array([g0l] + [np.zeros(Nx) for i in range(Nx - 2)] + [g1l])
b_star_y = np.array([g3l] + [np.zeros(Ny) for i in range(Ny - 2)] + [g2l])

# Solve the System Iteratively
U = T0
U_record = []
U_record.append(U)
print('initial U: ', U)

c1 = 0.5 * r; c2 = 1.0 - r; c3 = 0.5 * r  # constants

for ti in range(1, Nt):
    # For x (first iterative step in x (n -> n + 1/2))
    ux1 = np.array([U[i+1, :] for i in range(0, Nx-1)] + [np.zeros(Nx)])
    ux2 = np.array([U[i, :] for i in range(Nx)])
    ux3 = np.array([np.zeros(Nx)] + [U[i-1, :] for i in range(1, Nx)])
    b1 = c1 * ux1 + c2 * ux2 + c3 * ux3 + 0.5 * r * b_star_x
    U_star = solve_banded((1, 1), A1, b1)

    U = U_star.T

    # Boundary Conditions
    U[:, -1] = Thigh   # top
    U[-1, :] = Thigh   # right
    U[0, :] = Tlo      # left
    U[:, 0] = Tlo      # bottom

    # For y (second iterative step in y (n +1/2 -> n + 1))
    uy1 = np.array([U[:, i + 1] for i in range(0, Ny - 1)] + [np.zeros(Ny)])
    uy2 = np.array([U[:, i] for i in range(Ny)])
    uy3 = np.array([np.zeros(Ny)] + [U[:, i - 1] for i in range(1, Ny)])
    b2 = c1 * uy1 + c2 * uy2 + c3 * uy3 + 0.5 * r * b_star_y
    U_new = solve_banded((1, 1), A2, b2)

    # pdb.set_trace()
    U = U_new.T

    # Boundary Conditions
    U[:, -1] = Thigh  # top
    U[-1, :] = Thigh  # right
    U[0, :] = Tlo  # left
    U[:, 0] = Tlo  # bottom

    U_record.append(U)

print('final U: ', U)

# ========== Plot the final solution ==========
fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, U.T, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(cax)
cbar.set_label('Temp (C)')
plt.xlim(x_grid[0], x_grid[-1])
plt.ylim(y_grid[0], y_grid[-1])
ax.set_title('2D Heat Eqn. with CN (ADI) Method (t = %s sec, Nt = %s)' % (tf, Nt))
plt.xlabel('x (Nx = %s)' % Nx)
plt.ylabel('y (Ny = %s)' % Ny)
plt.grid()
plt.show()
# =============================================
