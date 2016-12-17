import numpy as np
import matplotlib.pyplot as plt

import pdb
# pdb.set_trace()

# ===================================================================
# Explicit Method for 2D heat equation

# ======================== Global Constants =========================
D = 4.25e-6         # Diffusion Coefficient (thermal diffusivity)

# Specify the grid in x and t
x0 = 0
L = 0.01       # length (L) in meter
Nx = 5         # number of grid points in x
dx = L / Nx    # grid size

Ny = Nx        # number of grid points in y
dy = dx        #

x_grid = np.linspace(0, L, Nx)
y_grid = np.linspace(0, L, Ny)

# x_grid = np.array([j * dx for j in range(Nx)])  # x-grid
# y_grid = np.array([j * dx for j in range(Nx)])  # y-grid

dx0 = (x_grid[-1] - x_grid[0]) / float(len(x_grid) - 1)
dt0 = (y_grid[-1] - y_grid[0]) / float(len(y_grid) - 1)
X = np.linspace(x_grid[0] - 0.5 * dx0, x_grid[-1] + 0.5 * dx0, len(x_grid) + 1)
Y = np.linspace(y_grid[0] - 0.5 * dt0, y_grid[-1] + 0.5 * dt0, len(y_grid) + 1)

tf = 1                      # sec
Nt = 500002                 # number of grid points in time
dt = tf / (Nt-1)            # time-step (= h)
t_grid = np.array([n * dt for n in range(Nt)])

sigma = (D * dt) / (2. * dx * dx)

# ===================================================================
# Specify the Initial Condition
Tlo = 0.0
Tmid = 20.0
Thigh = 50.0

T0 = np.ones((Nx, Ny))
T0 *= Tmid  # set all position to be Tmid

# Initial & Boundary Conditions
T0[:, -1] = Thigh           # top
T0[-1, :] = Thigh           # right
T0[0, :] = Tlo              # left
T0[:, 0] = Tlo              # bottom

# Plot the initial condition (2D)
fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, T0.T, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(cax)
cbar.set_label('Temp (C)')
plt.xlim(x_grid[0], x_grid[-1])
plt.ylim(y_grid[0], y_grid[-1])

ax.set_title('Initial Condition')
plt.xlabel('x (Nx = %s)' % Nx)
plt.ylabel('y (Ny = %s)' % Ny)
plt.grid()
# plt.show()

# ========== Check the stability condition ==========
r = dt / (dx * dx)
if r >= 0.5:
    print('r is %s, which is too large.' % r)
    Nt_new = 2 * tf / (dx * dx) + 1
    print('Nt needs to be at least %s' % Nt_new)
    dt = tf / (Nt_new - 1)  # new time-step (= h)
    r = dt / (dx * dx)
    print('new r is %s' % r)
if r <= 0.5:
    print('r is %s, which is fine' % r)

# ===================================================================
# Solve the System Iteratively (Explicit Method)
U = T0
U_new = np.zeros((Nx, Ny))

for ti in range(1, Nt):
    for m in range(1, Nx-1):
        for l in range(1, Ny-1):
            U_new[m][l] = U[m][l] + r * (U[m+1][l] + U[m-1][l] + U[m][l+1] + U[m][l-1] - 4 * U[m][l])

    U = U_new

    # Initial & Boundary Conditions
    U[:, -1] = Thigh       # top
    U[-1, :] = Thigh       # right
    U[0, :] = Tlo          # left
    U[:, 0] = Tlo          # bottom


# ========== Plot the numerical solution (at t = tf) ==========
fig, ax = plt.subplots()
cax = ax.pcolormesh(X, Y, U.T, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(cax)
cbar.set_label('Temp (C)')
plt.xlim(x_grid[0], x_grid[-1])
plt.ylim(y_grid[0], y_grid[-1])

ax.set_title('2D Heat Eqn. with Explicit Method (t = %s sec, Nt = %s)' % (tf, Nt))
plt.xlabel('x (Nx = %s)' % Nx)
plt.ylabel('y (Ny = %s)' % Ny)
plt.grid()
plt.show()


