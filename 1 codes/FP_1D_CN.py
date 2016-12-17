import numpy as np
import matplotlib.pyplot as plt
from FP1D_functions import pvt

# import pdb
# pdb.set_trace()

# 1D Fokker-Planck Equation using Crank-Nicolson Method

# Constants for the Bounded Accumulation Model
D = 1.0         # the variance of the momentary evidence (Diffusion coefficient)
k = 0.255       # +- 0.002 (fit parameter)
B = 39.4       # 39.4 +- 10.0  (fit parameter)
theta = 0.591   # +- 0.005 (fit parameter)
c = 0.512       # motion strength (0 to 0.512 = 0% - 51.2 %)
mu = k * c      # the strength of momentary evidence

# Specify the grid in x and t
L = 2 * B       # length (L)
Nx = 2000       # number of grid points in x
dx = L / (Nx - 1)  # grid size
x_grid = np.array([j * dx for j in range(-int(Nx/2), int(Nx/2)+1)])
print('dx is %s and peak value should be %s' % (dx, 1/dx))

# Specify the Initial Condition
midpt = int(Nx/2)
U = np.array([0.0 for i in range(-int(Nx/2), int(Nx/2)+1)])  # + [0.0 for i in range(0, midpt)]
U[0] = 0.0    # Boundary condition for all time (t)
U[-1] = 0.0   # Boundary condition for all time (t)
U[midpt] = 1/dx  # initial condition (v, t0)
# pdb.set_trace()

# plot initial condition
plt.subplot(2, 1, 1)
# plt.ylim((0., 2.1))
plt.xlabel('v')
plt.ylabel('$p(v,t)$')
plt.plot(x_grid, U, label='$p(v,t_0)$')
plt.legend(loc='upper left')
plt.grid()
# plt.show()

print('t1 start')
U1, U_record1, t_grid1, tf1 = pvt(0.05, dx, Nx+1, U, D, mu)  # 50 msec

# U2, U_record2, t_grid2, tf2 = pvt(0.1, dx, Nx+1, U, D, mu)   # 100 msec
print('t3 start')
U3, U_record3, t_grid3, tf3 = pvt(0.2, dx, Nx+1, U, D, mu)   # 200 msec
print('tf start')
tf = 0.9
U, U_record, t_grid, tf = pvt(tf, dx, Nx+1, U, D, mu)

# Plot the numerical solution
plt.subplot(2, 1, 2)
plt.plot(x_grid, U1, label='$p(v, t=%s)$' % tf1)
# plt.plot(x_grid, U2, label='$p(v, t=%s)$' % tf2)
plt.plot(x_grid, U3, label='$p(v, t=%s)$' % tf3)

plt.plot(x_grid, U, label='$p(v, t=%s)$' % tf)
# plt.ylim((0., 2.1))
plt.xlabel('$v$')
plt.ylabel('Log-scale $p(v,t)$')
plt.yscale('log')
plt.legend(loc='upper left')
plt.grid()
# plt.show()

print('2D plotting start')
# Plot the 2D numerical solution
U_record = np.array(U_record)
fig, ax = plt.subplots()
plt.xlabel('v(t)')
plt.ylabel('time $(sec)$')
from matplotlib.colors import LogNorm
twodmap = ax.pcolor(x_grid, t_grid, U_record, norm=LogNorm())  # vmin=0., vmax=1.2 (vmin=U_record.min(),
# vmax=U_record.max())
colorbar = plt.colorbar(twodmap)
colorbar.set_label('$U = p(v,t)$')
plt.show()
