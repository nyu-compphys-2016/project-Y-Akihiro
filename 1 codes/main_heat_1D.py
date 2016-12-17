import numpy as np
import matplotlib.pyplot as plt
from heat_1D_functions import temp, two_d_plot

# import pdb
# pdb.set_trace()

# Constants
D = 4.25e-6  # the diffusion coefficient

# Specify the grid in x and t
L = 0.01  # length (L) in meter
Nx = 100  # number of grid points in x
dx = L / Nx  # grid size
x_grid = np.array([j * dx for j in range(Nx)])

# Specify the Initial Condition
Tlo = 0.0
Tmid = 20.0
Thigh = 50.0

midpt = int(Nx / 2)
U = np.array([0 for i in range(0, Nx)])
U[midpt] = Thigh  # Delta-function

# U = np.array([Thigh for i in range(0, midpt)] + [Tlo for i in range(midpt, Nx)])

# Calculate the numerical solutions at different time (t)
temp1, time1, tg1, Trec1 = temp(0.0001, Nx, U, Tlo, Thigh, D, dx)  # t1 = 0.01
temp2, time2, tg2, Trec2 = temp(0.001, Nx, U, Tlo, Thigh, D, dx)  # t2 = 0.1
temp3, time3, tg3, Trec3 = temp(0.01, Nx, U, Tlo, Thigh, D, dx)  # t3 = 0.4
temp4, time4, t_grid, Trec4 = temp(0.03, Nx, U, Tlo, Thigh, D, dx)  # t4 = 1.0
temp5, time5, tg5, Trec5 = temp(0.05, Nx, U, Tlo, Thigh, D, dx)  # t5 = 10.0

# Plot initial condition
plt.subplot(1, 1, 1)
plt.ylim((0., 50))
plt.plot(x_grid, U, label='T$_{initial}$')

# Plot the numerical solution
# plt.subplot(2, 1, 2)
# plt.ylim((0., 2.1))
plt.plot(x_grid, temp1, label='T at %s sec' % time1)
plt.plot(x_grid, temp2, label='T at %s sec' % time2)
plt.plot(x_grid, temp3, label='T at %s sec' % time3)
plt.plot(x_grid, temp4, label='T at %s sec' % time4)
plt.plot(x_grid, temp5, label='T at %s sec' % time5)

plt.xlabel('x (m)')
plt.ylabel('T (C)')
plt.ylim((-1., 51))
plt.legend(loc='upper right')
plt.grid()
# plt.show()

two_d_plot(Trec5, x_grid, tg5)
