import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats as stt
from analytical_1D_heat import f_exact_gauss, temperature

# import pdb
# pdb.set_trace()

# =========================== GLOBAL CONSTANTS ==============================
D = 4.25e-6                         # the diffusion coefficient
L = 0.01                            # length (L) in meter

tf = 0.01                           # time in seconds
Nt = 1000                           # number of grid points in time
dt = tf / (Nt - 1)                  # time-step
t_grid = np.array([n * dt for n in range(Nt)])

# ====================== Initial Condition ======================
# Initial & Boundary Conditions
Thigh = 50.0                        # high temperature
Tlow = 0.0                          # low temperature

# =========================== Main Program ==============================
if __name__ == "__main__":

    # N = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    N = np.array([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000])  #, 16000, 33000])
    print(N)
    t = np.empty((N.shape[0], 1))
    err = np.empty((N.shape[0], 1))

    for i in range(len(N)):

        Nx = N[i]          # number of grid points in x
        midpt = int(Nx / 2)  # midpoint in x
        dx = L / Nx          # grid size
        x_grid = np.array([j * dx for j in range(Nx)])

        U = np.array([Tlow for i in range(0, Nx)])
        U[midpt] = Thigh  # Delta-function
        T0 = U

        print(N[i])
        n = N[i]

        t1 = time.time()
        # Calculate the numerical solution
        Unum = temperature(D, Nt, Nx, dx, dt, U)

        # Calculate the exact solution
        height = Unum[midpt]
        x0 = 0.005
        texact = tf
        Uexact = f_exact_gauss(x_grid, x0, texact, height, D)

        t2 = time.time()

        t[i, 0] = t2-t1

        error = 0
        for j in range(Nx):
            error += abs(Uexact[j] - Unum[j])
        err[i, 0] = error

    print('Calculation is finished.')
    # print("t[:,0] is ", t[:,0])

    fig, ax = plt.subplots()
    plt.plot(N, err[:, 0], 'k+', mew=4, ms=10, label='Test')

    # Slope analysis
    errfeul = err[:, 0]
    Ns = N

    mfeul, bfeul, rfeul, pfeul, stdfeul = stt.linregress(np.log10(Ns), np.log10(errfeul))

    plt.plot(Ns, errfeul, '--b', label="Slope m = %.3f" % mfeul)

    # plt.xlim([1e0, 1e7])
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"Number of points (Nx)")
    plt.ylabel(r"$L_1$ error")
    # plt.axis("equal")
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.show()
