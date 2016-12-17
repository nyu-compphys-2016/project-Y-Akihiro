import numpy as np
# import pdb;
# pdb.set_trace()


def pvt(tf, dx, Nx, U, D, mu):
    # tf = 0.5  # length of time (sec)
    Nt = 1000  # number of grid points in time
    dt = tf / (Nt - 1)  # time-step
    t_grid = np.array([n * dt for n in range(Nt)])

    sigma = (D * dt) / (2. * dx * dx)
    rho = (-mu * dt) / (4. * dx)

    # Create Tridiagonal Matrices
    A_u = np.diagflat([-sigma + rho for i in range(Nx - 1)], -1) + \
          np.diagflat([1. + sigma + rho] + [1. + 2. * sigma for i in range(Nx - 2)] + [1. + sigma - rho]) + \
          np.diagflat([-(sigma + rho) for i in range(Nx - 1)], 1)

    B_u = np.diagflat([sigma - rho for i in range(Nx - 1)], -1) + \
          np.diagflat([1. - sigma - rho] + [1. - 2. * sigma for i in range(Nx - 2)] + [1. - sigma + rho]) + \
          np.diagflat([sigma + rho for i in range(Nx - 1)], 1)

    # Solve the System Iteratively
    U_record = []
    U_record.append(U)

    for ti in range(1, Nt):
        U_new = np.linalg.solve(A_u, B_u.dot(U))
        U = U_new
        U[0] = 0    # Boundary condition
        U[-1] = 0   # Boundary condition
        U_record.append(U)
    return U, U_record, t_grid, tf
