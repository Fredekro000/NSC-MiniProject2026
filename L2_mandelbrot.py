import time, matplotlib.pyplot as plt, numpy as np

# Setup parameters
bound = 2
power = 2
max_iter = 100

x_max = 1
x_min = -2
y_max = 1.5
y_min = -1.5

size = 1024

# Grid function
def grid():
    x = np.linspace(x_min, x_max, size)
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y

    return C

# Mandelbrot function
def mandelbrot():
    C = grid()

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for _ in range(max_iter):
        mask = np.abs(Z) <= bound
        Z[mask] = Z[mask]**power + C[mask]
        M[mask] += 1
    
    return M


start = time.time()
M = mandelbrot()
elapsed = time.time() - start

ax = plt.axes()
plt.rc('text', usetex = True)
ax.set_aspect('equal')
graph = ax.pcolormesh(M, cmap='hot')
plt.colorbar(graph)
plt.xlabel("Real-Axis")
plt.ylabel("Imaginary-Axis")
plt.title(f"Computation took {elapsed:.3f} seconds")
plt.gcf().set_size_inches(5,4)
plt.show()