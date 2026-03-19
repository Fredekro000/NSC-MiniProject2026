import numpy as np, numba, matplotlib.pyplot as plt, dask, pytest, scipy, time

start = time.time()

# get evenly spaced numbers for both domains
# this is essentially a list with 1024 numbers, 
# spread out between -2 to 1, and -1.5 to 1.5
x_domain = np.linspace(-2.0, 1.0, 1024)
y_domain = np.linspace(-1.5, 1.5, 1024)

# initialize parameters
bound = 2
power = 2
max_iter = 100
colormap = 'hot'

# nested loops for grid version of mandelbrot
iteration_array = []
for y in y_domain:
    row = []
    for x in x_domain:
        c = complex(x, y)
        z = 0
        for iteration_number in range(max_iter):
            if(abs(z) >= bound):
                row.append(iteration_number)
                break
            else: z = z**power + c
        else:
            row.append(0)
    iteration_array.append(row)

# end time of computation
elapsed = time.time() - start

ax = plt.axes()
plt.rc('text', usetex = True)
ax.set_aspect('equal')
graph = ax.pcolormesh(x_domain, y_domain, iteration_array, cmap=colormap)
plt.colorbar(graph)
plt.xlabel("Real-Axis")
plt.ylabel("Imaginary-Axis")
plt.title(f"Computation took {elapsed:.3f} seconds")
plt.gcf().set_size_inches(5,4)
plt.show()