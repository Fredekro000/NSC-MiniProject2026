import time, matplotlib.pyplot as plt

# Create grid function
def grid(x_max, x_min, y_max, y_min, size):
    # Create the stepsize i.e. the value difference between each point
    step_x = (x_max - x_min) / (size - 1)
    step_y = (y_max - y_min) / (size - 1)

    # Create the lists, with respect to stepsize
    real = [x_min + step_x * i for i in range(size)]
    imag = [y_min + step_y * j for j in range(size)]
 
    return real, imag

# Mandelbrot loop function
def mandelbrot_naive(x_max, x_min, y_max, y_min, size):
    start = time.time()
    bound = 2
    power = 2
    max_iter = 100

    # Get the real and imaginary list
    real, imag = grid(x_max, x_min, y_max, y_min, size)

    # Create list for
    iteration_array = []

    for y in imag:
        row = []
        for x in real:
            c = complex(x, y)
            z = 0
            for iteration_number in range(max_iter):
                if(abs(z) >= bound):
                    row.append(iteration_number)
                    break
                z = z**power + c
            else:
                row.append(0)
        iteration_array.append(row)
    
    # End time of computation
    elapsed = time.time() - start
    #print(f"Computation took {elapsed:.3f} seconds")

    # Visualisation
    """
    ax = plt.axes()
    plt.rc('text', usetex = True)
    ax.set_aspect('equal')
    graph = ax.pcolormesh(real, imag, iteration_array, cmap='hot')
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    plt.title(f"Computation took {elapsed:.3f} seconds")
    plt.gcf().set_size_inches(5,4)
    plt.show() """

    
mandelbrot_naive(1, -2, 1.5, -1.5, 1024)