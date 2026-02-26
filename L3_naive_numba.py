from numba import njit
import numpy as np

# Fully compiled

@njit   
def mandelbrot_naive_numba(x_max, x_min, y_max, y_min, size):
    step_x = (x_max - x_min) / (size - 1)
    step_y = (y_max - y_min) / (size - 1)

    # Create the lists, with respect to stepsize
    real = [x_min + step_x * i for i in range(size)]
    imag = [y_min + step_y * j for j in range(size)]

     # Create list for
    iteration_array = []

    for y in imag:
        row = []
        for x in real:
            c = complex(x, y)
            z = 0
            for iteration_number in range(100):
                if(abs(z) >= 2):
                    row.append(iteration_number)
                    break
                z = z**2 + c
            else:
                row.append(0)
        iteration_array.append(row)