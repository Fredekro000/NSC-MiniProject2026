import numpy as np, time
from numba import njit

@njit
def mandelbrot_numba(x_max, x_min, y_max, y_min, size):
    step_x = (x_max - x_min) / (size - 1)
    step_y = (y_max - y_min) / (size - 1)

    result = np.zeros((size, size), dtype=np.int32)

    for j in range(size):
        y = y_min + j * step_y
        for i in range(size):
            x = x_min + i * step_x

            c_real = x
            c_imag = y

            z_real = 0.0
            z_imag = 0.0

            for iteration in range(100):

                if z_real*z_real + z_imag*z_imag >= 4:
                    result[j, i] = iteration
                    break

                new_real = z_real*z_real - z_imag*z_imag + c_real
                new_imag = 2*z_real*z_imag + c_imag

                z_real = new_real
                z_imag = new_imag

    return result

_ = mandelbrot_numba(1, -2, 1.5, -1.5, 64)

start = time.time()
mandelbrot_numba(1, -2, 1.5, -1.5, 1024)
elapsed = time.time() - start
print(f"Computation time: {elapsed:.3f}")