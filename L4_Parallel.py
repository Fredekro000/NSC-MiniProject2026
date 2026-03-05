from numba import njit, prange
from multiprocessing import Pool
import time, os, statistics, numpy as np

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        if z_real*z_real + z_imag*z_imag > 4.0:
            return i

        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2.0*z_real*z_imag + c_imag

        z_real = new_real
        z_imag = new_imag

    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):

    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):

        c_imag = y_min + (r + row_start) * dy

        for col in range(N):

            c_real = x_min + col * dx

            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100):

    nproc = os.cpu_count()
    rows_per_chunk = N // nproc

    tasks = []

    for p in range(nproc):
        start = p * rows_per_chunk
        end = (p+1) * rows_per_chunk if p < nproc-1 else N

        tasks.append((start, end, N, x_min, x_max, y_min, y_max, max_iter))

    with Pool(nproc) as pool:
        chunks = pool.starmap(mandelbrot_chunk, tasks)

    return np.vstack(chunks)

if __name__ == "__main__":

    start = time.time()
    result = mandelbrot_parallel(
        1024,
        -2.0, 1.0,
        -1.5, 1.5,
        100
    )

    elapsed = time.time() - start
    print(f"Computation time: {elapsed:.2f}")