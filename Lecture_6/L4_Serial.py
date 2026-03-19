import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics


@njit(fastmath=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: return i
        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2.0*z_real*z_imag + c_imag
        z_real = new_real
        z_imag = new_imag
    return max_iter

@njit(fastmath=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + row_start * dy + r * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return  out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def bench(fn, *args,  runs =5):
    fn(*args) # extra warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

_ = mandelbrot_serial(64, -2.0, 1.0, -1.5, 1.5, 100)

def reffing():
    ref = mandelbrot_serial(1024, -2.0, 1.0, -1.5, 1.5, 100)
    return ref

#t_split = bench(mandelbrot_serial, 1024, -2.0, 1.0, -1.5, 1.5, 100)
#print(f"Median computation time: {t_split:.3f}")