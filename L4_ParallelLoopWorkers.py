import numpy as np, time, os, statistics, matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool

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

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        parts = pool.map(_worker, chunks)
        
    return np.vstack(parts)

N, max_iter = 1024, 100
X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

# Serial baseline (Numba already warm after M1 warm-up)
times = []
for _ in range(3):
    t0 = time.perf_counter()
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    times.append(time.perf_counter() - t0)
t_serial = statistics.median(times)

if __name__ == '__main__':
    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks) # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)
            speedup = t_serial / t_par
            print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")   
""" 
Results
 1 workers: 0.053s, speedup=0.82x, eff=82%
 2 workers: 0.031s, speedup=1.44x, eff=72%
 3 workers: 0.034s, speedup=1.28x, eff=43%
 4 workers: 0.025s, speedup=1.74x, eff=44%
 5 workers: 0.025s, speedup=1.80x, eff=36%
 6 workers: 0.022s, speedup=1.97x, eff=33%
 7 workers: 0.020s, speedup=2.18x, eff=31%
 8 workers: 0.018s, speedup=2.51x, eff=31%
 9 workers: 0.017s, speedup=2.57x, eff=29%
10 workers: 0.018s, speedup=2.48x, eff=25%
11 workers: 0.016s, speedup=2.78x, eff=25%
12 workers: 0.018s, speedup=2.44x, eff=20% 
"""