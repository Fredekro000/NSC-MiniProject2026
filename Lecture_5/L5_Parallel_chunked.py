import numpy as np, time, os, statistics
from multiprocessing import Pool
from numba import njit


@njit(fastmath=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0:
            return i
        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2.0*z_real*z_imag + c_imag
        z_real = new_real
        z_imag = new_imag
    return max_iter


@njit(fastmath=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end-row_start, N), dtype=np.int32)

    dx = (x_max-x_min)/N
    dy = (y_max-y_min)/N

    for r in range(row_end-row_start):
        c_imag = y_min + row_start*dy + r*dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None):
    
    if n_chunks is None:
        #n_chunks = n_workers
        n_chunks = pool._processes

    chunk_size = max(1, N // n_chunks)

    chunks = []
    row = 0

    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
        row = end

    #with Pool(processes=n_workers) as pool:
        #pool.map(_worker,[(0, 1, N, x_min, x_max, y_min, y_max, max_iter)])
    parts = pool.map(_worker, chunks)

    return np.vstack(parts)


# PARAMETERS
N = 4096
max_iter = 100
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

#workers = 11
chunk_list = [1,2,4,8,16,32]


# SERIAL BASELINE
times = []
for _ in range(3):
    t0 = time.perf_counter()
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    times.append(time.perf_counter() - t0)
serial_time = statistics.median(times)
#print(f"Serial time: {serial_time:.5f}")

# PARALLEL TEST
if __name__ == "__main__":

    for workers in range(1, os.cpu_count() + 1):
        print(f"\nWorkers: {workers}")
        with Pool(processes=workers) as pool:
            
            # warming up JIT
            pool.map(_worker,[(0, 1, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)])

            for chunks in chunk_list:
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    mandelbrot_parallel(pool, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=workers, n_chunks=chunks)
                    times.append(time.perf_counter() - t0)
                t_parallel = statistics.median(times)

                speedup = serial_time / t_parallel
                efficiency = speedup / workers * 100
                lif = (workers * t_parallel) / serial_time - 1

                print(
                    f"chunks={chunks:2d} | "
                    f"time={t_parallel:.3f}s | "
                    f"speedup={speedup:.2f}x | "
                    f"eff={efficiency:.0f}% | "
                    f"LIF={lif:.2f}"
                )