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

if __name__ == '__main__':
    # Warmup
    _ = mandelbrot_serial(64, -2.0, 1.0, -1.5, 1.5, 100)
    _ = mandelbrot_parallel(64, -2.5, 1.0, -1.25, 1.25, n_workers=4)

    # Serial timing
    t0 = time.perf_counter()
    result0 = mandelbrot_serial(1024, -2.0, 1.0, -1.5, 1.5, 100)
    elapsed0 = time.perf_counter() - t0
    print(f"Serial computation time: {elapsed0:.3f}")

    # Parallel timing
    t1 = time.perf_counter()
    result1 = mandelbrot_parallel(1024, -2.0, 1.0, -1.5, 1.5, n_workers=4)
    elapsed1 = time.perf_counter() - t1
    print(f"Multiprocessing computation time: {elapsed1:.3f}")

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Serial plot
    axes[0].set_aspect('equal')
    graph0 = axes[0].imshow(result0, cmap='hot', origin='lower')
    axes[0].set_title(f"Serial ({elapsed0:.3f}s)")
    axes[0].set_xlabel("Real-Axis")
    axes[0].set_ylabel("Imaginary-Axis")
    fig.colorbar(graph0, ax=axes[0])

    # Parallel plot
    axes[1].set_aspect('equal')
    graph1 = axes[1].imshow(result1, cmap='hot', origin='lower')
    axes[1].set_title(f"Multiprocessing ({elapsed1:.3f}s)")
    axes[1].set_xlabel("Real-Axis")
    axes[1].set_ylabel("Imaginary-Axis")
    fig.colorbar(graph1, ax=axes[1])

    plt.tight_layout()
    plt.show()