import numpy as np
import time
import statistics
from numba import njit
from typing import Callable, Tuple


@njit(fastmath=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Compute the number of iterations for a single point in the Mandelbrot set.

    The function iterates z_{n+1} = z_n^2 + c until either:
    - The magnitude exceeds 2 (escape condition), or
    - The maximum number of iterations is reached.

    Parameters
    ----------
    c_real : float
        Real part of the complex number c.
    c_imag : float
        Imaginary part of the complex number c.
    max_iter : int
        Maximum number of iterations to perform.

    Returns
    -------
    int
        Number of iterations before escape, or max_iter if it does not escape.
    """
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i

        new_real = z_real * z_real - z_imag * z_imag + c_real
        new_imag = 2.0 * z_real * z_imag + c_imag

        z_real = new_real
        z_imag = new_imag

    return max_iter


@njit(fastmath=True)
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int
) -> np.ndarray:
    """
    Compute a horizontal chunk of the Mandelbrot set image.

    This function calculates iteration counts for a subset of rows,
    allowing parallelization by splitting the image into chunks.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive).
    row_end : int
        Ending row index (exclusive).
    N : int
        Total number of columns (image width).
    x_min : float
        Minimum x-value (real axis).
    x_max : float
        Maximum x-value (real axis).
    y_min : float
        Minimum y-value (imaginary axis).
    y_max : float
        Maximum y-value (imaginary axis).
    max_iter : int
        Maximum number of iterations per pixel.

    Returns
    -------
    np.ndarray
        2D array of shape (row_end - row_start, N) containing iteration counts.
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + row_start * dy + r * dy
        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100
) -> np.ndarray:
    """
    Compute the full Mandelbrot set image serially.

    This is a convenience wrapper around `mandelbrot_chunk`
    that computes the entire image in one call.

    Parameters
    ----------
    N : int
        Image resolution (NxN).
    x_min : float
        Minimum x-value (real axis).
    x_max : float
        Maximum x-value (real axis).
    y_min : float
        Minimum y-value (imaginary axis).
    y_max : float
        Maximum y-value (imaginary axis).
    max_iter : int, optional
        Maximum number of iterations per pixel (default is 100).

    Returns
    -------
    np.ndarray
        2D array of shape (N, N) containing iteration counts.
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def bench(fn: Callable, *args, runs: int = 5) -> float:
    """
    Benchmark a function by measuring its median execution time.

    A warm-up run is performed before timing to avoid JIT or caching effects.

    Parameters
    ----------
    fn : Callable
        Function to benchmark.
    *args
        Arguments to pass to the function.
    runs : int, optional
        Number of timed runs (default is 5).

    Returns
    -------
    float
        Median execution time in seconds.
    """
    # Warm-up run
    fn(*args)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times)


def reffing() -> Tuple[np.ndarray, float]:
    """
    Compute a reference Mandelbrot image and measure its runtime.

    Runs the serial Mandelbrot computation multiple times and returns
    both the result and the median execution time.

    Returns
    -------
    Tuple[np.ndarray, float]
        - Reference Mandelbrot image (1024x1024)
        - Median execution time in seconds
    """
    times = []

    for _ in range(3):
        t0 = time.perf_counter()
        ref = mandelbrot_serial(1024, -2.0, 1.0, -1.5, 1.5, 100)
        times.append(time.perf_counter() - t0)

    ref_time = statistics.median(times)
    return ref, ref_time


# Warm-up call to trigger JIT compilation
_ = mandelbrot_serial(64, -2.0, 1.0, -1.5, 1.5, 100)

# Benchmark large computation
t_split = bench(mandelbrot_serial, 4096, -2.0, 1.0, -1.5, 1.5, 100)
print(f"Median computation time: {t_split:.3f}")