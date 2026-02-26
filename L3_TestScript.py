import time , statistics

from L3_naive_numba import mandelbrot_naive_numba
from L3_hybrid_numba import mandelbrot_hybrid_numba


def bench(fn, *args,  runs =5):
    fn(*args) # extra warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# Warm up ( triggers JIT compilation -- exclude from timing )
_ = mandelbrot_hybrid_numba(1, -2, 1.5, -1.5, 64)
_ = mandelbrot_naive_numba(1, -2, 1.5, -1.5, 64)


t_hybrid = bench(mandelbrot_hybrid_numba, 1, -2, 1.5, -1.5, 1024)
t_full = bench(mandelbrot_naive_numba, 1, -2, 1.5, -1.5, 1024)

print (f"Hybrid: {t_hybrid:.3f}s")
print (f"Fully compiled: {t_full:.3f}s")
print (f"Ratio: {t_hybrid/t_full:.1f}x")