import time , statistics

from L3_naive_numba import mandelbrot_naive_numba
from L3_hybrid_numba import mandelbrot_hybrid_numba
from L2_mandelbrot import mandelbrot_numpy
from L1_mandelbrot import mandelbrot_naive


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

t_naive = bench(mandelbrot_naive, 1, -2, 1.5, -1.5, 1024)
t_numpy = bench(mandelbrot_numpy, 1, -2, 1.5, -1.5, 1024)
t_numba = bench(mandelbrot_naive_numba, 1, -2, 1.5, -1.5, 1024)

print (f"Naive: {t_naive:.3f}s")
print (f"Numpy: {t_numpy:.3f}s")
print (f"Numba: {t_numba:.3f}s")