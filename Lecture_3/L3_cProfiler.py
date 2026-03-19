import cProfile, pstats

from L2_mandelbrot import mandelbrot_numpy
from L1_mandelbrot import mandelbrot_naive

cProfile.run('mandelbrot_naive(1, -2, 1.5, -1.5, 1024)', 'naive_profile.prof')
cProfile.run('mandelbrot_numpy(1, -2, 1.5, -1.5, 1024)', 'numpy_profile.prof')


for name in ('naive_profile.prof', 'numpy_profile.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
