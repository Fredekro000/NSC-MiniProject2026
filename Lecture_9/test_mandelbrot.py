import pytest
from L4_Serial import mandelbrot_pixel, mandelbrot_serial
from L6_Dask import mandelbrot_dask
import numpy as np

# ----------------------
# 1. Simple unit tests
# ----------------------

def test_origin():
    assert mandelbrot_pixel(0.0, 0.0, 100) == 100

def test_far_outside():
    assert mandelbrot_pixel(5.0, 0.0, 100) == 1

def test_edge_case():
    assert mandelbrot_pixel(0.0, 2.0, 100) == 2


# ----------------------
# 2. Parametrized test 
# ----------------------

CASES = [
    (0.0, 0.0, 100, 100),
    (5.0, 0.0, 100, 1),
    (-2.5, 0.0, 100, 1),
]

@pytest.mark.parametrize("real, imag, max_iter, expected", CASES)
def test_pixel_parametrized(real, imag, max_iter, expected):
    assert mandelbrot_pixel(real, imag, max_iter) == expected


# ----------------------
# 3. Integration test 
# ----------------------

def test_dask_matches_serial():
    N = 32
    args = (N, -2.0, 1.0, -1.5, 1.5, 50)

    serial = mandelbrot_serial(*args)
    dask = mandelbrot_dask(*args)

    assert np.array_equal(serial, dask)