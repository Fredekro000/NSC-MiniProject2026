import time, matplotlib.pyplot as plt, numpy as np

# Setup parameters
bound = 2
power = 2
max_iter = 100

x_max = 1
x_min = -2
y_max = 1.5
y_min = -1.5

size = 1024

# Grid
def grid():
    x = np.linspace(x_min, x_max, size)
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)

    C = X + 1j + Y

    print (f" Shape : {C. shape }") 
    print (f" Type : {C. dtype }") 

