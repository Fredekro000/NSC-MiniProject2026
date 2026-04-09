from multiprocessing import Pool
import time
def square(x):
    time.sleep(0.1) # Simulate work
    return x * x

if __name__ == '__main__':
    numbers = list(range(100))
    
    start = time.time()
    results_serial = [square(x) for x in numbers]
    time_serial = time.time() - start
    print(f"Serial: {time_serial:.2f}s")
    
    with Pool(processes=4) as pool:
        start_ = time.time()
        results_parallel = pool.map(square, numbers)
    
    time_parallel = time.time() - start_
    print(f"Parallel: {time_parallel:.2f}s")
    speedup = time_serial / time_parallel
    print(f"Speedup: {speedup:.2f}x")


# Results
# Serial: 10.06s
# Parallel: 2.89s
# Speedup: 3.48x

import psutil, os
print(os.cpu_count()) # logical
print(psutil.cpu_count(logical=False)) # physical
# Logical: 12
# Physical: 6