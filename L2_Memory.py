import numpy as np, time

N = 10000

a = np.random.rand(N, N) 



start2 = time.time()
for j in range(N):
    col_sum = np.sum(a[:, j])
elapsed2 = time.time() - start2
print("Col sum: ", col_sum)
print(f"Computation took {elapsed2:.3f} seconds")

start = time.time()
for i in range(N):
    row_sum = np.sum(a[i, :])
elapsed = time.time() - start
print("Row sum: ", row_sum)
print(f"Computation took {elapsed:.3f} seconds")


