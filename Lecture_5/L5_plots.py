import numpy as np
import matplotlib.pyplot as plt

workers = np.arange(1,13)
chunks = [1,2,4,8,16,32]

speedup = np.array([
[0.78,0.77,0.78,0.79,0.80,0.73],
[0.84,1.37,1.49,1.53,1.47,1.50],
[0.84,1.42,1.61,2.11,2.07,2.08],
[0.61,1.21,1.33,2.16,2.40,2.60],
[0.78,1.28,1.66,2.46,2.88,2.96],
[0.79,1.37,1.67,2.61,3.38,3.03],
[0.76,1.32,1.62,2.66,3.35,3.23],
[0.79,0.90,1.63,1.96,2.56,3.50],
[0.79,1.25,1.59,2.57,3.52,3.83],
[0.79,1.31,1.56,2.39,3.87,3.75],
[0.76,1.31,1.60,2.48,3.62,3.74],
[0.76,1.33,1.65,2.55,3.76,3.72]
])

# Speedup vs workers
plt.figure()
for i,c in enumerate(chunks):
    plt.plot(workers, speedup[:,i], marker='o', label=f'chunks={c}')

plt.xlabel("Workers")
plt.ylabel("Speedup")
plt.title("Speedup vs Workers")
plt.legend()
plt.grid()
plt.show()


# Heatmap
plt.figure()
plt.imshow(speedup, aspect='auto')
plt.colorbar(label="Speedup")
plt.xticks(range(len(chunks)), chunks)
plt.yticks(range(len(workers)), workers)
plt.xlabel("Chunks")
plt.ylabel("Workers")
plt.title("Speedup Heatmap")
plt.show()