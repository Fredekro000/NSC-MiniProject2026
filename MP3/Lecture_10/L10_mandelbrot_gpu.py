import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float z_real = 0.0f, z_imag = 0.0f;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0f) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double z_real = 0.0, z_imag = 0.0;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

N, MAX_ITER = 1024, 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25


def mandelbrot_gpu_f32(ctx, queue, N, x_min=X_MIN, x_max=X_MAX,
                       y_min=Y_MIN, y_max=Y_MAX, max_iter=MAX_ITER):
    prog = cl.Program(ctx, KERNEL_F32).build()
    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    prog.mandelbrot_f32(queue, (N, N), None, image_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),)

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    return image


def mandelbrot_gpu_f64(ctx, queue, N, x_min=X_MIN, x_max=X_MAX,
                       y_min=Y_MIN, y_max=Y_MAX, max_iter=MAX_ITER):
    dev = ctx.devices[0]

    prog = cl.Program(ctx, KERNEL_F64).build()
    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    prog.mandelbrot_f64(
        queue, (N, N), None, image_dev,
        np.float64(X_MIN), np.float64(X_MAX),
        np.float64(Y_MIN), np.float64(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),)

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()
    return image


def benchmark(func, *args, runs=3):
    """Return median wall time (seconds) over `runs` calls."""
    import statistics
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":

    N = 4096
    runs = 3
    
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    #prog = cl.Program(ctx, KERNEL_SRC).build()
    dev = ctx.devices[0]
    print(f"Device: {dev.name}\n")

    print(f"Benchmarking N={N}, max_iter={MAX_ITER}, {runs} runs each:\n")

    float32 = benchmark(mandelbrot_gpu_f32, ctx, queue, N, runs=runs)
    img_f32 = mandelbrot_gpu_f32(ctx, queue, N)
    print(f"float32: {float32*1e3:.1f} ms")


    float64 = benchmark(mandelbrot_gpu_f64, ctx, queue, N, runs=runs)
    img_f64 = mandelbrot_gpu_f64(ctx, queue, N)
    print(f"float64: {float64*1e3:.1f} ms")

    print(f"Ratio float64/float32: {float64/float32:.2f}x")
    diff = np.abs(img_f32.astype(int) - img_f64.astype(int))
    print(f"Max pixel difference (f32 vs f64): {diff.max()}")

    fig, axes = plt.subplots(1, 2 if img_f64 is not None else 1,
                             figsize=(12 if img_f64 is not None else 6, 5))
    if img_f64 is None:
        axes = [axes]

    axes[0].imshow(img_f32, cmap='hot', origin='lower')
    axes[0].set_title(f"float32  ({float32*1e3:.1f} ms)")
    axes[0].axis('off')

    if img_f64 is not None:
        axes[1].imshow(img_f64, cmap='hot', origin='lower')
        axes[1].set_title(f"float64  ({float64*1e3:.1f} ms)")
        axes[1].axis('off')

    plt.suptitle(f"GPU Mandelbrot  N={N}  device: {dev.name}", fontsize=10)
    plt.tight_layout()
    out = "mandelbrot_opencl.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nSaved to {out}")


""" out = np.zeros(N, dtype=np.int32)
out_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

prog.hello(queue, (N,), None, out_dev)
cl.enqueue_copy(queue, out, out_dev)
queue.finish()

print(out)     # → [ 0  1  4  9 16 25 36 49]  """