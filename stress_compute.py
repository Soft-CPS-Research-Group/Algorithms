import time
import numpy as np

if __name__ == "__main__":
    size = 4000
    repeats = 150
    print(f"Starting heavy matrix benchmark: size={size}, repeats={repeats}")
    total = 0.0
    for i in range(1, repeats + 1):
        t0 = time.perf_counter()
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = a @ b
        total += c.sum()
        elapsed = time.perf_counter() - t0
        print(f"Iteration {i}/{repeats} done in {elapsed:.2f}s; checksum={total:.3e}")
        del a, b, c
    print("Benchmark complete.")
