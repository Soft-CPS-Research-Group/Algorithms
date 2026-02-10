import time
import torch

def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Using device: {torch.cuda.get_device_name(device.index)}")

    size = 12000
    repeats = 150
    dtype = torch.float32

    print(f"Running GPU matmul benchmark: size={size}, repeats={repeats}, dtype={dtype}")

    torch.cuda.empty_cache()
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.perf_counter()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"Warmup done in {time.perf_counter() - start:.2f}s")

    checksum = torch.zeros((), device=device)
    for i in range(1, repeats + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = torch.matmul(a, b)
        checksum += c.sum()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"Iter {i}/{repeats}: {elapsed:.2f}s, partial checksum={checksum.item():.3e}")

    del a, b, c
    torch.cuda.empty_cache()
    print("Benchmark complete.")

if __name__ == "__main__":
    main()
