import faiss
import numpy as np
import sys
import time
import math


def main():
    # Check if GPU support is available
    ngpu = faiss.get_num_gpus()
    print(f"Detected {ngpu} GPU(s).")

    if ngpu == 0:
        print("No GPU detected by Faiss.")
        sys.exit(1)
    elif ngpu == 1:
        print("Only 1 GPU detected. Using single GPU mode.")
        # IVFPQ parameters
        d = 256  # vector dimension
        nb = 1000000  # number of vectors to index
        nq = 1024  # number of queries
        nlist = int(4 * math.sqrt(nb))  # rule-of-thumb: 4 × √N
        m = 16  # PQ subvectors (must divide d)
        nbits = 8  # 8-bit codes

        # Create single GPU IVFPQ index
        res = faiss.StandardGpuResources()
        cpu_quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, m, nbits)

        # Convert to GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print(f"Using multi-GPU mode with {ngpu} GPUs.")
        # IVFPQ parameters
        d = 256  # vector dimension
        nb = 1000000  # number of vectors to index
        nq = 1024  # number of queries
        nlist = int(4 * math.sqrt(nb))  # rule-of-thumb: 4 × √N
        m = 16  # PQ subvectors (must divide d)
        nbits = 8  # 8-bit codes

        # Create multi-GPU IVFPQ index
        cpu_quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, m, nbits)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # one shard per GPU
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)

    # Generate data
    np.random.seed(0)
    # Training data for IVFPQ
    xt = np.random.random((50000, d)).astype("float32")
    # Database vectors
    xb = np.random.random((nb, d)).astype("float32")
    # Query vectors
    xq = np.random.random((nq, d)).astype("float32")

    # Train the index (required for IVFPQ)
    print("Training the IVFPQ index...")
    gpu_index.train(xt)
    print("Training completed.")

    # Set search parameters
    gpu_index.nprobe = 10  # number of clusters to search

    # Build the index once
    gpu_index.add(xb)
    if ngpu == 1:
        print(f"Added {gpu_index.ntotal} vectors to single GPU IVFPQ index.")
    else:
        print(f"Added {gpu_index.ntotal} vectors to multi-GPU IVFPQ index across {ngpu} GPUs.")

    # Time multiple search operations
    num_runs = 20
    k = 100
    search_times = []

    print(f"\nRunning gpu_index.search() {num_runs} times...")

    for run in range(num_runs):
        # Time the search operation
        start_time = time.time()
        D, I = gpu_index.search(xq, k)
        end_time = time.time()

        if run > 5:
            search_time = end_time - start_time
            search_times.append(search_time)

            print(f"Run {run + 1}: Searched {len(xq)} queries in {search_time:.4f} seconds")

    # Calculate statistics
    avg_time = np.mean(search_times)
    min_time = np.min(search_times)
    max_time = np.max(search_times)
    std_time = np.std(search_times)

    print(f"\nTiming Results for IVFPQ gpu_index.search():")
    print(f"Index parameters: nlist={nlist}, m={m}, nbits={nbits}, nprobe={gpu_index.nprobe}")
    print(f"Number of runs: {len(search_times)} (excluding first 6 warmup runs)")
    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Min time: {min_time:.4f} seconds")
    print(f"Max time: {max_time:.4f} seconds")
    print(f"Std deviation: {std_time:.4f} seconds")
    print(f"All times: {[f'{t:.4f}' for t in search_times]}")
    print(f"Average time per query: {avg_time/len(xq)*1000:.4f} ms")

    if ngpu == 1:
        print(f"\nFaiss single GPU IVFPQ test completed successfully.")
        print(f"Total vectors indexed: {gpu_index.ntotal} on 1 GPU")
    else:
        print(f"\nFaiss multi-GPU IVFPQ test completed successfully.")
        print(f"Total vectors indexed: {gpu_index.ntotal} across {ngpu} GPUs")

    # Show sample results from last search
    print(f"\nSample results from last search (k={k}):")
    print("distances:", D[0])
    print("indices  :", I[0])


if __name__ == "__main__":
    main()
