import timeit

import numba as nb
import numpy as np

@nb.njit(parallel=True)
def custom_isin(matrix, index_to_remove, set_arr):
    set_arr[index_to_remove] = True
    out = np.empty(matrix.shape, dtype=nb.boolean)

    for i in nb.prange(matrix.shape[0]):
        for j in nb.prange(matrix.shape[1]):
            if set_arr[matrix[i][j]]:
                out[i][j] = False
            else:
                out[i][j] = True
    set_arr[index_to_remove] = False
    return out

@nb.njit(parallel=True)
def custom_isin_set(matrix, index_to_remove):
    index_to_remove_set = set(index_to_remove)
    out = np.empty(matrix.shape, dtype=nb.boolean)

    for i in nb.prange(matrix.shape[0]):
        for j in nb.prange(matrix.shape[1]):
            if matrix[i][j] in index_to_remove_set:
                out[i][j] = False
            else:
                out[i][j] = True
    return out

# Set up the data
matrix = np.random.randint(0, 100000000, size=(800, 1000))
index_to_remove = np.random.randint(0, 100000000, size=800)

# Benchmark np.isin
def bench_np_isin():
    return ~np.isin(matrix, index_to_remove)

set_arr = np.empty(200000000, dtype=np.bool_)
for i in range(len(set_arr)):
    set_arr[i] = False

# Benchmark in1d_vec_nb
def bench_in1d_vec_nb():
    return custom_isin(matrix, index_to_remove, set_arr)

def bench_custom_isin_set():
    return custom_isin_set(matrix, index_to_remove)

# Run the benchmarks
number = 2000
np_isin_time = timeit.timeit(bench_np_isin, number=number)
in1d_vec_nb_time = timeit.timeit(bench_in1d_vec_nb, number=number)
custom_isin_set_time = timeit.timeit(bench_custom_isin_set, number=number)

print(f"np.isin time: {np_isin_time:.6f} seconds")
print(f"in1d_vec_nb time: {in1d_vec_nb_time:.6f} seconds")
print(f"custom_isin_set time: {custom_isin_set_time:.6f} seconds")
print(f"Speedup factor: {np_isin_time / in1d_vec_nb_time:.2f}x")
print(f"Speedup factor: {np_isin_time / custom_isin_set_time:.2f}x")
# Verify that both functions produce the same result
np_isin_result = bench_np_isin()
in1d_vec_nb_result = bench_in1d_vec_nb()
custom_isin_set_result = bench_custom_isin_set()
print(f"Results are identical: {np.array_equal(np_isin_result, in1d_vec_nb_result)}")
print(f"Results are identical: {np.array_equal(np_isin_result, custom_isin_set_result)}")
