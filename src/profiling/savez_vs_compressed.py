import timeit

import numpy as np

# Create a sample array
arr = np.random.rand(1000, 800, 100)


# Define functions to time
def save_with_savez():
    np.savez("array_savez.npz", arr=arr)


def save_with_savez_compressed():
    np.savez_compressed("array_savez_compressed.npz", arr=arr)


# Time the functions
savez_time = timeit.timeit(save_with_savez, number=5)
savez_compressed_time = timeit.timeit(save_with_savez_compressed, number=5)

print(f"Time taken by np.savez: {savez_time:.4f} seconds")
print(f"Time taken by np.savez_compressed: {savez_compressed_time:.4f} seconds")

# Compare the speeds
speed_difference = (savez_time - savez_compressed_time) / savez_time * 100
print(f"np.savez_compressed is {speed_difference:.2f}% faster than np.savez")
