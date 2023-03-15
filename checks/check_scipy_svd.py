import numpy as np
import scipy.linalg as la
import time

# Generate random matrix with 1000 rows and 500 columns
A = np.random.rand(1000, 500)

# Test numpy.linalg.svd
start_time = time.time()
U, S, VT = np.linalg.svd(A, full_matrices=False)
end_time = time.time()
np_time = end_time - start_time
print(f"np.linalg.svd took {np_time:.5f} seconds")

# Test scipy.linalg.svd
start_time = time.time()
U, S, VT = la.svd(A, full_matrices=False)
end_time = time.time()
scipy_time = end_time - start_time
print(f"scipy.linalg.svd took {scipy_time:.5f} seconds")

# Compare the execution times
if np_time < scipy_time:
    print("numpy.linalg.svd is faster")
else:
    print("scipy.linalg.svd is faster")
    
