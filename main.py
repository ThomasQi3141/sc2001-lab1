import random
import time
import sys
import matplotlib.pyplot as plt

# Increase recursion depth for deep mergesorts on large arrays
sys.setrecursionlimit(20000)

#Sorting Algorithms


def insertion_sort(arr, start, end):
    """
    Sorts arr[start...end] using Insertion Sort.
    Returns the number of key comparisons performed.
    """
    comparisons = 0
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        
        while j >= start:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            else:
                break
        arr[j + 1] = key
        
    return comparisons

def merge(arr, start, mid, end):
    """
    Merges two subarrays of arr[].
    First subarray is arr[start..mid]
    Second subarray is arr[mid+1..end]
    Returns number of comparisons.
    """
    # Create temp arrays (Slicing in Python creates copies)
    left = arr[start : mid + 1]
    right = arr[mid + 1 : end + 1]
    
    i = 0 # Initial index of first subarray
    j = 0 # Initial index of second subarray
    k = start # Initial index of merged subarray
    comparisons = 0
    
    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    # Copy the remaining elements of left[], if there are any
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    # Copy the remaining elements of right[], if there are any
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
        
    return comparisons

def original_mergesort(arr, start, end):
    """
    Standard Mergesort.
    Returns number of comparisons.
    """
    comparisons = 0
    if start < end:
        mid = (start + end) // 2
        
        comparisons += original_mergesort(arr, start, mid)
        comparisons += original_mergesort(arr, mid + 1, end)
        comparisons += merge(arr, start, mid, end)
        
    return comparisons

def hybrid_mergesort(arr, start, end, S):
    """
    Hybrid Sort: Mergesort switching to Insertion Sort when size <= S.
    Returns number of comparisons.
    """
    comparisons = 0
    
    # Calculate current size of the subarray
    size = end - start + 1
    
    # Base case: If size is small enough, use Insertion Sort
    if size <= S:
        return insertion_sort(arr, start, end)
    
    # Recursive step
    if start < end:
        mid = (start + end) // 2
        comparisons += hybrid_mergesort(arr, start, mid, S)
        comparisons += hybrid_mergesort(arr, mid + 1, end, S)
        comparisons += merge(arr, start, mid, end)
        
    return comparisons

# ==========================================
# 2. Data Generation
# ==========================================

def generate_input_data(size, max_val):
    """Generates a random list of integers."""
    return [random.randint(1, max_val) for _ in range(size)]

# ==========================================
# 3. Analysis Logic (Tasks c and d)
# ==========================================

def run_experiment_c_i():
    """
    Part (c) i: Fix S, plot comparisons vs different n (input sizes).
    """
    print("\n--- Running Experiment (c) i: Varying N, Fixed S ---")
    sizes = [1000, 5000, 10000, 20000, 50000, 100000] # Reduced for quick testing. Add 1M, 10M for final.
    fixed_S = 10 # Example threshold
    comparisons_list = []
    
    for n in sizes:
        data = generate_input_data(n, n) # Range [1..n]
        # Copy data to avoid sorting already sorted array if reused
        arr_copy = data[:] 
        comps = hybrid_mergesort(arr_copy, 0, len(arr_copy)-1, fixed_S)
        comparisons_list.append(comps)
        print(f"Size: {n}, Comparisons: {comps}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, comparisons_list, marker='o', label=f'Hybrid Sort (S={fixed_S})')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Key Comparisons')
    plt.title('Comparisons vs Input Size (Fixed S)')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_c_i.png')
    plt.show()

def run_experiment_c_ii():
    """
    Part (c) ii: Fix n, plot comparisons vs different S values.
    """
    print("\n--- Running Experiment (c) ii: Fixed N, Varying S ---")
    fixed_n = 50000 # Use a moderate size
    # S values to test. Avoid very large S as Insertion Sort becomes O(N^2)
    s_values = range(0, 105, 5) 
    comparisons_list = []
    
    # Generate one dataset to keep "n" consistent across trials
    original_data = generate_input_data(fixed_n, fixed_n)
    
    for s in s_values:
        arr_copy = original_data[:] 
        comps = hybrid_mergesort(arr_copy, 0, len(arr_copy)-1, s)
        comparisons_list.append(comps)
        # print(f"S: {s}, Comparisons: {comps}") # Uncomment for verbose output
        
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, comparisons_list, marker='x', color='r')
    plt.xlabel('Threshold (S)')
    plt.ylabel('Key Comparisons')
    plt.title(f'Comparisons vs Threshold S (Fixed n={fixed_n})')
    plt.grid(True)
    plt.savefig('experiment_c_ii.png')
    plt.show()
    
    # Simple check for optimal S in this run
    min_comps = min(comparisons_list)
    optimal_index = comparisons_list.index(min_comps)
    print(f"Optimal S based on comparisons for n={fixed_n} is approx: {s_values[optimal_index]}")
    return s_values[optimal_index]

def run_experiment_d(optimal_S):
    """
    Part (d): Compare Original vs Hybrid on Large Dataset
    Note: 10 Million in pure Python is extremely slow. 
    I will use 100,000 for demonstration. Change size variable to 10,000,000 for actual project.
    """
    print(f"\n--- Running Experiment (d): Original vs Hybrid (S={optimal_S}) ---")
    
    # WARNING: Setting this to 10,000,000 in Python will take a very long time.
    # Recommended for testing script: 100,000 or 200,000
    SIZE = 100_000 
    print(f"Generating data size: {SIZE}...")
    
    data_orig = generate_input_data(SIZE, SIZE * 10)
    data_hybrid = data_orig[:] # Exact copy
    
    # 1. Measure Original Mergesort
    print("Running Original Mergesort...")
    start_time = time.perf_counter()
    comps_orig = original_mergesort(data_orig, 0, len(data_orig)-1)
    end_time = time.perf_counter()
    time_orig = (end_time - start_time)
    
    # 2. Measure Hybrid Mergesort
    print("Running Hybrid Mergesort...")
    start_time = time.perf_counter()
    comps_hybrid = hybrid_mergesort(data_hybrid, 0, len(data_hybrid)-1, optimal_S)
    end_time = time.perf_counter()
    time_hybrid = (end_time - start_time)
    
    print("\nResults:")
    print(f"{'Algorithm':<20} | {'Comparisons':<15} | {'Time (Seconds)':<15}")
    print("-" * 55)
    print(f"{'Original Mergesort':<20} | {comps_orig:<15} | {time_orig:.4f}")
    print(f"{'Hybrid Mergesort':<20} | {comps_hybrid:<15} | {time_hybrid:.4f}")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Test implementation correctness
    print("--- Implementation Check ---")
    test_arr = generate_input_data(20, 100)
    print(f"Original: {test_arr}")
    # Run with small S
    comps = hybrid_mergesort(test_arr, 0, len(test_arr)-1, 5)
    print(f"Sorted:   {test_arr}")
    print(f"Comparisons: {comps}")
    
    # Check if actually sorted
    assert test_arr == sorted(test_arr), "Error: Array not sorted correctly!"
    print("Sort Logic Validated.\n")

    # 2. Run Analysis Tasks
    # Uncomment these lines to run the specific parts of the assignment
    
    # Part (c) i
    run_experiment_c_i()
    
    # Part (c) ii & iii (Finding optimal S)
    optimal_S = run_experiment_c_ii()
    
    # Part (d) Comparison
    run_experiment_d(optimal_S)

# from typing import List, Optional
# import random


# #ts the only thing that works
# def insertion_sort(arr: List[int], start: int, end: int) -> int:
#     comparisons = 0
#     for i in range(start + 1, end + 1):
#         key = arr[i]
#         j = i - 1
#         while j >= start:
#             comparisons += 1
#             if arr[j] > key:
#                 arr[j + 1] = arr[j]
#                 j -= 1
#             else:
#                 break
#         arr[j + 1] = key
#     return comparisons


# def merge(arr: List[int], low: int, high: int) -> int:
#     mid = (low + high) // 2
#     left = arr[low : mid + 1]
#     right = arr[mid + 1 : high + 1]
    
#     i, j, k = 0, 0, low
#     comparisons = 0
    
#     while i < len(left) and j < len(right):
#         comparisons += 1
#         if left[i] <= right[j]:
#             arr[k] = left[i]
#             i += 1
#         else:
#             arr[k] = right[j]
#             j += 1
#         k += 1

#     # Append remaining elements (no comparisons needed here)
#     while i < len(left):
#         arr[k] = left[i]
#         i += 1
#         k += 1
#     while j < len(right):
#         arr[k] = right[j]
#         j += 1
#         k += 1

#     return comparisons
# # # Merge implementation -> returns # comparisons
# # def merge(arr: List[int], low: int, high: int):
# #     if high - low <= 0:
# #         return 0
    
# #     # partition into 2
# #     mid = (low + high) // 2
# #     left = arr[low : mid + 1]
# #     right = arr[mid + 1 : high + 1]
    
# #     # left, right, write pointers
# #     i, j, k = 0, 0, low
# #     comparisons = 0
    
# #     while i < len(left) and j < len(right):
# #         # compares first elem from both halves
# #         comparisons += 1
# #         # take smaller elem
# #         if left[i] < right[j]:
# #             arr[k] = left[i]
# #             i += 1
# #             k += 1
# #         elif left[i] > right[j]:
# #             arr[k] = right[j]
# #             j += 1
# #             k += 1
# #         else:
# #             # special case: equal elems
# #             if i == len(left) - 1 and j == len(right) - 1:
# #                 # append both
# #                 arr[k] = left[i]
# #                 arr[k + 1] = right[j]
# #                 i += 1
# #                 j += 1
# #                 k += 2
# #                 break
# #             # otherwise just take from both
# #             arr[k] = left[i]
# #             arr[k + 1] = right[j]
# #             i += 1
# #             j += 1
# #             k += 2
# #         # otherwise, just append the rest (no more comps)
# #         while i < len(left):
# #             arr[k] = left[i]
# #             i += 1
# #             k += 1

# #         while j < len(right):
# #             arr[k] = right[j]
# #             j += 1
# #             k += 1

# #     return comparisons

# # Original merge sort implementation -> returns # comparisons
# def mergesort(arr: List[int], start: int, end: int):
#     mid = (start + end) // 2
#     if end - start <= 0:
#         return 0
    
#     comparisons = 0
    
#     # more than 1 elem -> mergesort recursively
#     if end - start > 1:
#         comparisons += mergesort(arr, start, mid)
#         comparisons += mergesort(arr, mid + 1, end)
        
#     # otherwise, just merge the 2
#     comparisons += merge(arr, start, end)
#     return comparisons

# # Hybrid mergesort -> returns # of comparisons
# def hybrid_mergesort(arr: List[int], start: int, end: int, threshold: int) -> int:
#     mid = (start + end) // 2

#     if end - start <= 0:
#         return 0

#     if end - start + 1 <= threshold:
#         return insertion_sort(arr, start, end)

#     comparisons = 0

#     if end - start > 1:
#         comparisons += hybrid_mergesort(arr, start, mid, threshold)
#         comparisons += hybrid_mergesort(arr, mid + 1, end, threshold)

#     comparisons += merge(arr, start, end)
#     return comparisons

# # function for generating random testing data
# def generate_random_array(size: int, x: int, seed: Optional[int] = None) -> List[int]:
#     rng = random.Random(seed)
#     return [rng.randint(1, x) for _ in range(size)]

# if __name__ == "__main__":
#     SIZE = 50
#     MAX_VAL = 200
#     my_list = generate_random_array(SIZE, MAX_VAL)
    
#     print(f"Original: {my_list}")

#     total_comparisons = hybrid_mergesort(my_list, 0, len(my_list) - 1,50)

#     # 3. View results
#     print(f"Sorted:   {my_list}")
#     print(f"Total Comparisons made: {total_comparisons}")
#     #print(f"Mergesort stats: {total_comparisons}")