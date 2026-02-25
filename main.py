import random
import time
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(20000)
# NOTE: run through venv -> source .venv/bin/activate


def insertion_sort(arr, start, end):
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
    left = arr[start : mid + 1]
    right = arr[mid + 1 : end + 1]
    
    i = 0 
    j = 0 
    k = start 
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

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
        
    return comparisons

def original_mergesort(arr, start, end):

    comparisons = 0
    if start < end:
        mid = (start + end) // 2
        
        comparisons += original_mergesort(arr, start, mid)
        comparisons += original_mergesort(arr, mid + 1, end)
        comparisons += merge(arr, start, mid, end)
        
    return comparisons

def hybrid_mergesort(arr, start, end, S):

    comparisons = 0
    
    size = end - start + 1
    
    if size <= S:
        return insertion_sort(arr, start, end)
    
    if start < end:
        mid = (start + end) // 2
        comparisons += hybrid_mergesort(arr, start, mid, S)
        comparisons += hybrid_mergesort(arr, mid + 1, end, S)
        comparisons += merge(arr, start, mid, end)
        
    return comparisons


def generate_input_data(size, max_val):
    return [random.randint(1, max_val) for _ in range(size)]



def run_experiment_c_i():

    print("\n--- Running Experiment (c) i: Varying N, Fixed S ---")
    sizes = [1000, 5000, 10000, 20000, 50000, 100000] 
    fixed_S = 10 
    comparisons_list = []
    
    for n in sizes:
        data = generate_input_data(n, n) # Range [1..n]
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

    print("\n--- Running Experiment (c) ii: Fixed N, Varying S ---")
    fixed_n = 50000 # Use a moderate size

    s_values = range(0, 105, 5) 
    comparisons_list = []
    

    original_data = generate_input_data(fixed_n, fixed_n)
    
    for s in s_values:
        arr_copy = original_data[:] 
        comps = hybrid_mergesort(arr_copy, 0, len(arr_copy)-1, s)
        comparisons_list.append(comps)

        
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, comparisons_list, marker='x', color='r')
    plt.xlabel('Threshold (S)')
    plt.ylabel('Key Comparisons')
    plt.title(f'Comparisons vs Threshold S (Fixed n={fixed_n})')
    plt.grid(True)
    plt.savefig('experiment_c_ii.png')
    plt.show()
    

    min_comps = min(comparisons_list)
    optimal_index = comparisons_list.index(min_comps)
    print(f"Optimal S based on comparisons for n={fixed_n} is approx: {s_values[optimal_index]}")
    return s_values[optimal_index]

def run_experiment_c_i_time():
    print("\n--- Running Experiment (c) i: Finding Optimal S (by Time) ---")
    fixed_n = 50000
    s_values = list(range(0, 100, 1))
    times_list = []
    original_data = generate_input_data(fixed_n, fixed_n)

    for s in s_values:
        run_times = []
        runs = 20

        for _ in range(runs):
            arr_copy = original_data[:]
            start_time = time.perf_counter()
            hybrid_mergesort(arr_copy, 0, len(arr_copy) - 1, s)
            end_time = time.perf_counter()
            run_times.append(end_time - start_time)

        run_times.sort()
        median_time = run_times[runs // 2]
        times_list.append(median_time)
        print(f"S: {s}, Median Time (20 runs): {median_time:.5f}s")

    plt.figure(figsize=(10, 6))
    plt.plot(s_values, times_list, marker='o', color='green', label='Median CPU Time (20 runs)')
    plt.xlabel('Threshold (S)')
    plt.ylabel('Median Time (seconds)')
    plt.title(f'Execution Time vs Threshold S (n={fixed_n})')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_c_iii_time.png')
    plt.show()

    min_time = min(times_list)
    optimal_index = times_list.index(min_time)
    optimal_S = s_values[optimal_index]
    print(f"\n>>> OPTIMAL S based on MEDIAN TIME is: {optimal_S} <<<")

    return optimal_S

def run_experiment_d(optimal_S):
    """Compares Original vs Hybrid Mergesort using median time over 20 runs."""
    print(f"\n--- Running Experiment (d): Original vs Hybrid (S={optimal_S}) ---")

    # WARNING: Setting this to 10,000,000 in Python will take a very long time.
    # Recommended for testing script: 100,000 or 200,000
    SIZE = 1_000_000
    RUNS = 20

    print(f"Generating data size: {SIZE}...")
    base_data = generate_input_data(SIZE, SIZE * 10)

    # --- Comparisons (single run on same data) ---
    data_orig_comps = base_data[:]
    data_hybrid_comps = base_data[:]

    print("Counting comparisons (single run each)...")
    comps_orig = original_mergesort(data_orig_comps, 0, len(data_orig_comps) - 1)
    comps_hybrid = hybrid_mergesort(data_hybrid_comps, 0, len(data_hybrid_comps) - 1, optimal_S)

    # --- Timing (median of 20 runs) ---
    print(f"Measuring execution time (median of {RUNS} runs)...")
    times_orig = []
    times_hybrid = []

    for i in range(RUNS):
        # Original mergesort timing
        data_orig = base_data[:]
        start_time = time.perf_counter()
        original_mergesort(data_orig, 0, len(data_orig) - 1)
        end_time = time.perf_counter()
        times_orig.append(end_time - start_time)

        # Hybrid mergesort timing
        data_hybrid = base_data[:]
        start_time = time.perf_counter()
        hybrid_mergesort(data_hybrid, 0, len(data_hybrid) - 1, optimal_S)
        end_time = time.perf_counter()
        times_hybrid.append(end_time - start_time)

        print(f"Run {i+1:2d}/{RUNS} completed")

    times_orig.sort()
    times_hybrid.sort()
    median_orig = times_orig[RUNS // 2]
    median_hybrid = times_hybrid[RUNS // 2]

    print("\nResults:")
    print(f"{'Algorithm':<20} | {'Comparisons':<15} | {'Median Time (s)':<16}")
    print("-" * 60)
    print(f"{'Original Mergesort':<20} | {comps_orig:<15} | {median_orig:.6f}")
    print(f"{'Hybrid Mergesort':<20} | {comps_hybrid:<15} | {median_hybrid:.6f}")


def run_experiment_optimal_s_vs_n():
    print("\n--- Running Experiment: n vs Optimal S (by CPU Time) ---")
    
    n_values = [
        200, 400, 1_000,
        2_000, 4_000, 10_000,
        20_000, 40_000, 100_000,
        200_000
    ]
    s_values = list(range(0, 200, 5))
    
    optimal_s_list = []
    best_times = []
    
    for n in n_values:
        print(f"\nTesting n = {n}")
        original_data = generate_input_data(n, n)
        
        best_s = None
        best_time = float("inf")
        
        for s in s_values:
            total_time = 0
            runs = 3
            
            for _ in range(runs):
                arr_copy = original_data[:]
                start_time = time.perf_counter()
                hybrid_mergesort(arr_copy, 0, len(arr_copy) - 1, s)
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
            
            avg_time = total_time / runs
            print(f"  S: {s}, Avg Time: {avg_time:.5f}s")
            
            if avg_time < best_time:
                best_time = avg_time
                best_s = s
        
        optimal_s_list.append(best_s)
        best_times.append(best_time)
        print(f">>> For n={n}, optimal S is {best_s} (Avg Time: {best_time:.5f}s)")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, optimal_s_list, marker='o', color='purple', label='Optimal S (by CPU Time)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Optimal Threshold (S)')
    plt.title('Optimal S vs Input Size n')
    plt.grid(True)
    plt.legend()
    plt.savefig('optimal_s_vs_n.png')
    plt.show()
    
    return n_values, optimal_s_list, best_times


# Main Execution

if __name__ == "__main__":
    print("--- Implementation Check ---")
    test_arr = generate_input_data(20, 100)
    print(f"Original: {test_arr}")
    comps = hybrid_mergesort(test_arr, 0, len(test_arr)-1, 5)
    print(f"Sorted:   {test_arr}")
    print(f"Comparisons: {comps}")
    
    assert test_arr == sorted(test_arr), "Error: Array not sorted correctly!"
    print("Sort Logic Validated.\n")

    
    # Part (c) i
    run_experiment_c_i()
    
    # Part (c) ii & iii (Finding optimal S)
    run_experiment_c_ii()

    optimal_S = run_experiment_c_i_time()
    
    # Part (d) Comparison
    run_experiment_d(optimal_S)

    # run_experiment_optimal_s_vs_n()
    