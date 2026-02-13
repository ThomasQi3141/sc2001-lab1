from typing import List

# Insertion sort implementation -> returns # comparisons

def insertion_sort(arr: List[int], start: int, end: int) -> int:
    comparisons = 0
    # insertion sort implementation (sorts from start to end)
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

# Merge implementation -> returns # comparisons
def merge(arr: List[int], low: int, high: int):
    if high - low <= 0:
        return 0
    
    # partition into 2
    mid = (low + high) // 2
    left = arr[low : mid + 1]
    right = arr[mid + 1 : high + 1]
    
    # left, right, write pointers
    i, j, k = 0, 0, low
    comparisons = 0
    
    while i < len(left) and j < len(right):
        # compares first elem from both halves
        comparisons += 1
        # take smaller elem
        if left[i] < right[j]:
            arr[k] = left[i]
            i += 1
            k += 1
        elif left[i] > right[j]:
            arr[k] = right[j]
            j += 1
            k += 1
        else:
            # special case: equal elems
            if i == len(left) - 1 and j == len(right) - 1:
                # append both
                arr[k] = left[i]
                arr[k + 1] = right[j]
                i += 1
                j += 1
                k += 2
                break
            # otherwise just take from both
            arr[k] = left[i]
            arr[k + 1] = right[j]
            i += 1
            j += 1
            k += 2
        # otherwise, just append the rest (no more comps)
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

    return comparisons

# Original merge sort implementation -> returns # comparisons
def mergesort(arr: List[int], start: int, end: int):
    mid = (start + end) // 2
    if end - start <= 0:
        return 0
    
    comparisons = 0
    
    # more than 1 elem -> mergesort recursively
    if end - start > 1:
        comparisons += mergesort(arr, start, mid)
        comparisons += mergesort(arr, mid + 1, end)
        
    # otherwise, just merge the 2
    comparisons += merge(arr, start, end)
    return comparisons

# Hybrid mergesort -> returns # of comparisons
def hybrid_mergesort(arr: List[int], start: int, end: int, threshold: int) -> int:
    mid = (start + end) // 2

    if end - start <= 0:
        return 0

    if end - start + 1 <= threshold:
        return insertion_sort(arr, start, end)

    comparisons = 0

    if end - start > 1:
        comparisons += hybrid_mergesort(arr, start, mid, threshold)
        comparisons += hybrid_mergesort(arr, mid + 1, end, threshold)

    comparisons += merge(arr, start, end)
    return comparisons

