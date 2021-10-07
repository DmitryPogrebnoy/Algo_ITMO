import random

def double_partition(array, p, r):
    idx_left, idx_right = p, r
    middle_idx = random.randint(p, r)
    middle_value = array[middle_idx]
    while idx_left <= idx_right:
        while array[idx_left] < middle_value:
            idx_left += 1
        while array[idx_right] > middle_value:
            idx_right -= 1

        if idx_left <= idx_right:
            array[idx_left], array[idx_right] = array[idx_right], array[idx_left]
            idx_left, idx_right = idx_left + 1, idx_right - 1
    return idx_left, idx_right


def sort(array, p=None, r=None):
    if p is None:
        p = 0
    if r is None:
        r = len(array) - 1
    if p < r:
        q_left, q_right = double_partition(array, p, r)
        sort(array, p, q_right)
        sort(array, q_left, r)