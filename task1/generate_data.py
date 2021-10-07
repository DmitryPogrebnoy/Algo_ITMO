import argparse
import os
import os.path as osp
import timeit
import tqdm

import numpy as np
import pandas as pd

import sort.bubblesort
import sort.quicksort
import sort.timsort


def const_f(vector):
    return 100


def sum_f(vector):
    return np.sum(vector)


def prod_f(vector):
    return np.prod(vector)


def direct_poly(vector, x_val=1.5):
    x_values = []
    poly_output = 0
    for degree in range(len(vector)):
        if degree == 0:
            x_values.append(1)
        else:
            x_values.append(x_values[-1] * x_val)
        poly_output += x_values[-1] * vector[degree]
    return poly_output


def horners_poly(vector, x_val=1.5):
    poly_output = 0
    for degree in list(range(len(vector)))[::-1]:
        if degree == len(vector) - 1:
            poly_output += vector[degree]
        else:
            poly_output += x_val * poly_output + vector[degree]
    return poly_output


def _filter_times(times):
    times_arr = np.asarray(times)
    times_result = times_arr[(times_arr != times_arr.max()) & (times_arr != times_arr.min())]
    if times_result.size == 0:
        return times_arr
    else:
        return times_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task 1")
    parser.add_argument("--random_state", type=int, default=10, help="Random state for random generator")
    parser.add_argument("--output_file",
                        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task1.csv"),
                        help="Output file")
    args = parser.parse_args()

    np.random.seed(args.random_state)
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)

    data = []
    for n in tqdm.tqdm(range(1, 2001), desc="Iterate over n"):
        vectors = np.split(np.random.uniform(low=0, high=100, size=n * 10), 10)
        current_sample = {"n": n}
        for operation in [
            "const_f", "sum_f", "prod_f",
            "direct_poly", "horners_poly",
            "sort.bubblesort.sort", "sort.quicksort.sort", "sort.timsort.sort",
        ]:
            times = []
            for vector in vectors:
                vector = vector.tolist()
                times.append(timeit.timeit(stmt=f"{operation}(vector)", globals=globals(), number=1))
            current_sample[operation] = _filter_times(times).mean()

        matrices = np.split(np.random.uniform(low=0, high=100, size=(2 * 10, n, n)), 10, axis=0)
        times = []
        for matrix in matrices:
            times.append(timeit.timeit(stmt="matrix[0].dot(matrix[1])", globals=globals(), number=1))
        current_sample["matrix_mult"] = _filter_times(times).mean()

        data.append(current_sample)

    df = pd.DataFrame(data)
    df.to_csv(args.output_file, index=False)
