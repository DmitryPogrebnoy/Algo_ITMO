import argparse
from itertools import product
import os.path as osp

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.optimize
import tqdm


def x_3(array):
    return array ** 3


def module_x(array):
    return np.abs(array - 0.2)


def sin(array):
    return array * np.sin(1 / array)


def brute_force(func, left, right, epsilon=0.001):
    x_array = np.arange(left, right, epsilon)
    values = func(x_array)
    min_f = np.min(values)
    idx_min = np.argmin(values)
    return {"min": x_array[idx_min], "func_min": min_f, "iterations": x_array.shape[0]}


def dichotomy(func, left, right, epsilon=0.001, delta=None):
    if delta is None:
        delta = epsilon / 2
    coords = [[left, right]]
    iters = 0
    while (right - left) > epsilon:
        x1, x2 = (left + right - delta) / 2, (left + right + delta) / 2
        f_x1, f_x2 = map(func, [x1, x2])
        if f_x1 <= f_x2:
            right = x2
        else:
            left = x1
        iters += 2
        coords.append([left, right])
    min_f = func((right - left) / 2 + right)
    return {"min": (right - left) / 2 + right, "func_min": min_f, "iterations": iters, "coords": coords}


def golden_section(func, left, right, epsilon=0.001):
    delta = (3 - np.sqrt(5)) / 2
    coords = [[left, right]]
    x1, x2 = left + delta * (right - left), right - delta * (right - left)
    f_x1, f_x2 = map(func, [x1, x2])
    iters = 2
    while (right - left) > epsilon:
        if f_x1 <= f_x2:
            right = x2
            x2 = x1
            f_x2 = f_x1
            calc_x2 = False
        else:
            left = x1
            x1 = x2
            f_x1 = f_x2
            calc_x2 = True
        coords.append([left, right])

        if calc_x2:
            x2 = right - delta * (right - left)
            f_x2 = func(x2)
        else:
            x1 = left + delta * (right - left)
            f_x1 = func(x1)
        iters += 1
    min_f = func((right - left) / 2 + right)
    return {"min": (right - left) / 2 + right, "func_min": min_f, "iterations": iters, "coords": coords}


def linear_approx(X, a, b):
    return X * a + b


def rational_approx(X, a, b):
    return a / (1 + b * X)


def loss(func, X, a, b, y_true, a_bounds=(0, 1), b_bounds=(0, 1), apply_bounds=False):
    # Artificial bounds are here are for Nelder-Mead algorithm which have no constraints
    #  and tends to find optimal solution outside the bounds for bruteforce algorithm
    if apply_bounds:
        if a < a_bounds[0] or a > a_bounds[1]:
            return 10 ** 10
        if b < b_bounds[0] or b > b_bounds[1]:
            return 10 ** 10
    approx = func(X, a, b)
    return np.sum((approx - y_true) ** 2)


def brute_force_opt(func, X, y_true, a_bounds=(0, 1), b_bounds=(0, 1), epsilon=0.001):
    a_values = np.arange(a_bounds[0], a_bounds[1] + epsilon, epsilon)
    b_values = np.arange(b_bounds[0], b_bounds[1] + epsilon, epsilon)

    min_loss = 10 ** 10
    min_args = None
    for a in a_values:
        for b in b_values:
            loss_value = loss(func, X, a, b, y_true)
            if loss_value < min_loss:
                min_args = {"a": a, "b": b}
                min_loss = loss_value
    return {"loss": min_loss, "args": min_args, "iterations": a_values.shape[0] * b_values.shape[0]}


def gauss_opt(func, X, y_true, a_bounds=(0, 1), b_bounds=(0, 1), epsilon=0.001):
    a, b = map(np.mean, [a_bounds, b_bounds])
    a_prev, b_prev = a_bounds[0], b_bounds[0]
    loss_prev = loss(func, X, a, b, y_true)

    min_loss = 10 ** 10
    iters = 1
    loss_values = []
    coords = [[a, b]]

    while (
            scipy.spatial.distance.euclidean([a_prev, b_prev], [a, b]) > epsilon and np.abs(
        loss_prev - min_loss) > epsilon
    ):
        for opt_var in ["a", "b"]:
            if opt_var == "a":
                aux_func = lambda x: loss(func, X, a=x, b=b, y_true=y_true)
                opt = golden_section(aux_func, a_bounds[0], b_bounds[1], epsilon=epsilon)
                a_prev = a
                a = opt["min"]
            else:
                aux_func = lambda x: loss(func, X, a=a, b=x, y_true=y_true)
                opt = golden_section(aux_func, b_bounds[0], b_bounds[1], epsilon=epsilon)
                b_prev = b
                b = opt["min"]

            iters += opt["iterations"]
            min_loss = opt["func_min"]
            loss_values.append(min_loss)
            coords.append([a, b])
    return {
        "loss": min_loss,
        "args": {"a": a, "b": b},
        "iterations": iters,
        "loss_values": loss_values,
        "coords": coords,
    }


def nelder_mead_opt(func, X, y_true, a_bounds=(0, 1), b_bounds=(0, 1), epsilon=0.001):
    opt_func = lambda x: loss(func, X=X, a=x[0], b=x[1], y_true=y_true, a_bounds=a_bounds, b_bounds=b_bounds,
                              apply_bounds=True)
    a0, b0 = map(np.mean, [a_bounds, b_bounds])
    result = scipy.optimize.minimize(
        opt_func, x0=np.asarray([a0, b0]), method="Nelder-Mead", options={"xatol": epsilon, "fatol": epsilon}
    )
    return {"loss": result.fun, "args": {"a": result.x[0], "b": result.x[1]}, "iterations": result.nfev}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather data for task 2")
    parser.add_argument("--output_1d",
                        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task2_1d.csv"),
                        help="Output file")
    parser.add_argument("--output_data_2d",
                        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task2_data_2d.csv"),
                        help="Output file")
    parser.add_argument("--output_2d",
                        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task2_2d.csv"),
                        help="Output file")
    parser.add_argument("--random_state", type=int, default=111, help="Random state for random generator")
    args = parser.parse_args()

    np.random.seed(args.random_state)

    # 1d optimization
    data = []
    for optimizer in ("brute_force", "dichotomy", "golden_section"):
        for func, interval in zip(("x_3", "module_x", "sin"), ([0, 1], [0, 1], [0.1, 1])):
            result = eval(optimizer)(eval(func), interval[0], interval[1])
            data.append(
                {"optimizer": optimizer, "func": func, "min": result["min"], "func_min": result["func_min"],
                 "iterations": result["iterations"]}
            )
    data = pd.DataFrame(data)
    data.to_csv(args.output_1d, index=False)

    # 2d optimization
    # Generate data
    alpha, beta = np.random.uniform(size=2)
    print(f'Alpha: {alpha}, beta: {beta}')
    X = np.arange(0, 1.01, 0.01)
    deltas = np.random.normal(size=X.shape)
    y_clean = alpha * X + beta
    y = alpha * X + beta + deltas
    opt_data = pd.DataFrame(np.vstack([X, y_clean, y]).T, columns=['X', 'y_clean', 'y'])
    opt_data.to_csv(args.output_data_2d, index=False)

    # Gather optimization data
    data_opt = []

    for method, approx_func in tqdm.tqdm(
            product(["brute_force_opt", "gauss_opt", "nelder_mead_opt"], ["linear_approx", "rational_approx"]),
            total=6,
            desc="Optimizing 2D",
    ):
        opt_res = eval(method)(eval(approx_func), X, y, a_bounds=(-2, 2), b_bounds=(-2, 2))
        data_opt.append(
            {
                "method": method,
                "approx_func": approx_func,
                "loss": opt_res["loss"],
                "a": opt_res["args"]["a"],
                "b": opt_res["args"]["b"],
                "iterations": opt_res["iterations"],
            }
        )

    data_opt = pd.DataFrame(data_opt)
    data_opt.to_csv(args.output_2d, index=False)
