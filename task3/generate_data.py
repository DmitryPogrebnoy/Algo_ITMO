import argparse
from itertools import product
import os.path as osp

import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import fmin_cg, minimize, least_squares


def linear(X, a, b):
    return X * a + b


def grad_linear(X, a, b):
    return X, np.full_like(X, 1)


def rational(X, a, b):
    return a / (1 + b * X)


def grad_rational(X, a, b):
    return 1 / (1 + b * X), -a * X / ((1 + b * X) ** 2)


def loss(func, X, a, b, y_true, reduction=True):
    approx = func(X, a, b)
    if reduction:
        return np.sum((approx - y_true) ** 2)
    else:
        return approx - y_true


def grad_loss(func, X, a, b, y_true):
    approx = func(X, a, b)
    return 2 * (approx - y_true)


def gd(func_name, X, y_true, a_init, b_init, lr=1e-4, epsilon=1e-4, max_epochs=1e5):
    a, b = a_init, b_init
    losses = []
    n_iters = 0
    while True:
        losses.append(loss(eval(func_name), X, a, b, y_true))
        grad_lf = grad_loss(eval(func_name), X, a, b, y_true)
        grad_fa, grad_fb = eval(f"grad_{func_name}")(X, a, b)
        grad_a, grad_b = np.sum(grad_lf * grad_fa), np.sum(grad_lf * grad_fb)
        a -= grad_a * lr
        b -= grad_b * lr
        n_iters += 1
        if (np.abs(grad_a) < epsilon and np.abs(grad_b) < epsilon) or n_iters > max_epochs:
            break
    return {"loss": losses[-1], "args": {"a": a, "b": b}, "iterations": n_iters}


def conj_gd(func_name, X, y_true, a_init, b_init, lr=1e-4, epsilon=1e-4, max_epochs=int(1e5)):
    def func(x):
        return loss(eval(func_name), X, a=x[0], b=x[1], y_true=y_true)

    def grad_func(x):
        grad_lf = grad_loss(eval(func_name), X, a=x[0], b=x[1], y_true=y_true)
        grad_fa, grad_fb = eval(f"grad_{func_name}")(X, a=x[0], b=x[1])
        grad_a, grad_b = np.sum(grad_lf * grad_fa), np.sum(grad_lf * grad_fb)
        return np.asarray((grad_a, grad_b))

    result = fmin_cg(
        func, x0=np.asarray([a_init, b_init]), fprime=grad_func, gtol=epsilon, maxiter=max_epochs, full_output=True
    )
    return {"loss": result[1], "args": {"a": result[0][0], "b": result[0][1]}, "iterations": result[2]}


def newton(func_name, X, y_true, a_init, b_init, lr=1e-4, epsilon=1e-4, max_epochs=1e5):
    def func(x):
        return loss(eval(func_name), X, a=x[0], b=x[1], y_true=y_true)

    def grad_func(x):
        grad_lf = grad_loss(eval(func_name), X, a=x[0], b=x[1], y_true=y_true)
        grad_fa, grad_fb = eval(f"grad_{func_name}")(X, a=x[0], b=x[1])
        grad_a, grad_b = np.sum(grad_lf * grad_fa), np.sum(grad_lf * grad_fb)
        return np.asarray((grad_a, grad_b))

    result = minimize(
        func, x0=np.asarray([a_init, b_init]), jac=grad_func, tol=epsilon, options={"maxiter": max_epochs}
    )

    return {"loss": result.fun, "args": {"a": result.x[0], "b": result.x[1]}, "iterations": result.nfev}


def lm(func_name, X, y_true, a_init, b_init, lr=1e-4, epsilon=1e-4, max_epochs=1e5):
    def func(x):
        return loss(eval(func_name), X, a=x[0], b=x[1], y_true=y_true, reduction=False)

    def grad_func(x):
        grad_fa, grad_fb = eval(f"grad_{func_name}")(X, a=x[0], b=x[1])
        return np.vstack([grad_fa, grad_fb]).T

    result = least_squares(
        func, x0=np.asarray([a_init, b_init]), jac=grad_func, method="lm", gtol=epsilon, max_nfev=int(max_epochs)
    )
    return {"loss": result.cost * 2, "args": {"a": result.x[0], "b": result.x[1]}, "iterations": result.nfev}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task3")
    parser.add_argument(
        "--input_task2", default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task2_data_2d.csv"),
        help="True data from task 2"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=1e5, help="Maximum of epochs")
    parser.add_argument("--output", default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task3.csv"),
                        help="Output file")

    args = parser.parse_args()

    df_true = pd.read_csv(args.input_task2)

    a_init, b_init = 0, 0
    data_opt = []
    for method, func_name in tqdm.tqdm(
            product(["gd", "conj_gd", "newton", "lm"], ["linear", "rational"]),
            total=8,
            desc="Finding minimums",
    ):
        opt_res = eval(method)(
            func_name, df_true["X"], df_true["y"], a_init, b_init, lr=args.lr, max_epochs=args.max_epochs
        )
        data_opt.append(
            {
                "method": method,
                "approx_func": func_name,
                "loss": opt_res["loss"],
                "a": opt_res["args"]["a"],
                "b": opt_res["args"]["b"],
                "iterations": opt_res["iterations"],
            }
        )

    data_opt = pd.DataFrame(data_opt)
    data_opt.to_csv(args.output, index=False)
