import argparse
import os.path as osp

import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import minimize, least_squares, differential_evolution
import pyswarms as ps


def generate(X):
    f_x = 1 / ((X ** 2) - (3 * X) + 2)
    f_x = np.clip(f_x, -100, 100)
    noise = np.random.normal(size=X.shape)
    return f_x, f_x + noise


def rational(X, params):
    params = np.asarray(params)
    if params.ndim == 1:
        return np.clip((params[0] * X + params[1]) / (X ** 2 + params[2] * X + params[3]), -100, 100)
    if params.ndim == 2:
        return np.clip(
            (params[:, 0] * X[:, None] + params[:, 1]) / (X[:, None] ** 2 + params[:, 2] * X[:, None] + params[:, 3]),
            -100,
            100,
        )


def grad_rational(X, params):
    nom = params[0] * X + params[1]
    denom = X ** 2 + params[2] * X + params[3]
    zero_grad = ((nom / denom) > 100) & ((nom / denom) < 100)
    grad_ = []
    grad_.append(X / denom)
    grad_.append(1 / denom)
    grad_.append(-nom * X / (denom ** 2))
    grad_.append(-nom / (denom ** 2))
    grad_ = np.asarray(grad_).T
    grad_[zero_grad] = 0
    return grad_


def loss(func, X, params, y_true, reduction=True):
    approx = func(X, params)
    if params.ndim == 1:
        if reduction:
            return np.sum((approx - y_true) ** 2)
        else:
            return approx - y_true
    if params.ndim == 2:
        return np.sum((approx - y_true[:, None]) ** 2, axis=0)


def grad_loss(func, X, params, y_true):
    approx = func(X, params)
    return 2 * (approx - y_true)


def nelder_mead(func_name, X, y_true, params_init, epsilon=1e-3, max_epochs=int(1e3)):
    def func(x):
        return loss(eval(func_name), X=X, params=x, y_true=y_true)

    result = minimize(
        func,
        x0=params_init,
        method="Nelder-Mead",
        options={"maxiter": max_epochs, "maxfev": max_epochs, "xatol": epsilon, "fatol": epsilon},
    )
    return {"loss": result.fun, "args": result.x, "iterations": result.nfev}


def lm(func_name, X, y_true, params_init, epsilon=1e-3, max_epochs=int(1e3)):
    def func(x):
        return loss(eval(func_name), X, params=x, y_true=y_true, reduction=False)

    def grad_func(x):
        grad_ = eval(f"grad_{func_name}")(X, x)
        return grad_

    result = least_squares(
        func, x0=params_init, jac=grad_func, method="lm", ftol=epsilon, gtol=epsilon, max_nfev=max_epochs
    )
    return {"loss": result.cost * 2, "args": result.x, "iterations": result.nfev}


def diff_ev(func_name, X, y_true, params_init, epsilon=1e-3, max_epochs=int(1e3)):
    def func(x):
        return loss(eval(func_name), X=X, params=x, y_true=y_true)
    
    bounds = [(-4, 4), (-4, 4), (-4, 4), (-4, 4)]
    init_pop = np.vstack([np.random.uniform(-4, 4, (14, 4)), params_init.reshape(1, -1)])

    result = differential_evolution(
        func, bounds=bounds, init=init_pop, maxiter=max_epochs, atol=epsilon, seed=42
    )
    return {"loss": result.fun, "args": result.x, "iterations": result.nfev // 15}


def func(x, func_name, X, y_true):
    return loss(eval(func_name), X=X, params=x, y_true=y_true)


def part_swarm(func_name, X, y_true, params_init, epsilon=1e-3, max_epochs=int(1e3)):
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=4, center=params_init, options=options)
    cost, x = optimizer.optimize(func, iters=max_epochs, func_name=func_name, X=X, y_true=y_true)
    return {"loss": cost, "args": x, "iterations": max_epochs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task4")
    parser.add_argument("--random_state", type=int, default=111, help="Random state for random generator")
    parser.add_argument("--max_epochs", type=int, default=int(1e3), help="Maximum of epochs")
    parser.add_argument(
        "--output_true",
        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task4_true.csv"),
        help="Output file with generated data",
    )
    parser.add_argument(
        "--output_approx",
        default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task4.csv"),
        help="Output file with approximated parameters",
    )

    args = parser.parse_args()

    np.random.seed(args.random_state)

    X = np.arange(0, 3.003, 0.003)
    y_true, y = generate(X)
    df_true = pd.DataFrame({"X": X, "y_clean": y_true, "y": y})
    df_true.to_csv(args.output_true)

    data_opt = []
    for try_num in tqdm.tqdm(range(6), desc="Finding optimal solution"):
        init_params = np.random.uniform(-4, 4, size=(4,))
        if try_num == 5:
            init_params = np.asarray([0, 1, -3, 2]) # true params
        print(init_params)
        for method in ["nelder_mead", "lm", "diff_ev", "part_swarm"]:
            opt_res = eval(method)("rational", df_true["X"], df_true["y"], init_params, max_epochs=args.max_epochs)
            data_opt.append(
                {
                    "method": method,
                    "approx_func": "rational",
                    "loss": opt_res["loss"],
                    "init_params": init_params.tolist(),
                    "params": opt_res["args"].tolist(),
                    "iterations": opt_res["iterations"],
                    "try_num": try_num,
                    "true_init": try_num == 5,
                }
            )

    data_opt = pd.DataFrame(data_opt)
    data_opt.to_csv(args.output_approx, index=False)
