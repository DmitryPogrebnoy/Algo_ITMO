import argparse
import os.path as osp
import timeit
import tqdm

import numpy as np
import pandas as pd


def flat_to_real_ids(ids, size=100):
    row_indices = np.arange(size - 1, 0, step=-1)
    cum_row_indices = np.cumsum(row_indices)
    ids_row = []
    for idx in ids:
        row_idx = (cum_row_indices <= idx).sum() - (cum_row_indices == idx).sum()
        if row_idx == 0:
            col_idx = idx + row_indices[::-1][row_idx] - 1
        else:
            col_idx = (
                    idx - cum_row_indices[row_idx - 1] + row_indices[::-1][row_idx] - 1
            )
        ids_row.append([row_idx, col_idx])
    return np.asarray(ids_row)


def create_adj_matrix(size=100, n_edges=200):
    m = np.zeros((size, size))
    n_indices = (size - 1) * size // 2
    all_indices = np.arange(0, n_indices, dtype=int)
    sample_ids = np.random.choice(all_indices, size=n_edges, replace=False)
    m_ids = flat_to_real_ids(sample_ids, size=size)
    edge_weights = np.random.randint(-100, 100, size=(n_edges,))
    edge_weights[edge_weights == 0] = 50
    m[m_ids[:, 0], m_ids[:, 1]] = edge_weights
    m[m_ids[:, 1], m_ids[:, 0]] = edge_weights
    return m


def floyd_warshall(graph, V):
    """
    Arguments
    ---------
    graph   (np.ndarray) : Adjacency matrix of a graph. If node doesn't exist,
                           it has weight 0.
    V              (int) : Number of vertices in a graph
    """

    dist = graph.copy()
    dist[(dist == 0) & ~np.diagflat(np.full(V, True))] = INF

    for k in range(V):

        # pick all vertices as source one by one
        for i in range(V):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i, j]
                dist[i][j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist


def matrix_chain_order(p):
    n = len(p)
    m = np.zeros((n, n))

    # m[i, j] = Minimum number of scalar multiplications needed 
    # to compute the matrix A[i]A[i + 1]...A[j] = A[i..j] where 
    # dimension of A[i] is p[i-1] x p[i] 

    # cost is zero when multiplying one matrix. 
    for i in range(1, n):
        m[i][i] = 0

    # L is chain length. 
    for L in range(2, n):
        for i in range(1, n - L + 1):
            j = i + L - 1
            m[i][j] = INF
            for k in range(i, j):

                # Compute the cost of operation
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q

    return m[1][n - 1]


INF = 9999999

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate time dependence of algorithms")
    parser.add_argument("--random_state", type=int, default=111, help="Random state for random generator")
    parser.add_argument("--output", default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task8.csv"),
                        help="Output file")

    args = parser.parse_args()
    np.random.seed(args.random_state)

    data = []
    for V in tqdm.tqdm(range(55, 9, -5), desc="Floyd-Warshall"):
        for E in [V * 2, V * 3, V * 4]:
            graph = create_adj_matrix(size=V, n_edges=E)
            t = timeit.timeit(
                stmt=f"floyd_warshall(graph, V)", globals=globals(), number=5
            )
            data.append({"algo": "fw", "main_n": V, "n_edges": E, "time": t})

    for n_matrices in tqdm.tqdm(range(100, 4, -5), desc="Matrix chain"):
        m_chain = np.random.randint(1, 100, size=(n_matrices + 1,))
        t = timeit.timeit(stmt=f"matrix_chain_order(m_chain)", globals=globals(), number=10)
        data.append({"algo": "mc", "main_n": n_matrices, "n_edges": None, "time": t})

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Data were saved to {args.output}")
