import argparse
from collections import deque
import os.path as osp
import pickle

import numpy as np

def flat_to_real_ids(ids, size=100):
    row_indices = np.arange(size - 1, 0, step=-1)
    cum_row_indices = np.cumsum(row_indices)
    out_ids = []
    for idx in ids:
        row_idx = (cum_row_indices <= idx).sum() - (cum_row_indices == idx).sum()
        if row_idx == 0:
            col_idx = idx + row_indices[::-1][row_idx] - 1
        else:
            col_idx = idx - cum_row_indices[row_idx - 1] + row_indices[::-1][row_idx] - 1
        out_ids.append([row_idx, col_idx])
        out_ids.append([col_idx, row_idx])
    return np.asarray(out_ids)


def create_adj_matrix(size=100, n_edges=200):
    m = np.zeros((size, size))
    n_indices = (size - 1) * size // 2
    all_indices = np.arange(0, n_indices, dtype=int)
    sample_ids = np.random.choice(all_indices, size=n_edges, replace=False)
    m_ids = flat_to_real_ids(sample_ids, size=size)
    m[m_ids[:, 0], m_ids[:, 1]] = 1
    return m


def adj_matrix_to_list(matrix):
    out_list = {}
    for row_idx, row in enumerate(matrix):
        out_list[row_idx] = set(np.where(row != 0)[0])
    return out_list


def connected_components(graph):
    seen = set()

    for root in graph.keys():
        if root not in seen:
            seen.add(root)
            component = []
            queue = deque([root])
            n_calls = 0

            while queue:
                node = queue.popleft()
                component.append(node)
                n_calls += 1
                for neighbor in graph[node]:
                    n_calls += 1
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield {"component": component, "n_calls": n_calls}


def shortest_path(graph, start, goal):
    if start == goal:
        return [start], 1
    visited = {start}
    queue = deque([(start, [])])
    n_calls = 0

    while queue:
        current, path = queue.popleft()
        visited.add(current)
        n_calls += 1
        for neighbor in graph[current]:
            n_calls += 1
            if neighbor == goal:
                return path + [current, neighbor], n_calls
            if neighbor in visited:
                continue
            queue.append((neighbor, path + [current]))
            visited.add(neighbor)
    return None, n_calls


def shortest_paths_all(graph):
    paths = []
    for i, from_vert in enumerate(list(graph.keys())[:-1]):
        for to_vert in list(graph.keys())[i + 1 :]:
            path, n_calls = shortest_path(graph, from_vert, to_vert)
            paths.append({"vertices": {from_vert, to_vert}, "path": path, "path_len": len(path) if path is not None else 0, "n_calls": n_calls})
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task5")
    parser.add_argument("--random_state", type=int, default=111, help="Random state for generator")
    parser.add_argument("--output", default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "task5.pkl"), help="Output file")

    args = parser.parse_args()

    np.random.seed(args.random_state)

    adj_matrix = create_adj_matrix(size=100, n_edges=200)
    adj_list = adj_matrix_to_list(adj_matrix)

    conn_comp = list(connected_components(adj_list))
    paths = shortest_paths_all(adj_list)

    with open(args.output, "wb") as fout:
        pickle.dump(
            {"adj_matrix": adj_matrix, "adj_list": adj_list, "conn_components": conn_comp, "shortest_paths": paths},
            fout,
        )
    print(f"Data were saved to {args.output}")
