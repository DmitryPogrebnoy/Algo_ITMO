import argparse
import os.path as osp
import pickle
import timeit
import tqdm
import itertools

import networkx as nx
import numpy as np


def dijkstra_time(graph):
    data = []
    for i, start_v in enumerate(tqdm.tqdm(list(graph.nodes)[:-1], desc="Generating Dijkstra")):
        for end_v in list(graph.nodes)[i + 1:]:
            t = timeit.timeit(stmt=f"nx.dijkstra_path(graph, {start_v}, {end_v})", globals=globals(), number=10)
            path = nx.dijkstra_path(graph, start_v, end_v)
            data.append({"vertices": [start_v, end_v], "path": path, "path_len": len(path), "time": t})
    return data


def bf_time(graph):
    data = []
    for i, start_v in enumerate(tqdm.tqdm(list(graph.nodes)[:-1], desc="Generating Bellman-Ford")):
        for end_v in list(graph.nodes)[i + 1:]:
            t = timeit.timeit(stmt=f"nx.bellman_ford_path(graph, {start_v}, {end_v})", globals=globals(), number=10)
            path = nx.bellman_ford_path(graph, start_v, end_v)
            data.append({"vertices": [start_v, end_v], "path": path, "path_len": len(path), "time": t})
    return data


def create_grid_w_obst(dimM=10, dimN=20, obst=10):
    grid = nx.generators.lattice.grid_2d_graph(dimM, dimN)
    indices_m = np.arange(dimM)
    indices_n = np.arange(dimN)
    indices = np.array(list(itertools.product(indices_m, indices_n)))
    generator = np.random.default_rng(111)
    obst_indices = generator.choice(indices, size=obst, replace=False)
    for idx in obst_indices:
        grid.remove_node(tuple(idx))
    return grid


def astar_time(grid):
    data = []
    for i, start_v in enumerate(tqdm.tqdm(list(grid.nodes)[:-1], desc="Generating A*")):
        for end_v in list(grid.nodes)[i + 1:]:
            try:
                t = timeit.timeit(stmt=f"nx.astar_path(grid, {start_v}, {end_v})", globals=globals(), number=10)
                path = nx.astar_path(grid, start_v, end_v)
            except nx.exception.NetworkXNoPath:
                t = None
                path = []
            data.append({"vertices": [start_v, end_v], "path": path, "path_len": len(path), "time": t})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task6")
    parser.add_argument("--random_state", type=int, default=111, help="Random state for generator")
    parser.add_argument("--output", default=osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",  "task6.pkl"), help="Output file")

    args = parser.parse_args()

    np.random.seed(args.random_state)

    graph = nx.generators.random_graphs.gnm_random_graph(n=100, m=500, seed=args.random_state)
    for (u, v) in graph.edges():
        graph.edges[u, v]["weight"] = np.random.randint(1, 1000)

    dijkstra_data = dijkstra_time(graph)
    bf_data = bf_time(graph)

    grid = create_grid_w_obst(10, 20, 40)
    astar_data = astar_time(grid)

    with open(args.output, "wb") as fout:
        pickle.dump(
            {"g": graph, "dijkstra": dijkstra_data, "bf": bf_data, "grid": grid, "astar": astar_data},
            fout,
        )
    print(f"Data were saved to {args.output}")
