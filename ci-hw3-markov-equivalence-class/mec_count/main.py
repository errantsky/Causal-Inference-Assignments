from functools import reduce, partial
from operator import mul
import numpy as np
from networkx import connected_component_subgraphs, adjacency_matrix, from_numpy_matrix


def get_essential_size(essential_graph):
    adj_mat = adjacency_matrix(essential_graph).toarray()

    undirected = adj_mat * adj_mat.transpose()
    undirected = np.tril(undirected)

    return reduce(
        mul,
        map(
            SizeMEC, list(connected_component_subgraphs(from_numpy_matrix(undirected)))
        ),
    )


def ChainCom(uccg, vertex):
    A = {vertex}
    B = set(uccg.nodes)
    B.remove(vertex)

    while len(B) != 0:
        T = {w for w in B if any([w in uccg.neighbors(a) for a in A])}

        for t in T:
            for c in A:
                if (t, c) in uccg.edges():
                    uccg.remove_edge(t, c)

        any_edge_was_oriented = True
        while any_edge_was_oriented:
            any_edge_was_oriented = False
            for (y, z) in uccg.subgraph(T).edges():
                for x in uccg.nodes():
                    if (
                        (x, y) in uccg.edges()
                        and (y, x) not in uccg.edges()
                        and (y, z) in uccg.edges()
                        and (z, y) in uccg.edges()
                        and (x, z) not in uccg.edges()
                        and (z, x) not in uccg.edges()
                    ):
                        uccg.remove_edge(z, y)
                        any_edge_was_oriented = True

        A = T
        B -= T

    adj_mat = adjacency_matrix(uccg).toarray()

    undirected = adj_mat * adj_mat.transpose()
    undirected = np.tril(undirected)

    return list(connected_component_subgraphs(from_numpy_matrix(undirected)))


def SizeMEC(uccg):
    num_nodes = uccg.number_of_nodes()
    num_edges = uccg.number_of_edges()

    if num_edges == num_nodes - 1:
        return num_nodes
    elif num_edges == num_nodes:
        return 2 * num_nodes
    elif num_edges == num_nodes * (num_nodes - 1) / 2:
        return np.math.factorial(num_nodes)
    elif num_edges == num_nodes * (num_nodes - 1) / 2 - 1:
        return 2 * np.math.factorial(num_nodes - 1) - np.math.factorial(num_nodes - 2)
    elif num_edges == num_nodes * (num_nodes - 1) / 2 - 2:
        return (num_nodes ** 2 - num_nodes - 4) * np.math.factorial(num_nodes - 3)

    sj_list = []
    for node in range(num_nodes):
        chain_components = ChainCom(uccg.to_directed(), node)
        sj_list.append(reduce(mul, map(SizeMEC, chain_components)))

    return sum(sj_list)
