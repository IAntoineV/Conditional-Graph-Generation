
import torch
import networkx as nx
from community import best_partition
from graph_utils import dense_to_edge_index, edge_index_to_dense
STATS = ["node", "edge", "degre", "triangles", "g_cluster_coef", "max_k_core", "communities"]


def get_num_triangle(adj_dense):
    """
    This function output the number of triangles of our graph. It is a differentiable function (gradient flow through it).
    :param adj_dense: torch dense adjacency matrix
    :return: torch float scalar, the number of triangles of our graph.
    """
    return (adj_dense @ adj_dense @ adj_dense).diagonal().sum() / 6


def get_mean_degree(adj_dense):
    """
    This function output the average degree in the graph. It is a differentiable function (gradient flow through it).
    :param adj_dense: torch dense adjacency matrix
    :return: torch float scalar, the average degree in the graph.
    """
    return adj_dense.sum() / adj_dense.shape[0]

def get_nb_edges(adj_dense):
    return adj_dense.sum() /2

def get_g_cluster_coef(adj_dense, num_triangle):
    """
    Formula : $$num_triangles/ ( 1/2 * sum(degree*(degree-1)) ) $$

    where * is the hadamard matrix product (component per component)
    :param adj_dense:
    :param num_triangle:
    :return:
    """
    degrees = adj_dense.sum(dim=1)
    return  6 * num_triangle / (degrees * (degrees-1)).sum()


def get_max_k_core(edge_index, num_nodes):
    """
    Get max_k_core, function tested on our train dataset. It works.
    :param edge_index:
    :param num_nodes:
    :return:
    """
    # Initialize degree for each node
    row, col = edge_index
    degree = torch.zeros(num_nodes, dtype=torch.int64)
    degree.index_add_(0, row, torch.ones_like(row))
    degree.index_add_(0, col, torch.ones_like(col))
    degree = degree / 2
    # Mask for active nodes (initially all nodes are active)
    node_mask = torch.ones(num_nodes, dtype=torch.bool)

    k = 1
    while True:
        # Find nodes to remove (degree <= k and are still active)
        nodes_to_remove = (degree <= k) & node_mask
        # If no nodes are removed and the number of remaining nodes is still >= k, we can safely increase k
        if not nodes_to_remove.any():
            remaining_nodes = node_mask.sum()  # Get the number of remaining nodes
            if remaining_nodes < k + 1:  # If remaining nodes < k+1, stop (can't form a valid k-core)
                break
            k += 1  # Increase k and continue
            continue

        # Mark the nodes to be removed
        node_mask[nodes_to_remove] = False

        # Update the edge list by removing edges involving the removed nodes
        edge_mask = node_mask[row] & node_mask[col]
        row, col = row[edge_mask], col[edge_mask]

        # Recompute degrees after removal (only for remaining nodes)
        degree = torch.zeros(num_nodes, dtype=torch.int64)
        degree.index_add_(0, row, torch.ones_like(row))
        degree.index_add_(0, col, torch.ones_like(col))
        degree = degree / 2

    return k  # Return the largest k-core


def get_nb_communities(adj):
    """
    This function output communities close at +-1 from the target value in our data. Those algorithms are only approximations of the true community.
    :param adj: adjacency matrix
    :return:
    """
    G = nx.from_numpy_array(adj.cpu().numpy())
    G = nx.from_numpy_array(adj.cpu().numpy())

    # Apply the Louvain method for community detection
    partition = best_partition(G)
    # Get the number of communities (i.e., unique values in the partition)
    num_communities = len(set(partition.values()))
    return num_communities


def create_features(adj,num_nodes, edge_index = None):
    if edge_index is None:
        edge_index = dense_to_edge_index(adj)
    features = [num_nodes, get_nb_edges(adj), get_mean_degree(adj), get_num_triangle(adj)]
    num_triangles = features[-1]
    features = features + [get_g_cluster_coef(adj, num_triangles), get_max_k_core(edge_index, num_nodes), get_nb_communities(adj)]
    return torch.Tensor(features).float()

