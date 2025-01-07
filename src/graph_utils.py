
import torch

def dense_to_edge_index(adj_matrix):
    # Get the indices of non-zero elements in the adjacency matrix
    edge_indices = torch.nonzero(adj_matrix)

    # Convert the result to a 2xN tensor (source, target)
    return edge_indices.t()


def edge_index_to_dense(edge_index, num_nodes):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)).to_dense()