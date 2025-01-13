import numpy as np
import torch
from extract_data import get_num_triangle_batched



def generate_graph_node_triangles(num_nodes_batched, num_triangles_batched, proba_list, max_nb_nodes=None):
    batch_size = num_nodes_batched.shape[0]
    if max_nb_nodes is None:
        max_nb_nodes = torch.max(num_nodes_batched).item()
    max_nb_nodes = int(max_nb_nodes)
    # Create mask
    mask = torch.zeros((batch_size, max_nb_nodes, max_nb_nodes), dtype=torch.bool)
    for i, num_nodes in enumerate(num_nodes_batched):
        mask[i, :int(num_nodes), :int(num_nodes)] = 1
    adj = torch.zeros((batch_size, max_nb_nodes, max_nb_nodes))

    done = torch.zeros((batch_size,), dtype=torch.bool)
    for k, p in enumerate(proba_list):
        # Generate random edges
        edge_to_add = mask * torch.bernoulli(torch.full(mask.shape, p))
        adj = torch.maximum(adj, edge_to_add)

        # Ensure symmetry
        adj = torch.maximum(adj, adj.permute(0, 2, 1))
        adj[:, np.arange(max_nb_nodes), np.arange(max_nb_nodes)] = 0
        # Calculate number of triangles
        nb_triangles_gen = get_num_triangle_batched(adj)

        # Update mask and done tracker
        for i in range(batch_size):
            if nb_triangles_gen[i] > num_triangles_batched[i]:
                done[i] = True
                mask[i] = 0

        if torch.all(done):
            print(f"quick break at {k}th iteration")
            break

    return adj


import torch
import torch.nn.functional as F

class GraphGenerator(torch.nn.Module):
    def __init__(self, num_nodes_batched, device, tau=1):
        """
        Initialize the GraphGenerator with a batch of logit matrices.
        """
        super().__init__()
        self.num_nodes_batched = num_nodes_batched.to(device)
        self.nb_adj = len(num_nodes_batched)
        self.num_max_nodes=  torch.max(self.num_nodes_batched).item()
        # Create a learnable logit matrix for each graph in the batch
        self.adjs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(num_nodes, num_nodes))
            for num_nodes in num_nodes_batched
        ]).to(device)

        self.device = device
        self.tau = tau
    def edge_probabilities(self):
        """
        Compute edge probabilities using a sigmoid function.
        """
        return [torch.sigmoid(adj) for adj in self.adjs]

    @property
    def adj_padded(self):
        adj_padded = torch.zeros((self.nb_adj, self.num_max_nodes, self.num_max_nodes)) - 50
        for k, (num_node, adj) in enumerate(zip(self.num_nodes_batched, self.adjs)):
            adj_padded[k, :num_node, :num_node] = adj
        adj_padded.to(self.device)
        return adj_padded
    def generate_graph(self, tau=1, hard=True):
        logits = self.adj_padded
        # Generate Gumbel noise for each element in the matrix
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0, 1)

        # Apply Gumbel-Softmax transformation
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)

        # Softmax is applied elementwise to the adjacency matrix
        y_soft = torch.sigmoid(gumbels)  # Sigmoid ensures probabilistic edge weights remain in [0, 1]
        #y_soft[:, np.arange(self.num_max_nodes), np.arange(self.num_max_nodes)] *= 0
        y_soft = 1/2 * ( y_soft + y_soft.permute(0, 2, 1))
        if hard:
            # Hard thresholding for binary adjacency matrix
            y_hard = (y_soft > 0.5).float()  # Threshold at 0.5 for binary edges
            # Straight-through estimator for gradients
            return y_hard - y_soft.detach() + y_soft
        else:
            # Return soft adjacency matrix
            return y_soft