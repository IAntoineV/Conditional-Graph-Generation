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
        gumbels[:, torch.arange(self.num_max_nodes), torch.arange(self.num_max_nodes)] -= 100
        # Softmax is applied elementwise to the adjacency matrix
        y_soft = torch.sigmoid(gumbels)  # Sigmoid ensures probabilistic edge weights remain in [0, 1]
        #y_soft[:, torch.arange(self.num_max_nodes), torch.arange(self.num_max_nodes)] *= 0
        y_soft = 1/2 * ( y_soft + y_soft.permute(0, 2, 1))
        if hard:
            # Hard thresholding for binary adjacency matrix
            y_hard = (y_soft > 0.5).float()  # Threshold at 0.5 for binary edges
            # Straight-through estimator for gradients
            return y_hard - y_soft.detach() + y_soft
        else:
            # Return soft adjacency matrix
            return y_soft


class GraphGenerator2(torch.nn.Module):
    def __init__(self, num_nodes_batched, device, tau=1):
        """
        Initialize the GraphGenerator with a batch of logit matrices.
        """
        super().__init__()
        self.num_nodes_batched = num_nodes_batched.to(device)
        self.nb_adj = len(num_nodes_batched)
        self.num_max_nodes = torch.max(self.num_nodes_batched).item()
        self.device = device
        self.tau = tau

        # Create a single learnable tensor for all adjacency matrices
        self.adj_logits = torch.nn.Parameter(
            torch.randn(self.nb_adj, self.num_max_nodes * (self.num_max_nodes - 1) // 2, 2, device=device)
        )

    def generate_graph(self, tau=None, hard=True):
        if tau is None:
            tau = self.tau

        # Reshape logits and apply Gumbel-Softmax
        x = self.adj_logits
        x = F.gumbel_softmax(x, tau=tau, hard=hard)[:, :, 0]  # Use the first output of Gumbel-Softmax

        # Create adjacency matrices
        adj = torch.zeros(self.nb_adj, self.num_max_nodes, self.num_max_nodes, device=x.device)
        idx = torch.triu_indices(self.num_max_nodes, self.num_max_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + adj.transpose(1, 2)  # Make adjacency symmetric
        mask = torch.zeros(self.nb_adj, self.num_max_nodes, self.num_max_nodes, device=x.device, dtype=torch.bool)
        for k, num_node in enumerate(self.num_nodes_batched):
            mask[k, :num_node, :num_node] = 1
        adj = adj * mask
        return adj



class GraphGenerator3(torch.nn.Module):
    def __init__(self, num_nodes_batched, num_communities_batched, device, tau=1):
        """
        Initialize the GraphGenerator with a batch of logit matrices.
        """
        super().__init__()
        self.num_nodes_batched = num_nodes_batched.to(device)
        self.nb_adj = num_communities_batched.sum().int()
        self.num_max_nodes = torch.max(self.num_nodes_batched).item()
        self.device = device
        self.tau = tau

        # Create a single learnable tensor for all adjacency matrices
        self.adj_logits = torch.nn.Parameter(
            torch.randn(self.nb_adj, self.num_max_nodes * (self.num_max_nodes - 1) // 2, 2, device=device)
        )
        self.nb_graphs = len(num_communities_batched)
        batch_indices = torch.arange(self.nb_graphs)
        # Repeat each batch index based on the values in num_communities_tensor
        self.batch = batch_indices.repeat_interleave(num_communities_batched).to(self.device)
        assert len(self.batch) == self.nb_adj, "should always be true"

    def generate_graph(self, tau=None, hard=True):
        if tau is None:
            tau = self.tau

        # Reshape logits and apply Gumbel-Softmax
        x = self.adj_logits
        x = F.gumbel_softmax(x, tau=tau, hard=hard)[:, :, 0]  # Use the first output of Gumbel-Softmax

        # Create adjacency matrices
        adj = torch.zeros(self.nb_adj, self.num_max_nodes, self.num_max_nodes, device=x.device)
        idx = torch.triu_indices(self.num_max_nodes, self.num_max_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + adj.transpose(1, 2)  # Make adjacency symmetric

        return adj
    def regroup(self, adjs_subgraph):
        adjs = torch.zeros((self.nb_graphs, self.num_max_nodes, self.num_max_nodes), dtype=torch.float).to(
            adjs_subgraph.device)
        adjs.index_add_(0, self.batch, adjs_subgraph)

        mask = torch.zeros(self.nb_graphs, self.num_max_nodes, self.num_max_nodes, device=adjs.device, dtype=torch.bool)
        for k, num_node in enumerate(self.num_nodes_batched):
            mask[k, :num_node, :num_node] = 1
        adjs = adjs * mask
        return adjs