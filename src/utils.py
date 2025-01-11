import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers, extract_feats_using_model
from extract_data import features_diff

def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):
    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_' + dataset + '.pt'
        desc_file = './data/' + dataset + '/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename=graph_id))
            fr.close()
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './data/dataset_' + dataset + '.pt'
        graph_path = './data/' + dataset + '/graph'
        desc_path = './data/' + dataset + '/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1:]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")
                # load dataset to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                        node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:, idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
                mn = min(G.number_of_nodes(), spectral_emb_dim)
                mn += 1
                x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1, 2]) / torch.sum(mask, dim=[1, 2]))  # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1, 2]) / torch.sum(mask, dim=[1, 2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1])  # (N)
    var_term = ((x - mean.view(-1, 1, 1, 1).expand_as(x)) * mask) ** 2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1]))  # (N)
    mean = mean.view(-1, 1, 1, 1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1, 1, 1, 1).expand_as(x)  # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def subgraph_augment(graph, ratio=0.8):
    """Create an augmented view of the graph via subgraph sampling"""
    # Get device of the input graph
    device = graph.edge_index.device

    num_nodes = int(graph.num_nodes * ratio)
    # Ensure at least one node is selected
    num_nodes = max(1, num_nodes)

    # Create node mask on the same device as the graph
    node_mask = torch.randperm(graph.num_nodes, device=device)[:num_nodes]

    return graph.subgraph(node_mask)


def edge_drop(graph, p=0.1):
    """Randomly drop edges with probability p"""

    edge_mask = torch.rand(graph.num_edges) > p
    return graph.edge_subgraph(edge_mask)



def preprocess_dataset_with_pretrained_embedder(dataset, n_max_nodes, spectral_emb_dim):
    data_lst = []
    if dataset == 'test':
        filename = './embedding_jina_full/jina_embed_dataset_' + dataset + '.pt'
        desc_file = './data/' + dataset + '/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print("Using preprocess_dataset_with_pretrained_embedder")
            print(f'Dataset {filename} loaded from file')

        else:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('llm_model/all-MiniLM-L6-v2')
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = model.encode([desc], convert_to_tensor=True).squeeze(0)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feats_stats = torch.tensor(feats_stats, dtype=torch.float32, device=device).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename=graph_id))
            fr.close()
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './embedding_jina_full/jina_embed_dataset_' + dataset + '.pt'
        graph_path = './data/' + dataset + '/graph'
        desc_path = './data/' + dataset + '/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1:]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")
                # load dataset to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                        node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:, idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)
                mn = min(G.number_of_nodes(), spectral_emb_dim)
                mn += 1
                x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats_using_model(model, fstats)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feats_stats = torch.tensor(feats_stats, dtype=torch.float32, device=device).unsqueeze(0)

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst

class Scaler:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, device=None):
        if device is not None:
            self.set_device(device)
        return x/ self.sigma
    def set_device(self, device):
        self.sigma = self.sigma.to(device)
        pass

from extract_data import create_features

MU_STATS = torch.Tensor([3.0606e+01, 2.2626e+02, 1.2925e+01, 1.3899e+03, 5.0697e-01, 1.1442e+01,
        3.3475e+00])
SIGMA_STATS = torch.Tensor([1.1808e+01, 2.3441e+02, 1.0175e+01, 2.7951e+03, 3.2121e-01, 9.9933e+00,
        1.4503e+00])
SCALER_STATS = Scaler(SIGMA_STATS)
SCALER_proj_STATS = Scaler(SIGMA_STATS[:5])
def MSE_reconstruction_loss(adj_matrices, num_nodes_batched, features_true):
    features_true_projected = features_true[:,:5]
    features_pred=features_diff(adj_matrices, num_nodes_batched)
    delta_normalized =  SCALER_proj_STATS(features_pred - features_true_projected, device=adj_matrices.device)
    return (delta_normalized**2).sum(dim=-1).sqrt().mean()


def MAE_reconstruction_loss(adj_matrices, num_nodes_batched, features_true, device=None):
    features_true_projected = features_true[:, :5]
    features_pred = features_diff(adj_matrices, num_nodes_batched)
    delta_normalized = SCALER_proj_STATS(features_pred - features_true_projected, device=adj_matrices.device)
    return delta_normalized.abs().sum(dim=-1).mean()


def compute_MAE(adj_matrices, num_nodes_batched, features_true):
    features_pred = torch.stack(list(map(lambda x: create_features(*x), zip(adj_matrices, num_nodes_batched))))

    delta_normalized = SCALER_STATS(features_pred - features_true, device=adj_matrices.device)
    return delta_normalized.abs().sum(dim=-1).mean()

