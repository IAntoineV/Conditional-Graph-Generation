import argparse

import csv
import os.path

from tqdm import tqdm

import numpy as np
from datetime import datetime
import torch


import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from autoencoder import VariationalAutoEncoder, VariationalAutoEncoderWithInfoNCE
from denoise_model import DenoiseNN, p_losses, sample
from denoise_model import p_losses_with_reg
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, subgraph_augment, edge_drop, \
    preprocess_dataset_with_pretrained_embedder, augment_graph, cosine_beta_schedule
from utils import compute_MAE, compute_normal_MAE, compute_normal_MSE

from graph_utils import get_num_nodes
from extract_data import create_features

np.random.seed(13)

def compute_inference_loop(model, dataset, name_set, batch_size=5):
    """
    broadcast one graph to batch_size and takes best graph
    """
    mae_generated_normalized = 0
    mse_generated = 0
    mae_generated =0
    test_size = 0
    mae_per_component = torch.zeros(7)
    total_generated_adj = []
    total_graph_ids = []
    total_mae = []
    bs = batch_size
    for k, data in enumerate(tqdm(dataset, desc=f"Processing {name_set} set", )):
        data = data.to(device)
        stat = data.stats.detach().cpu().expand(batch_size,-1)
        graph_ids = data.filename
        total_graph_ids.append(graph_ids)
        shape = (bs, args.latent_dim)
        x_sample = torch.randn(shape, device=device)
        # print(f"{data.stats.shape=}")
        # print(f"{data.stats.shape=}")
        adj = model.decode_mu(x_sample, data.stats.expand(batch_size, -1)).detach().cpu()
        print(f"{adj.mean(dim=0)}=")
        print(f"{adj.std(dim=0)}=")
        stat_d = torch.reshape(stat, (-1, args.n_condition)).detach().cpu()
        num_nodes = get_num_nodes(adj)
        mae_stats = compute_MAE(adj, num_nodes, stat)
        print(f"{mae_stats.shape=}")
        print(f"{mae_stats=}")
        idx_best_adj = torch.argmin(mae_stats)
        best_adj = adj[idx_best_adj].unsqueeze(0)
        total_generated_adj.append(best_adj)
        num_nodes = get_num_nodes(best_adj)
        mae_generated += compute_normal_MAE(best_adj, num_nodes ,stat)
        mse_generated +=  compute_normal_MSE(best_adj, num_nodes ,stat)
        mae_generated_normalized += compute_MAE(best_adj, num_nodes, stat)
        mae_per_component += (stat-torch.stack(list(map(lambda x: create_features(*x), zip(best_adj, num_nodes))))).abs().mean(dim=0)
        

    nb_batch = len(dataset)
    mae_generated_normalized = mae_generated_normalized / nb_batch
    mse_generated = mse_generated / nb_batch
    mae_generated = mae_generated / nb_batch
    logs_inference = f"normalized MAE : {mae_generated_normalized:.3g} \n MAE : {mae_generated:.3g} \n MSE : {mse_generated:.3g} \n MAE per component  {list(mae_per_component /nb_batch )}"

    print(logs_inference)

    print("best MAE", )
    return torch.cat(total_generated_adj, dim=0), total_graph_ids




def compute_stats_autoencoder_inference(model, loader, name_set):
    mae_generated_normalized = 0
    mse_generated = 0
    mae_generated =0
    test_size = 0
    mae_per_component = torch.zeros(7)
    total_generated_adj = []
    total_graph_ids = []

    for k, data in enumerate(tqdm(loader, desc=f"Processing {name_set} set", )):
        data = data.to(device)
        stat = data.stats.detach().cpu()
        bs = stat.size(0)
        test_size+=bs
        graph_ids = data.filename
        total_graph_ids = total_graph_ids + graph_ids
        shape = (bs, args.latent_dim)
        x_sample = torch.randn(shape, device=device)
        # print(f"{data.stats.shape=}")
        # print(f"{data.stats.shape=}")
        adj = model.decode_mu(x_sample, data.stats).detach().cpu()
        total_generated_adj.append(adj)
        stat_d = torch.reshape(stat, (-1, args.n_condition)).detach().cpu()

        num_nodes = get_num_nodes(adj)
        mae_generated += compute_normal_MAE(adj, num_nodes ,stat)
        mse_generated +=  compute_normal_MSE(adj, num_nodes ,stat)
        mae_generated_normalized += compute_MAE(adj, num_nodes, stat)
        mae_per_component += (stat-torch.stack(list(map(lambda x: create_features(*x), zip(adj, num_nodes))))).abs().mean(dim=0)

    nb_batch = len(loader)
    mae_generated_normalized = mae_generated_normalized / nb_batch
    mse_generated = mse_generated / nb_batch
    mae_generated = mae_generated / nb_batch
    logs_inference = f"normalized MAE : {mae_generated_normalized:.3g} \n MAE : {mae_generated:.3g} \n MSE : {mse_generated:.3g} \n MAE per component  {list(mae_per_component /nb_batch )}"

    print(logs_inference)
    return torch.cat(total_generated_adj, dim=0), total_graph_ids



"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0,
                    help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=512,
                    help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=1000,
                    help="Number of training epochs for the autoencoder (default: 200)")

# Training with InfoNCE loss
parser.add_argument('--train-infonce', action='store_true', default=False,
                    help="Flag to enable/disable the training of the VAE with constrastive InfoNCE loss (default: denabled)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64,
                    help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256,
                    help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32,
                    help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50,
                    help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2,
                    help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3,
                    help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10,
                    help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100,
                    help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512,
                    help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers-denoise', type=int, default=3,
                    help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=True,
                    help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_false', default=True,
                    help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128,
                    help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7,
                    help="Number of distinct condition properties used in conditional vector (default: 7)")

# Beta for loss in VAE
parser.add_argument('--beta-vae', type=float, default=1e-5, help="Beta for loss in VAE")

# Gamma for loss in VAE
parser.add_argument('--gamma-vae', type=float, default=4e-2, help="Gamma for loss in VAE")

# Flag use extracted number or text text_embedding
parser.add_argument('--use-text-embedding', action='store_true', default=False,
                    help="Flag to enable/disable use of LLM embedding (default: False)")

# Flag to use GAT Layers in the encoder
parser.add_argument('--use-gat', action='store_true', default=False, help="Flag to use GAT Layers in the encoder")

# Flag to use Principal Neighbourhood Aggregation (PNA) in the encoder
parser.add_argument('--use-pna', action='store_true', default=False, help="Flag to use PNA Convs in the encoder")

# Flag to use OneCycleLR scheduler for denoiser
parser.add_argument('--use-denoiser-onecyclelr', action='store_true', default=False,
                    help="Flag to use OneCycleLR scheduler for denoiser")

# Number of graph layers used in decoder
parser.add_argument('--n-dec-graph-layers', type=int, default=2,
                    help="Number of graph layers used in the decoder, if --use-decoder is in: ['gat', 'global']")

# If using GATDecoder, number of heads used
parser.add_argument('--n-dec-heads', type=int, default=1,
                    help="Number of heads used in GATConvs of GATDecoder")

# Weight decay denoiser AdamW
parser.add_argument('--denoiser-weight-decay', type=float, default=None, help="Weight decay for AdamW")

# Tau argument for decoder gumbel_softmax
parser.add_argument('--tau', type=float, default=1.0, help="tau argument for gumbel_softmax in Decoder")

# Type of global pooling for encoder
parser.add_argument('--use-pooling', type=str, default="add", help="Type of pooling to us in encoder")

# Use and compute MAE
# mae / mae_n / mse_n / none
parser.add_argument('--loss-use-ae', type=str, default="none", help="Type of loss to use for VAE training")

# mae / mae_n / mse_n / none
parser.add_argument('--loss-use-dn', type=str, default="none", help="Type of loss to use for denoising training")

# coef for
parser.add_argument('--lbd-reg', type=float, default=1e-3, help="coefficient scaling the feature loss")

# use_decoder = "decoder_stats", None, "global"
parser.add_argument('--use-decoder', type=str, default=None, help="Which decoder to use")

# Latent size for data.stats if use_decoder =="decoder_stats"
parser.add_argument('--stats-latent-size', type=int, default=64,
                    help="Latent size for data.stats if use_decoder =='decoder_stats' (default: 64)")

parser.add_argument('--not-use-mae', action='store_true', default=False,
                    help="Flag to enable/disable compute of MAE, usefuk when using embeddings (default: enabled)")


logs = ""
logs += f" Arguments used : \n"
args = parser.parse_args()
print(args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.use_text_embedding:
    preprocess_function = preprocess_dataset_with_pretrained_embedder
else:
    preprocess_function = preprocess_dataset
# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_function("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_function("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_function("test", args.n_max_nodes, args.spectral_emb_dim)

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f"Using date {date} for saving models and files")
# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# PNA specific
aggregators = ["mean", "min", "max", "std"]
scalers = ["identity", "amplification", "attenuation"]
deg=None

autoencoder = VariationalAutoEncoderWithInfoNCE(args.spectral_emb_dim + 1, args.hidden_dim_encoder,
                                                args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder,
                                                args.n_layers_decoder, args.n_max_nodes, use_gat=args.use_gat, use_pna=args.use_pna,
                                                tau=args.tau, use_pooling=args.use_pooling,
                                                aggregators=aggregators, scalers=scalers, deg=deg, use_decoder=args.use_decoder,
                                                stats_latent_size=args.stats_latent_size, n_dec_graph_layers=args.n_dec_graph_layers, n_dec_heads=args.n_dec_heads).to(device)
print("autoencoder:")
print(autoencoder)

date_to_load = "2025_01_14_14_59_23"

print(f"Loading autoencoder from {date_to_load}")
checkpoint = torch.load(f'models/{date_to_load}_autoencoder_infonce.pth.tar')
autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()
print("Infering on val set")
val_generated_adj, _  =compute_stats_autoencoder_inference(autoencoder, val_loader, "val")

print("Inference on test")
# test_generated_adj, graph_ids  =compute_inference_loop(autoencoder, testset, "test")
test_generated_adj, graph_ids  =compute_stats_autoencoder_inference(autoencoder, test_loader, "test")
# Save to a CSV file
with open(f"outputs/{date}_output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])

    for k, adj in enumerate(tqdm(test_generated_adj, desc='Processing test set for output generation', )):
        
        Gs_generated = construct_nx_from_adj(adj.numpy())
        # Define a graph ID
        graph_id = graph_ids[k]

        # Convert the edge list to a single string
        edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
        # Write the graph ID and the full edge list as a single row
        writer.writerow([graph_id, edge_list_text])

stat = torch.concat([data.stats  for data in test_loader], dim=0)
stat_mean= list(stat.mean(dim=0))
stat_std = list(stat.std(dim=0))
print(f"\n\n Logs inference \n")
print(f"\n Stats mean : {stat_mean} \n Stats std : {stat_std}")
print(logs)
