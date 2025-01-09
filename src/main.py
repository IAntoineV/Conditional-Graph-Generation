import argparse

import csv

from tqdm import tqdm

import numpy as np
from datetime import datetime
import torch


import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder, VariationalAutoEncoderWithInfoNCE
from denoise_model import DenoiseNN, p_losses, sample
from src.denoise_model import p_losses_with_features_reg
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, subgraph_augment, edge_drop, \
    preprocess_dataset_with_pretrained_embedder
from utils import compute_MAE


np.random.seed(13)

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
parser.add_argument('--lr', type=float, default=5e-3,
                    help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.2,
                    help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256,
                    help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200,
                    help="Number of training epochs for the autoencoder (default: 200)")

# Training with InfoNCE loss
parser.add_argument('--train-infonce', action='store_true', default=True,
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
parser.add_argument('--n-layers-encoder', type=int, default=8,
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
parser.add_argument('--timesteps', type=int, default=700, help="Number of timesteps for the diffusion (default: 500)")

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
parser.add_argument('--beta-vae', type=float, default=1e-6, help="Beta for loss in VAE")

# Gamma for loss in VAE
parser.add_argument('--gamma-vae', type=float, default=1e-5, help="Gamma for loss in VAE")

# Flag use extracted number or text text_embedding
parser.add_argument('--use-text-embedding', action='store_true', default=False,
                    help="Flag to enable/disable use of LLM embedding (default: False)")

# Flag to use GAT Layers in the encoder
parser.add_argument('--use-gat', action='store_true', default=False, help="Flag to use GAT Layers in the encoder")

# Flag to use OneCycleLR scheduler for denoiser
parser.add_argument('--use-denoiser-onecyclelr', action='store_true', default=False,
                    help="Flag to use OneCycleLR scheduler for denoiser")

# Weight decay denoiser AdamW
parser.add_argument('--denoiser-weight-decay', type=float, default=None, help="Weight decay for AdamW")

# Tau argument for decoder gumbel_softmax
parser.add_argument('--tau', type=float, default=1.0, help="tau argument for gumbel_softmax in Decoder")

# Type of global pooling for encoder
parser.add_argument('--use-pooling', type=str, default="add", help="Type of pooling to us in encoder")


# coef for
parser.add_argument('--lbd-reg', type=float, default=1e-2, help="coefficient scaling the feature loss")

args = parser.parse_args()

print(f"{args=}")
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

# initialize VGAE model
autoencoder = VariationalAutoEncoderWithInfoNCE(args.spectral_emb_dim + 1, args.hidden_dim_encoder,
                                                args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder,
                                                args.n_layers_decoder, args.n_max_nodes, use_gat=args.use_gat,
                                                tau=args.tau, use_pooling=args.use_pooling).to(device)
print("autoencoder:")
print(autoencoder)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs_autoencoder,
                                                steps_per_epoch=len(train_loader), pct_start=0.1, final_div_factor=100)

# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        train_loss_all_infonce = 0
        train_loss_all_feats = 0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            # Create augmented view
            data_aug = edge_drop(data)
            data_aug = data_aug.to(device)

            optimizer.zero_grad()
            # beta = autoencoder.get_beta(epoch, max_epochs=args.epochs_autoencoder)
            # Updated loss function now returns InfoNCE loss as well
            loss, infos= autoencoder.loss_with_mse_reg(data, data_aug, beta=args.beta_vae,
                                                                  gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

            train_loss_all_recon += infos["recon"].item()
            train_loss_all_kld += infos["kld"].item()
            train_loss_all_infonce += infos["infonce"].item()
            train_loss_all_feats += infos["mse_features"].item()
            cnt_train += 1

            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch) + 1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0
        val_loss_all_infonce = 0
        val_loss_all_feats = 0
        val_mae = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                # Create augmented view for validation as well
                data_aug = edge_drop(data)
                data_aug = data_aug.to(device)

                loss, infos = autoencoder.loss_with_mse_reg(data, data_aug, beta=args.beta_vae,
                                                                     gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                val_loss_all_recon += infos["recon"].item()
                val_loss_all_kld += infos["kld"].item()
                val_loss_all_infonce += infos["infonce"].item()
                val_loss_all_feats += infos["mse_features"].item()
                val_loss_all += loss.item()
                val_mae += infos["mae"]
                cnt_val += 1
                val_count += torch.max(data.batch) + 1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'{dt_t} Epoch: {epoch:04d}, '
                  f'Train Loss: {train_loss_all / cnt_train:.5f}, '
                  f'Train Recon: {train_loss_all_recon / cnt_train:.2f}, '
                  f'Train KLD: {train_loss_all_kld / cnt_train:.2f}, '
                  f'Train InfoNCE: {train_loss_all_infonce / cnt_train:.2f}, '
                  f'Train MSE Features : {train_loss_all_feats / cnt_train:.2f}, '
                  f'Val Loss: {val_loss_all / cnt_val:.5f}, '
                  f'Val Recon: {val_loss_all_recon / cnt_val:.2f}, '
                  f'Val KLD: {val_loss_all_kld / cnt_val:.2f}, '
                  f'Val InfoNCE: {val_loss_all_infonce / cnt_val:.2f}, '
                  f'Val MAE : {val_mae / cnt_val:.2f}',
                    f'Val MSE Features : {val_loss_all_feats / cnt_val:.2f}')

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'models/{date}_autoencoder_infonce.pth.tar')
    print("Taking best autoencoder to train denoiser")
    print(f"Loading autoencoder from {date}")
    checkpoint = torch.load(f'models/{date}_autoencoder_infonce.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])
else:
    date_to_load = "2025_01_04_14_34_17"
    print(f"Loading autoencoder from {date_to_load}")
    checkpoint = torch.load(f'models/{date_to_load}_autoencoder_infonce.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()

# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise,
                          n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
if args.denoiser_weight_decay is not None:
    print("Using weight decay {args.denoiser_weight_decay} and AdamW")
    optimizer = torch.optim.AdamW(denoise_model.parameters(), lr=args.lr, weight_decay=args.denoiser_weight_decay)
else:
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
if args.use_denoiser_onecyclelr:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs_denoise,
                                                    steps_per_epoch=len(train_loader), pct_start=0.1,
                                                    final_div_factor=100)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise + 1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss,infos = p_losses_with_features_reg(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod,
                                                    sqrt_one_minus_alphas_cumprod, autoencoder.decoder, data.stats,
                            loss_type="huber", lbd_reg=args.lbd_reg)
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        val_mae_feats = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss,infos = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                                loss_type="huber")
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)

                # compute MAE
                stat = data.stats
                bs = stat.size(0)
                samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps,
                                 betas=betas, batch_size=bs)
                x_sample = samples[-1]
                adj = autoencoder.decode_mu(x_sample)
                val_mae_feats += compute_MAE(adj.detach().cpu(), (adj.sum(dim=2) >= 1).sum(dim=-1).detach().cpu(),
                                             stat.detach().cpu())

        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Val MAE: {:.5f} '.format(dt_t, epoch,
                                                                                        train_loss_all / train_count,
                                                                                        val_loss_all / val_count,
                                                                                        val_mae_feats / val_count))

        scheduler.step()

        if best_val_loss >= val_mae_feats:
            best_val_loss = val_mae_feats
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'models/{date}_denoise_model.pth.tar')
    print(f"Best Val Loss denoising : {best_val_loss / val_count}")
    print("Taking last denoising model")
    # print("Taking best denoiser to generate test data")
    # checkpoint = torch.load(f'models/{date}_denoise_model.pth.tar')
    # denoise_model.load_state_dict(checkpoint['state_dict'])
else:
    date_denoiser_to_load = "2025_01_04_14_34_17"
    print(f"Loading denoiser from {date_denoiser_to_load}")
    checkpoint = torch.load(f'models/{date_denoiser_to_load}_denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

    print("Computing denoising val loss with selected autoencoder and denoiser")
    autoencoder.eval()
    denoise_model.eval()
    with torch.no_grad():
        val_loss_all = 0
        val_count = 0
        mae_feats = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss,infos = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

            stat = data.stats
            bs = stat.size(0)

            samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps,
                             betas=betas, batch_size=bs)
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample)
            mae_feats += compute_MAE(adj.detach().cpu(), (adj.sum(dim=2) >= 1).sum(dim=-1).detach().cpu(),
                                     stat.detach().cpu())

    print('Val Loss: {:.5f}'.format(val_loss_all / val_count))

    print(f"MAE feats sur  val : {mae_feats / val_count}")

denoise_model.eval()

del train_loader, val_loader

# Save to a CSV file
with open(f"outputs/{date}_output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set', )):
        data = data.to(device)

        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas,
                         batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample)
        stat_d = torch.reshape(stat, (-1, args.n_condition))

        for i in range(stat.size(0)):

            Gs_generated = construct_nx_from_adj(adj[i, :, :].detach().cpu().numpy())
            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])