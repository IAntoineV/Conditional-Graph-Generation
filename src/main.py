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
parser.add_argument('--beta-vae', type=float, default=1e-6, help="Beta for loss in VAE")

# Gamma for loss in VAE
parser.add_argument('--gamma-vae', type=float, default=1e-5, help="Gamma for loss in VAE")

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

# Weight decay denoiser AdamW
parser.add_argument('--denoiser-weight-decay', type=float, default=None, help="Weight decay for AdamW")

# Tau argument for decoder gumbel_softmax
parser.add_argument('--tau', type=float, default=1.0, help="tau argument for gumbel_softmax in Decoder")

# Type of global pooling for encoder
parser.add_argument('--use-pooling', type=str, default="add", help="Type of pooling to us in encoder")

# Use and compute MAE
# mae / mae_n / mse_n / none
parser.add_argument('--loss-use-ae', type=str, default="mae", help="Type of loss to use for VAE training")

# mae / mae_n / mse_n / none
parser.add_argument('--loss-use-dn', type=str, default="mae", help="Type of loss to use for denoising training")

# coef for
parser.add_argument('--lbd-reg', type=float, default=1e-3, help="coefficient scaling the feature loss")


logs = ""
logs += f" Arguments used : \n"
args = parser.parse_args()
print(args)
for key,val in vars(args).items():
    logs+= f"{key} : {val}\n"
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

logs += f"\n\n Date : {date} \n\n"
print(f"Using date {date} for saving models and files")
# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# initialize VGAE model
aggregators = ["mean", "min", "max", "std"]
scalers = ["identity", "amplification", "attenuation"]
deg=None
if args.use_pna:
    # from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in trainset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in trainset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

autoencoder = VariationalAutoEncoderWithInfoNCE(args.spectral_emb_dim + 1, args.hidden_dim_encoder,
                                                args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder,
                                                args.n_layers_decoder, args.n_max_nodes, use_gat=args.use_gat, use_pna=args.use_pna,
                                                tau=args.tau, use_pooling=args.use_pooling,
                                                aggregators=aggregators, scalers=scalers, deg=deg).to(device)
print("autoencoder:")
print(autoencoder)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs_autoencoder,
                                                steps_per_epoch=len(train_loader), pct_start=0.1, final_div_factor=100)



best_log_training_ae = {}
logs_training_ae = {}
logs += "\n\n Autoencoder training: \n"
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
            if args.loss_use_ae == "none":
                loss, infos = autoencoder.loss_function(data, data_aug, beta=args.beta_vae,
                                                        gamma=args.gamma_vae)
            elif args.loss_use_ae == "mse":
                loss, infos = autoencoder.loss_with_mse_reg(data, data_aug, beta=args.beta_vae,
                                                            gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                train_loss_all_feats += infos["mse_features"].item()
            elif args.loss_use_ae == "mae":
                loss, infos = autoencoder.loss_with_mae_reg(data, data_aug, beta=args.beta_vae,
                                                            gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                train_loss_all_feats += infos["mae_features"].item()
            elif args.loss_use_ae == "mae_n":
                loss, infos = autoencoder.loss_with_mae_normalized_reg(data, data_aug, beta=args.beta_vae,
                                                                       gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                train_loss_all_feats += infos["mae_normalized_features"].item()
            else:
                raise KeyError('loss_use wrong key')
            
            train_loss_all_recon += infos["recon"].item()
            train_loss_all_kld += infos["kld"].item()
            train_loss_all_infonce += infos["infonce"].item()
        
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

                if args.loss_use_ae == "none":
                    loss, infos  =autoencoder.loss_function(data, data_aug, beta=args.beta_vae,
                                                                     gamma=args.gamma_vae)
                elif args.loss_use_ae == "mse_r":
                    loss, infos = autoencoder.loss_with_mse_reg(data, data_aug, beta=args.beta_vae,
                                                                     gamma=args.gamma_vae, lbd_reg=args.lbd_reg)
                           
                    val_loss_all_feats += infos["mse_features"].item()
                    val_mae += infos["mae"]
                elif args.loss_use_ae == "mae":
                    loss, infos = autoencoder.loss_with_mae_reg(data, data_aug, beta=args.beta_vae,
                                                                gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                    val_loss_all_feats += infos["mae_features"].item()
                    val_mae += infos["mae"]
                elif args.loss_use_ae == "mae_n":
                    loss, infos = autoencoder.loss_with_mae_normalized_reg(data, data_aug, beta=args.beta_vae,
                                                                gamma=args.gamma_vae, lbd_reg=args.lbd_reg)

                    val_loss_all_feats += infos["mae_normalized_features"].item()
                    val_mae += infos["mae"]
                else:
                    raise KeyError('loss_use wrong key')

                val_loss_all_recon += infos["recon"].item()
                val_loss_all_kld += infos["kld"].item()
                val_loss_all_infonce += infos["infonce"].item()
                val_loss_all += loss.item()
           
                cnt_val += 1
                val_count += torch.max(data.batch) + 1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            logs_training_ae = {
                "Epoch": epoch,
                "Train Loss": train_loss_all / cnt_train,
                "Train Recon": train_loss_all_recon / cnt_train,
                "Train KLD": train_loss_all_kld / cnt_train,
                "Train InfoNCE": train_loss_all_infonce / cnt_train,
                "Train MSE Features": train_loss_all_feats / cnt_train,
                "Val Loss": val_loss_all / cnt_val,
                "Val Recon": val_loss_all_recon / cnt_val,
                "Val KLD": val_loss_all_kld / cnt_val,
                "Val InfoNCE": val_loss_all_infonce / cnt_val,
                "Val MAE": val_mae / cnt_val,
                "Val MSE Features": val_loss_all_feats / cnt_val
            }
            print(f'{dt_t} {", ".join([f"{key} : {value:.4g}" for key,value in logs_training_ae.items()])}')

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'models/{date}_autoencoder_infonce.pth.tar')
            best_log_training_ae = logs_training_ae
    print("Taking best autoencoder to train denoiser")
    print(f"Loading autoencoder from {date}")
    for key,value in best_log_training_ae.items():
        logs+=f"\n{key} : {value:.4g}"

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


log_training_denoiser = {}
best_log_training_denoiser = {}
logs += "\n \n Diffusion : \n"
if args.not_use_mae:
    print("Warning. Won't use MAE during training. MAe won't be computed.")
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
            if args.loss_use_dn == "none":
                loss,infos = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            loss_type="huber")
            else:
                loss,infos = p_losses_with_reg(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, autoencoder.decoder, data.stats,
                            loss_type="huber", loss_to_use=args.loss_use_dn)
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
                if not args.not_use_mae:
                    if args.use_text_embedding:
                        stat = data.mae_stats
                    val_mae_feats += x_g.size(0) * compute_MAE(adj.detach().cpu(), (adj.sum(dim=2) >= 1).sum(dim=-1).detach().cpu(),
                                                stat.detach().cpu())

        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_training_denoiser = {"epoch": epoch, "Train Loss":train_loss_all / train_count, "val_loss" : val_loss_all / val_count, "val_mae" : val_mae_feats / val_count,}
        print(f'{dt_t} {", ".join([f"{key} : {value:.4g}" for key,value in log_training_denoiser.items()])}')

        scheduler.step()

        if not args.not_use_mae:
            if best_val_loss >= val_mae_feats:
                best_val_loss = val_mae_feats
                torch.save({
                    'state_dict': denoise_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'models/{date}_denoise_model.pth.tar')
        else:
            if best_val_loss >= val_loss_all:
                best_val_loss = val_loss_all
                torch.save({
                    'state_dict': denoise_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'models/{date}_denoise_model.pth.tar')

    print(f"Best Val Loss denoising : {best_val_loss / val_count}")
    print("Taking last denoising model")
    for key,val in best_log_training_denoiser.items():
        logs += f"\n {key} : {val:.4g} "
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
        nb_batch = len(val_loader)
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
            if args.use_text_embedding:
                stat = data.mae_stats
            mae_feats += x_g.size(0) * compute_MAE(adj.detach().cpu(), (adj.sum(dim=2) >= 1).sum(dim=-1).detach().cpu(),
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
    mae_generated_normalized = 0
    mse_generated = 0
    mae_generated =0
    test_size = 0

    for k, data in enumerate(tqdm(test_loader, desc='Processing test set', )):
        data = data.to(device)
        stat = data.stats.detach().cpu()
        bs = stat.size(0)
        test_size+=bs
        graph_ids = data.filename

        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas,
                         batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample).detach().cpu()
        stat_d = torch.reshape(stat, (-1, args.n_condition)).detach().cpu()

        num_nodes = get_num_nodes(adj)
        mae_generated += compute_normal_MAE(adj, num_nodes ,stat)
        mse_generated +=  compute_normal_MSE(adj, num_nodes ,stat)
        mae_generated_normalized += compute_MAE(adj, num_nodes, stat)
        mae_per_component = torch.stack(list(map(lambda x: create_features(*x), zip(adj, num_nodes)))).abs().mean(dim=0)
        for i in range(stat.size(0)):

            Gs_generated = construct_nx_from_adj(adj[i, :, :].numpy())
            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])

    nb_batch = len(test_loader)
    mae_generated_normalized = mae_generated_normalized / nb_batch
    mse_generated = mse_generated / nb_batch
    mae_generated = mae_generated / nb_batch
    logs_inference = f"normalized MAE : {mae_generated_normalized:.3g} \n MAE : {mae_generated:.3g} \n MSE : {mse_generated:.3g} \n MAE per component {list(mae_per_component)}"

stat = torch.concat([data.stats  for data in test_loader], dim=0)
stat_mean= list(stat.mean(dim=0))
stat_std = list(stat.std(dim=0))
logs += f"\n\n Logs inference \n {logs_inference}"
logs += f"\n Stats mean : {stat_mean} \n Stats std : {stat_std}"
print(logs)
if not os.path.exists("./training_logs/"):
    os.mkdir("./training_logs/")
with open(f"./training_logs/{date}_report.txt", "w") as f:
    f.write(logs)