import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GATConv, PNAConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from utils import compute_MAE
from utils import MSE_reconstruction_loss, MAE_reconstruction_loss
from graph_utils import get_num_nodes


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, tau=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.tau = tau
        print(f"Using {self.tau} as a value for tau in the Decoder")

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in
                                                            range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=self.tau, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, pool_type="add"):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                nn.LeakyReLU(0.2),
                                                nn.BatchNorm1d(hidden_dim),
                                                nn.Linear(hidden_dim, hidden_dim),
                                                nn.LeakyReLU(0.2))
                                  ))
        for layer in range(n_layers - 1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                    nn.LeakyReLU(0.2),
                                                    nn.BatchNorm1d(hidden_dim),
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.LeakyReLU(0.2))
                                      ))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        if pool_type=="mean":
            print("mean")
            self.global_pool = global_mean_pool
        elif pool_type=="max":
            print("max")
            self.global_pool = global_max_pool
        else:
            print("add")
            self.global_pool = global_add_pool

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = self.global_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out
    
# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, use_pna=False, pool_type="add", aggregators=None, scalers=None, deg=None):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        if use_pna:
            print("Using PNA_Encoder")
            self.encoder = PNAEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout=0.2, aggregators=aggregators, scalers=scalers, deg=deg)
        else:
            self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, pool_type=pool_type)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def decode_mu(self, mu):
        adj = self.decoder(mu)
        return adj

    def loss_function(self, data, beta=0.05):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)

        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kld

        return loss, recon, kld


class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads=1, dropout=0.2, use_pooling="add"):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout
            )
        )

        for layer in range(n_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout
                )
            )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.use_pooling = use_pooling
        print(f"Using pooling {self.use_pooling}")

        if use_pooling=="mean":
            print("mean")
            self.global_pool = global_mean_pool
        elif use_pooling=="max":
            print("max")
            self.global_pool = global_max_pool
        elif use_pooling=="add":
            print("add")
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"{self.use_pooling} pooling is not supported")


    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        out = self.global_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class PNAEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, aggregators=None, scalers=None, deg=None, pool_type="add"):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(PNAConv(input_dim, hidden_dim, aggregators=aggregators, scalers=scalers, deg=deg))
        for layer in range(n_layers - 1):
            self.convs.append(PNAConv(hidden_dim, hidden_dim, aggregators=aggregators, scalers=scalers, deg=deg))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
        if pool_type=="mean":
            print("mean")
            self.global_pool = global_mean_pool
        elif pool_type=="max":
            print("max")
            self.global_pool = global_max_pool
        else:
            print("add")
            self.global_pool = global_add_pool


    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = self.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)

        out = self.global_pool(x, data.batch)

        out = self.bn(out)
        out = self.fc(out)
        return out
    

class VariationalAutoEncoderWithInfoNCE(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes,
                 use_gat=False, use_pna=False, tau=1, use_pooling="add", aggregators=None, scalers=None, deg=None):
        super(VariationalAutoEncoderWithInfoNCE, self).__init__()
        print("Training VariationalAutoEncoderWithInfoNCE")
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.tau = tau
        if use_gat:
            print("Using GAT_Encoder")
            self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, heads=2, dropout=0.2, use_pooling=use_pooling)
        elif use_pna:
            print("Using PNA_Encoder")
            self.encoder = PNAEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout=0.2, aggregators=aggregators, scalers=scalers, deg=deg)
        else:
            self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, tau=self.tau)

        # Projection head for InfoNCE loss
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(hidden_dim_enc, latent_dim)
        )

        print(f"Number of params encoder : {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}")
        print(f"Number of params Decoder : {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}")
        print(f"Number of params logvar : {sum(p.numel() for p in self.fc_logvar.parameters() if p.requires_grad)}")

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def decode_mu(self, mu):
        adj = self.decoder(mu)
        return adj

    def infonce_loss(self, z_i, z_j, temperature=0.07):
        """
        Compute InfoNCE loss for the batch
        z_i, z_j are batches of latent vectors from two augmented views
        """
        # Project the vectors
        z_i = self.projection(z_i)
        z_j = self.projection(z_j)

        # Normalize the vectors
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / temperature

        # Labels are all diagonal elements (positive pairs)
        labels = torch.arange(z_i.size(0), device=z_i.device)

        # Compute InfoNCE loss
        loss_i = F.cross_entropy(similarity_matrix, labels)
        loss_j = F.cross_entropy(similarity_matrix.T, labels)

        return (loss_i + loss_j) / 2

    def get_beta(self, epoch, max_epochs, beta_max=0.05):
        return min(epoch / (max_epochs * 0.2), 1.0) * beta_max

    def metrics(self, data, data_aug, beta=0.05, gamma=0.005, full_mae=False):
        infos = {}
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)
        adj_cpu = adj.detach().cpu()
        num_nodes = get_num_nodes(adj_cpu)
        if full_mae:
            mae = compute_MAE(adj_cpu, num_nodes, data.stats.detach().cpu())
        else:
            mae = MAE_reconstruction_loss(adj_cpu, num_nodes, data.stats.detach().cpu())
        infos["mae"] = mae
        # Augmented data encodings
        x_g_aug = self.encoder(data_aug)
        mu_aug = self.fc_mu(x_g_aug)
        logvar_aug = self.fc_logvar(x_g_aug)
        z_aug = self.reparameterize(mu_aug, logvar_aug)

        # Original VAE losses
        # recon = F.l1_loss(adj, data.A, reduction='mean') +  F.binary_cross_entropy(adj, data.A, reduction='mean')
        recon = F.mse_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # InfoNCE loss
        infonce = self.infonce_loss(z, z_aug)

        # Combined loss
        loss = recon + beta * kld + gamma * infonce
        infos["recon"] = recon
        infos["kld"] = kld
        infos["infonce"] = infonce
        infos["loss"] = loss
        return loss, infos

    def loss_function(self, data, data_aug, beta=0.05, gamma=0.005):
        infos = {}
        # Original VAE encodings
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)

        # Augmented data encodings
        x_g_aug = self.encoder(data_aug)
        mu_aug = self.fc_mu(x_g_aug)
        logvar_aug = self.fc_logvar(x_g_aug)
        z_aug = self.reparameterize(mu_aug, logvar_aug)

        # Original VAE losses
        # recon = F.l1_loss(adj, data.A, reduction='mean') +  F.binary_cross_entropy(adj, data.A, reduction='mean')
        recon = F.mse_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # InfoNCE loss
        infonce = self.infonce_loss(z, z_aug)

        # Combined loss
        loss = recon + beta * kld + gamma * infonce
        infos["recon"] = recon
        infos["kld"] = kld
        infos["infonce"] = infonce
        infos["loss"] = loss
        return loss, infos


    def loss_with_mse_reg(self, data, data_aug, beta=0.05, gamma=0.005, lbd_reg = 1e-3, full_mae=False):

        infos = {}
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)

        adj_cpu = adj.detach().cpu()
        num_nodes = get_num_nodes(adj_cpu)
        if full_mae:
            mae = compute_MAE(adj_cpu, num_nodes, data.stats.detach().cpu())
        else:
            mae = MAE_reconstruction_loss(adj_cpu, num_nodes, data.stats.detach().cpu())
        infos["mae"]=mae
        # Augmented data encodings
        x_g_aug = self.encoder(data_aug)
        mu_aug = self.fc_mu(x_g_aug)
        logvar_aug = self.fc_logvar(x_g_aug)
        z_aug = self.reparameterize(mu_aug, logvar_aug)

        # Original VAE losses
        # recon = F.l1_loss(adj, data.A, reduction='mean') +  F.binary_cross_entropy(adj, data.A, reduction='mean')
        recon = F.mse_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # InfoNCE loss
        infonce = self.infonce_loss(z, z_aug)

        num_nodes = get_num_nodes(adj)
        mse_features = MSE_reconstruction_loss(adj, num_nodes, data.stats)


        # Combined loss
        loss = recon + beta * kld + gamma * infonce + lbd_reg * mse_features
        infos["recon"] = recon
        infos["kld"] = kld
        infos["infonce"] = infonce
        infos["mse_features"] = mse_features
        infos["loss"] = loss

        return loss, infos
