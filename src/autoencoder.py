import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GATConv, SAGEConv, PNAConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from utils import MAE_reconstruction_loss_normalized
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


class DecoderWithStats(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, stats_input_size = 7, stats_latent_size=64, tau=1):
        super(DecoderWithStats, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.stats_latent_size = stats_latent_size
        self.tau = tau
        print(f"Using {self.tau} as a value for tau in the Decoder")
        self.stats_layer = nn.Linear(stats_input_size,stats_latent_size)
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim + stats_latent_size, hidden_dim) for i in
                                                            range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, stats):

        stats_latent = self.stats_layer(stats)
        x = self.relu(self.mlp[0](x))

        for i in range(1, self.n_layers - 1):
            xstats = torch.cat((x, stats_latent),dim=1)
            x = self.relu(self.mlp[i](xstats))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=self.tau, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj



class GobalModificationsNN(nn.Module):
    def __init__(self, input_dim, stats_latent_size, hidden_dim, output_dim, n_layers):
        """
        Special adjacency update network gloabbly aware. (type of PointNet NN)
        """
        super(GobalModificationsNN, self).__init__()
        input_layer = nn.Sequential(nn.Linear(input_dim+stats_latent_size, hidden_dim),
                      nn.LeakyReLU(0.2),
                      nn.Linear(hidden_dim, hidden_dim))
        mlp_layers =  [input_layer] + [ nn.Sequential(nn.Linear(2*hidden_dim + stats_latent_size, hidden_dim),
                      nn.LeakyReLU(0.2),
                      nn.Linear(hidden_dim, hidden_dim))
            for i in range(n_layers - 2)]
        self.hidden_layers = nn.ModuleList(mlp_layers)
        self.last_layer = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
    def forward(self,adj, global_features:torch.Tensor):
        """
        Forward pass for GlobalModification
        """
        b,n,_ = adj.shape
        global_features = global_features.view(global_features.size(0),1,global_features.size(1)) # (b,1,stat_dim)
        flow = torch.cat((adj, global_features), dim=-1) # (b,n,n+stat_dim)
        for layer in self.hidden_layers:
            flow = layer(flow) # (b,n,h)
            aggr_features = torch.sum(flow, dim=1).unsqueeze(1)  # (b,h)
            flow = torch.cat((flow, global_features, aggr_features), dim=-1) # (b,n,n+stat_dim+h)

        output = self.last_layer(aggr_features)
        # Restore the original batch and sequence dimensions
        x = output.view(b, n, -1)
        return x


class DecoderWithStatsGlobal(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, stats_input_size = 7, stats_latent_size=64, tau=1, n_layers_last = 2):
        super(DecoderWithStatsGlobal, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.stats_latent_size = stats_latent_size
        self.tau = tau
        print(f"Using {self.tau} as a value for tau in the Decoder")
        self.stats_layer = nn.Linear(stats_input_size,stats_latent_size)
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim + stats_latent_size, hidden_dim) for i in
                                                            range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, n_nodes**2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.n_layers_last = n_layers_last
        self.last_layer_graph = torch.nn.ModuleList()
        self.last_layer_graph = GobalModificationsNN(self.n_nodes,self.stats_latent_size , self.n_nodes,self.n_nodes, self.n_layers_last )

    def forward(self, x, stats, **kwargs):

        stats_latent = self.stats_layer(stats)
        x = self.relu(self.mlp[0](x))

        for i in range(1, self.n_layers - 1):
            xstats = torch.cat((x, stats_latent),dim=1)
            x = self.relu(self.mlp[i](xstats))

        x = self.mlp[self.n_layers - 1](x)
        adj_logits = torch.reshape(x, (x.size(0), self.n_nodes, self.n_nodes))

        adj_logits = self.last_layer_graph(adj_logits)
        return self.gumbel_adj(adj_logits, **kwargs)

    def gumbel_adj(self, adj, tau=1, hard=True):

        logits = adj

        gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0, 1)

        gumbels = (logits + gumbels) / tau  # Gumbel(logits, tau)
        gumbels[:, torch.arange(self.num_max_nodes), torch.arange(self.num_max_nodes)] -=1000 #take away self loops
        y_soft = torch.sigmoid(gumbels)  # to proba
        y_soft = 1 / 2 * (y_soft + y_soft.permute(0, 2, 1)) # symmetry of adj
        if hard:
            # Hard thresholding for binary adjacency matrix
            y_hard = (y_soft > 0.5).float()  # Threshold at 0.5 for binary edges
            # Straight-through estimator for gradients
            return y_hard - y_soft.detach() + y_soft
        else:
            # Return soft adjacency matrix
            return y_soft

class GATDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, tau=1, n_graph_layers=1, heads=1, dropout=0.2):
        super(GATDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.tau = tau
        self.graph_layers = n_graph_layers
        self.heads = heads
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers - 2)],
            nn.Linear(hidden_dim, n_nodes * hidden_dim)
        )

        self.gat_convs = nn.ModuleList()
        for _ in range(n_graph_layers):
            self.gat_convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout
                )
            )

        self.last_layer = nn.Linear(hidden_dim, n_nodes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x) #(batch_size, n_nodes * hidden_dim)
        x = x.view(batch_size, self.n_nodes, -1)  # (batch_size, n_nodes, hidden_dim)

        for i in range(self.graph_layers):
            # Create edge index for a fully connected graph
            edge_index = torch.combinations(torch.arange(self.n_nodes, device=x.device), r=2).t().contiguous()
            batch_edge_index = torch.cat([edge_index + i * self.n_nodes for i in range(batch_size)], dim=1)

            x_flat = x.view(-1, x.size(-1))  # (batch_size * n_nodes, hidden_dim)
            x_flat = self.gat_convs[i](x_flat, batch_edge_index)
            x_flat = F.elu(x_flat)
            x_flat = F.dropout(x_flat, self.dropout, training=self.training)
            x = x_flat.view(batch_size, self.n_nodes, -1)  # -> (batch_size, n_nodes, hidden_dim)

        x = self.last_layer(x)  # (batch_size, n_nodes, n_nodes)

        x = F.gumbel_softmax(x, tau=self.tau, hard=True, dim=-1)
        adj = x + x.transpose(1, 2)
        adj = torch.clamp(adj, 0, 1)
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

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for layer in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = torch.nn.functional.leaky_relu(batch_norm(x), 0.2)

            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out




# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, use_pna=False, pool_type="add", aggregators=None, scalers=None, deg=None, use_decoder=None, stats_latent_size=64, n_dec_heads=1, n_dec_graph_layers=1):
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
        self.use_decoder = use_decoder
        if use_decoder is None:
            self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        elif use_decoder =="decoder_stats":
            self.decoder= DecoderWithStats(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes,stats_latent_size=stats_latent_size)
        elif use_decoder == "gat":
            self.decoder = GATDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, tau=1, n_graph_layers=n_dec_graph_layers, heads=n_dec_heads)
        else:
            raise ValueError(f"{use_decoder} decoder is not supported")

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(x_g, data.stats)
        else:
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
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(x_g, data.stats)
        else:
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
                 use_gat=False, use_pna=False, tau=1, use_pooling="add", aggregators=None, scalers=None, deg=None,use_decoder=None, stats_latent_size=64, n_dec_graph_layers=1, n_dec_heads=1):
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
        self.use_decoder = use_decoder
        if use_decoder is None:
            self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, tau=self.tau)
        elif use_decoder =="decoder_stats":
            self.decoder= DecoderWithStats(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes,stats_latent_size=stats_latent_size)
        elif use_decoder == "gat":
            self.decoder = GATDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, tau, n_graph_layers=n_dec_graph_layers, heads=n_dec_heads)
        else:
            raise ValueError(f"{use_decoder} decoder is not supported")

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
        if self.use_decoder =="decoder_stats":
            adj = self.decoder(x_g, data.stats)
        else:
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

    def decode_mu(self, mu, stats):
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(mu, stats)
        else:
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
        if self.use_decoder =="decoder_stats":
            adj = self.decoder(z, data.stats)
        else:
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
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(z, data.stats)
        else:
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
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(z, data.stats)
        else:
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
    def loss_with_mae_reg(self, data, data_aug, beta=0.05, gamma=0.005, lbd_reg = 1e-3, full_mae=False):

        infos = {}
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        if self.use_decoder=="decoder_stats":
           adj = self.decoder(z, data.stats)
        else:
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
        mse_features = MAE_reconstruction_loss(adj, num_nodes, data.stats)


        # Combined loss
        loss = recon + beta * kld + gamma * infonce + lbd_reg * mse_features
        infos["recon"] = recon
        infos["kld"] = kld
        infos["infonce"] = infonce
        infos["mae_features"] = mse_features
        infos["loss"] = loss

        return loss, infos

    def loss_with_mae_normalized_reg(self, data, data_aug, beta=0.05, gamma=0.005, lbd_reg = 1e-3, full_mae=False):

        infos = {}
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        if self.use_decoder =="decoder_stats":
            adj = self.decoder(z, data.stats)
        else:
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
        mse_features = MAE_reconstruction_loss_normalized(adj, num_nodes, data.stats)


        # Combined loss
        loss = recon + beta * kld + gamma * infonce + lbd_reg * mse_features
        infos["recon"] = recon
        infos["kld"] = kld
        infos["infonce"] = infonce
        infos["mae_normalized_features"] = mse_features
        infos["loss"] = loss

        return loss, infos