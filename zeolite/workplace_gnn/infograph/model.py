import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv, Set2Set
from torch.nn import GRU, Linear, ReLU, Sequential

from utils import global_global_loss_, local_global_loss_

""" Feedforward neural network"""


class FeedforwardNetwork(nn.Module):
    """
    3-layer feed-forward neural networks with jumping connections
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(feat):
        feat: Tensor
            [N * D], input features
    """

    def __init__(self, in_dim, hid_dim):
        super(FeedforwardNetwork, self).__init__()

        self.block = Sequential(
            Linear(in_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
        )

        self.jump_con = Linear(in_dim, hid_dim)

    def forward(self, feat):
        block_out = self.block(feat)
        jump_out = self.jump_con(feat)

        out = block_out + jump_out

        return out


""" Semisupervised Setting """


class NNConvEncoder(nn.Module):
    """
    Encoder based on dgl.nn.NNConv & GRU & dgl.nn.set2set pooling
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(graph, nfeat, efeat):
        graph: DGLGraph
        nfeat: Tensor
            [N * D1], node features
        efeat: Tensor
            [E * D2], edge features
    """

    def __init__(self, in_dim, hid_dim):
        super(NNConvEncoder, self).__init__()

        self.lin0 = Linear(in_dim, hid_dim)

        # mlp for edge convolution in NNConv
        block = Sequential(
            Linear(5, 128), ReLU(), Linear(128, hid_dim * hid_dim)
        )

        self.conv = NNConv(
            hid_dim,
            hid_dim,
            edge_func=block,
            aggregator_type="mean",
            residual=False,
        )
        self.gru = GRU(hid_dim, hid_dim)

        # set2set pooling
        self.set2set = Set2Set(hid_dim, n_iters=3, n_layers=1)

    def forward(self, graph, nfeat, efeat):
        out = F.relu(self.lin0(nfeat))
        h = out.unsqueeze(0)

        feat_map = []

        # Convolution layer number is 3
        for i in range(3):
            m = F.relu(self.conv(graph, out, efeat))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)

        out = self.set2set(graph, out)

        # out: global embedding, feat_map[-1]: local embedding
        return out, feat_map[-1]


class InfoGraphS(nn.Module):
    """
    InfoGraph* model for semi-supervised setting
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(graph):
        graph: DGLGraph

    unsupforward(graph):
        graph: DGLGraph

    """

    def __init__(self, in_dim, hid_dim):
        super(InfoGraphS, self).__init__()

        self.sup_encoder = NNConvEncoder(in_dim, hid_dim)
        self.unsup_encoder = NNConvEncoder(in_dim, hid_dim)

        self.fc1 = Linear(2 * hid_dim, hid_dim)
        self.fc2 = Linear(hid_dim, 1)

        # unsupervised local discriminator and global discriminator for local-global infomax
        self.unsup_local_d = FeedforwardNetwork(hid_dim, hid_dim)
        self.unsup_global_d = FeedforwardNetwork(2 * hid_dim, hid_dim)

        # supervised global discriminator and unsupervised global discriminator for global-global infomax
        self.sup_d = FeedforwardNetwork(2 * hid_dim, hid_dim)
        self.unsup_d = FeedforwardNetwork(2 * hid_dim, hid_dim)

    def forward(self, graph, nfeat, efeat):
        sup_global_emb, sup_local_emb = self.sup_encoder(graph, nfeat, efeat)

        sup_global_pred = self.fc2(F.relu(self.fc1(sup_global_emb)))
        sup_global_pred = sup_global_pred.view(-1)

        return sup_global_pred

    def unsup_forward(self, graph, nfeat, efeat, graph_id):
        sup_global_emb, sup_local_emb = self.sup_encoder(graph, nfeat, efeat)
        unsup_global_emb, unsup_local_emb = self.unsup_encoder(
            graph, nfeat, efeat
        )

        g_enc = self.unsup_global_d(unsup_global_emb)
        l_enc = self.unsup_local_d(unsup_local_emb)

        sup_g_enc = self.sup_d(sup_global_emb)
        unsup_g_enc = self.unsup_d(unsup_global_emb)

        # Calculate loss
        unsup_loss = local_global_loss_(l_enc, g_enc, graph_id)
        con_loss = global_global_loss_(sup_g_enc, unsup_g_enc)

        return unsup_loss, con_loss
