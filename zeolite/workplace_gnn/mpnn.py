# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn.pytorch import NNConv
from dgl.nn.pytorch import Set2Set


# pylint: disable=W0221
class MPNNGNN(nn.Module):
    """MPNN.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN and returns the updated node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    """

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64,
                 edge_hidden_feats=128, num_step_message_passing=6):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats


# pylint: disable=W0221
class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=16,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)


from zeolite.data_formatted.prepare_dgl_data import LqgDataset


def evaluate(model, loader, num, device):
    error = 0
    for graphs, targets in loader:
        graphs = graphs.to(device)

        nfeat, efeat = graphs.ndata["label"], graphs.edata["voltage"]
        targets = targets.to(device)
        error += (model(graphs, nfeat, efeat) - targets).abs().sum().item()

    error = error / num
    return error


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    from torch.optim import Adam
    import torch
    from dgl.dataloading import GraphDataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import torch as th

    TARGET_INDEX = 1
    DEVICE = "cuda"
    dataset = LqgDataset(save_dir="../data_formatted/data_dgl_50k", target_index=TARGET_INDEX)
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    num_val = int(num_examples * 0.1)
    num_test = num_examples - num_train - num_val

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_train + num_test))
    val_sampler = SubsetRandomSampler(torch.arange(num_train + num_test, num_examples))
    unsup_sampler = SubsetRandomSampler(torch.arange(num_examples))

    train_loader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=5, drop_last=True)
    test_loader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=5, drop_last=True)
    val_loader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=5, drop_last=True)

    print("======== target = {} ========".format(TARGET_INDEX))
    print("train size:", len(train_loader))
    print("test size:", len(test_loader))

    # Set up directory for saving results
    model = MPNNPredictor(node_in_feats=1, edge_in_feats=3).to(DEVICE)
    print(model)

    optimizer = th.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=0.000001
    )

    best_val_error = float("inf")
    test_error = float("inf")

    for epoch in range(40):
        """Training"""
        model.train()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        iteration = 0
        sup_loss_all = 0
        unsup_loss_all = 0
        consis_loss_all = 0

        for sup_data in tqdm.tqdm(train_loader, desc=f"epoch: {epoch}"):
            sup_graph, sup_target = sup_data

            sup_graph = sup_graph.to(DEVICE)

            sup_nfeat, sup_efeat = (
                sup_graph.ndata["label"],
                sup_graph.edata["voltage"],
            )

            sup_nfeat = sup_nfeat.float().reshape(-1, 1).to(DEVICE)
            sup_efeat = sup_efeat.float().to(DEVICE)

            sup_target = sup_target.reshape(-1, 1)
            sup_target = sup_target.float().to(DEVICE)

            optimizer.zero_grad()

            sup_loss = F.mse_loss(
                model(sup_graph, sup_nfeat, sup_efeat), sup_target
            )
            # unsup_loss, consis_loss = model.unsup_forward(
            #     unsup_graph, unsup_nfeat, unsup_efeat, unsup_graph_id
            # )

            # loss = sup_loss + unsup_loss + args.reg * consis_loss
            loss = sup_loss

            loss.backward()

            sup_loss_all += sup_loss.item()
            # unsup_loss_all += unsup_loss.item()
            # consis_loss_all += consis_loss.item()

            optimizer.step()

        print(
            "Epoch: {}, Sup_Loss: {:4f}, Unsup_loss: {:.4f}, Consis_loss: {:.4f}".format(
                epoch, sup_loss_all, unsup_loss_all, consis_loss_all
            )
        )

        model.eval()

        val_error = evaluate(model, val_loader, num_val, DEVICE)
        scheduler.step(val_error)

        if val_error < best_val_error:
            best_val_error = val_error
            test_error = evaluate(model, test_loader, num_test, DEVICE)

        print(
            "Epoch: {}, LR: {}, val_error: {:.4f}, best_test_error: {:.4f}".format(
                epoch, lr, val_error, test_error
            )
        )
