import argparse

import torch as th
import torch.nn.functional as F

from model import InfoGraphS


def argument():
    parser = argparse.ArgumentParser(description="InfoGraphS")

    # data source params
    parser.add_argument(
        "--target", type=str, default="1", help="Choose regression task index"
    )
    parser.add_argument(
        "--train_num", type=int, default=20000, help="Size of training set"
    )

    # training params
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index, default:-1, using CPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="Training batch size."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=100, help="Validation batch size."
    )

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument("--wd", type=float, default=0, help="Weight decay.")

    # model params
    parser.add_argument(
        "--hid_dim", type=int, default=64, help="Hidden layer dimensionality"
    )
    parser.add_argument(
        "--reg", type=float, default=0.001, help="Regularization coefficient"
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    return args


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
    # Step 1: Prepare graph data   ===================================== #
    args = argument()
    label_keys = [args.target]
    print(args)

    dataset = LqgDataset(save_dir="../../data_formatted/data_dgl_50k", target_index=1)
    # node attr "label"
    # edge attr "voltage"

    import torch
    from dgl.dataloading import GraphDataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    num_val = int(num_examples * 0.1)
    num_test = num_examples - num_train - num_val

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_train + num_test))
    val_sampler = SubsetRandomSampler(torch.arange(num_train + num_test, num_examples))
    unsup_sampler = SubsetRandomSampler(torch.arange(num_examples))

    train_loader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=5, drop_last=False)
    test_loader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=5, drop_last=False)
    val_loader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=5, drop_last=False)
    unsup_loader = GraphDataLoader(
        dataset, sampler=unsup_sampler, batch_size=5, drop_last=False)

    print("======== target = {} ========".format(args.target))

    in_dim = 1

    # Step 2: Create model =================================================================== #
    model = InfoGraphS(in_dim, args.hid_dim)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=0.000001
    )

    # Step 4: training epochs =============================================================== #
    best_val_error = float("inf")
    test_error = float("inf")

    for epoch in range(args.epochs):
        """Training"""
        model.train()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        iteration = 0
        sup_loss_all = 0
        unsup_loss_all = 0
        consis_loss_all = 0

        for sup_data, unsup_data in zip(train_loader, unsup_loader):
            sup_graph, sup_target = sup_data
            # unsup_graph, _ = unsup_data

            sup_graph = sup_graph.to(args.device)
            # unsup_graph = unsup_graph.to(args.device)

            sup_nfeat, sup_efeat = (
                sup_graph.ndata["label"],
                sup_graph.edata["voltage"],
            )
            # unsup_nfeat, unsup_efeat, unsup_graph_id = (
            #     unsup_graph.ndata["label"],
            #     unsup_graph.edata["voltage"],
            #     unsup_graph.ndata["graph_id"],
            # )

            sup_target = sup_target
            sup_target = sup_target.to(args.device)

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

        val_error = evaluate(model, val_loader, num_val, args.device)
        scheduler.step(val_error)

        if val_error < best_val_error:
            best_val_error = val_error
            test_error = evaluate(model, test_loader, num_test, args.device)

        print(
            "Epoch: {}, LR: {}, val_error: {:.4f}, best_test_error: {:.4f}".format(
                epoch, lr, val_error, test_error
            )
        )
