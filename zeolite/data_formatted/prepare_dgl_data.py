from __future__ import annotations

import os
import os.path

import dgl
import pandas as pd
import torch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from tqdm import tqdm

from crystalgraph.utils import json_load, FilePath

from loguru import logger

NODE_SYMBOL_MAPPER = {"Si1 O4": 1}  # TODO this only works for silicate
REGRESSION_CSV = "F:\project\CrystalGraph\zeolite\data_formatted\data_zpp_50k.csv"
LQG_ID_COLUMN = "zeo_id"
LQG_FOLDER = "F:\project\CrystalGraph\zeolite\data\PBU_LQG_50k"

REGRESSION_DF = pd.read_csv(REGRESSION_CSV)


def get_regression_data(target_index: int) -> tuple[torch.Tensor, str]:
    assert target_index > 0
    data = REGRESSION_DF.iloc[:, target_index]
    return torch.from_numpy(data.values), data.name


# TODO create UQG dgl graph
# TODO populate addition isomorphic graph
def create_dgl_from_lqg(lqg_json: FilePath | dict) -> dgl.DGLGraph | None:
    if isinstance(lqg_json, str):
        assert os.path.isfile(lqg_json)
        lqg_dict = json_load(lqg_json)
    else:
        assert isinstance(lqg_json, dict)
        lqg_dict = lqg_json

    nodes = lqg_dict['nl_data']['nodes']
    edges = lqg_dict['nl_data']['links']

    df_nodes = pd.DataFrame.from_records(nodes)
    # convention: use the range index of df_nodes as node ids in dgl graph
    real_node_id_to_dgl_node_id = dict(zip(df_nodes["id"], df_nodes.index))
    edges_src_dgl = []
    edges_dst_dgl = []
    edges_voltage = []
    for e in edges:
        src_id = e['direction'][0]
        dst_id = e['direction'][1]
        src_id_dgl = real_node_id_to_dgl_node_id[src_id]
        dst_id_dgl = real_node_id_to_dgl_node_id[dst_id]
        edges_src_dgl.append(src_id_dgl)
        edges_dst_dgl.append(dst_id_dgl)
        edges_voltage.append(e['voltage'])
    edges_voltage = torch.tensor(edges_voltage)
    edges_src_dgl = torch.tensor(edges_src_dgl)
    edges_dst_dgl = torch.tensor(edges_dst_dgl)
    graph = dgl.graph(
        (edges_src_dgl, edges_dst_dgl), num_nodes=df_nodes.shape[0]
    )
    try:
        graph.ndata['label'] = torch.tensor([NODE_SYMBOL_MAPPER[n['symbol']] for n in nodes])
    except KeyError as e:
        logger.critical(f"{lqg_json}: {e.__str__()}")
        return
    graph.edata['voltage'] = edges_voltage
    return graph


class LqgDataset(DGLDataset):
    def __init__(self, target_index: int, save_dir: FilePath, n_jobs=12):
        self.target_index = target_index
        self.n_jobs = n_jobs
        super().__init__(name="Lqg", save_dir=save_dir)

    def process(self):
        regression_ids = REGRESSION_DF[LQG_ID_COLUMN].tolist()
        lqg_json_files = [
            f"{LQG_FOLDER}/{lqg_id}.json" for lqg_id in regression_ids
        ]
        with joblib_progress("Prepare dgl dataset...", total=len(lqg_json_files)):
            graphs = Parallel(n_jobs=self.n_jobs)(
                delayed(create_dgl_from_lqg)(lqg_json_file) for lqg_json_file in lqg_json_files
            )
        self.regression_data, self.regression_target_name = get_regression_data(self.target_index)
        self.graphs = []
        self.labels = []
        for g, v in zip(graphs, self.regression_data):
            if g is not None:
                self.graphs.append(g)
                self.labels.append(v)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_info.pkl')
        save_info(info_path, {
            'regression_target_name': self.regression_target_name,
            'target_index': self.target_index,
        })

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_info.pkl')
        info_dict = load_info(info_path)
        self.target_index = info_dict['target_index']
        self.regression_target_name = info_dict['regression_target_name']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_dgl_graph.bin')
        info_path = os.path.join(self.save_path, f'{self.name}_{self.target_index}_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)


def create_all_datasets():
    for target_index in tqdm(range(1, len(REGRESSION_DF.columns))):
        dataset = LqgDataset(target_index=target_index, save_dir="data_dgl_50k")
        dataset.save()


if __name__ == '__main__':
    create_all_datasets()
