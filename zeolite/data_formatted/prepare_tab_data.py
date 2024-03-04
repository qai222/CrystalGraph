from __future__ import annotations

import glob
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from crystalgraph.features import LqgFeatureSet
from crystalgraph.qgraph import LQG
from crystalgraph.utils import json_load, FilePath, tqdm_joblib, json_dump

REGRESSION_CSV = "F:\project\CrystalGraph\zeolite\data_formatted\data_zpp_50k.csv"
LQG_ID_COLUMN = "zeo_id"
LQG_FOLDER = "F:\project\CrystalGraph\zeolite\data\PBU_LQG_50k"

REGRESSION_DF = pd.read_csv(REGRESSION_CSV)
# REGRESSION_DF = REGRESSION_DF.head(1000)  # test

ALLOWED_NODE_SYMBOL = {"Si1 O4"}


def load_lqg(json_file: FilePath) -> LQG:
    d = json_load(json_file)
    return LQG.from_dict(d)


def load_data(n_jobs=1) -> tuple[list[LQG], list[dict]]:
    regression_ids = REGRESSION_DF[LQG_ID_COLUMN].tolist()
    lqg_json_files = [
        f"{LQG_FOLDER}/{lqg_id}.json" for lqg_id in regression_ids
    ]
    if n_jobs == 1:
        graphs = []
        for jf in tqdm(lqg_json_files):
            graphs.append(load_lqg(jf))
    else:
        with tqdm_joblib(tqdm(desc="load lqgs...", total=len(lqg_json_files))) as progress_bar:
            graphs = Parallel(n_jobs=5)(delayed(load_lqg)(jf) for jf in lqg_json_files)

    prop_records = []
    gs = []
    for g, rid, r in tqdm(zip(graphs, regression_ids, REGRESSION_DF.to_dict(orient="records"))):
        if set(g.symbols.values()) != ALLOWED_NODE_SYMBOL:
            continue
        prop_records.append(r)
        gs.append(g)
    return gs, prop_records


def export_tabular_data_record(qg: LQG, identifier: str):
    lfs = LqgFeatureSet.from_lqg(qg)
    r = lfs.as_dict()
    json_dump(r, f"data_tab_50k/{identifier}.json")
    return r


def prepare_tabular_data_output(n_jobs=1):
    lqgs, prop_records = load_data(n_jobs)

    if n_jobs == 1:
        for qg, prop in tqdm(zip(lqgs, prop_records)):
            export_tabular_data_record(qg, prop[LQG_ID_COLUMN])
    else:
        with tqdm_joblib(tqdm(desc="calculate lfs...", total=len(lqgs))) as progress_bar:
            Parallel(n_jobs=n_jobs)(
                delayed(export_tabular_data_record)(qg, prop[LQG_ID_COLUMN]) for qg, prop in zip(lqgs, prop_records))


def export_dfs():
    records = []
    for jf in tqdm(sorted(glob.glob("data_tab_50k/*.json"))):
        zeo_id = os.path.basename(jf).replace(".json", "")
        r = json_load(jf)
        r['zeo_id'] = zeo_id
        records.append(r)
    df_feat = pd.DataFrame.from_records(records)
    df_feat.set_index(keys=LQG_ID_COLUMN, inplace=True)
    df_feat.sort_index(inplace=True)
    df_target = REGRESSION_DF.set_index(keys=LQG_ID_COLUMN)
    df_target = df_target.loc[df_feat.index]
    df_feat.to_csv("data_tab_feat.csv")
    df_target.to_csv("data_tab_target.csv")


def prepare_tabular_data(n_jobs=1):  # this somehow hangs on windows...
    lqgs, prop_records = load_data(n_jobs)

    if n_jobs == 1:
        lfss = []
        for lqg in tqdm(lqgs):
            lfs = LqgFeatureSet.from_lqg(lqg)
            lfss.append(lfs)
    else:
        with tqdm_joblib(tqdm(desc="calculate lfs...", total=len(lqgs))) as progress_bar:
            lfss = Parallel(n_jobs=n_jobs)(delayed(LqgFeatureSet.from_lqg)(qg) for qg in lqgs)
    df_lfss = []
    df_prop = []
    for lfs, prop in zip(lfss, prop_records):
        if lfs is None:
            continue

        df_lfss.append(lfs.as_dict())
        df_prop.append(prop)
    return pd.DataFrame.from_records(df_lfss), pd.DataFrame.from_records(df_prop)


if __name__ == '__main__':
    # df_feat, df_target = prepare_tabular_data(n_jobs=12)
    # df_feat.to_csv("data_tab_50k/feat.csv", index=False)
    # df_target.to_csv("data_tab_50k/target.csv", index=False)
    prepare_tabular_data_output(n_jobs=12)
    export_dfs()
