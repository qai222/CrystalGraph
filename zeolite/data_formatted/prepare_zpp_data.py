import glob
import os.path
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from loguru import logger
from pandas._typing import FilePath


def parse_res(file: FilePath):
    """
    diameters of, respectively, 
    the largest included sphere, 
    the largest free sphere,
    and the largest included sphere along free sphere path
    """
    with open(file, "r") as f:
        values = f.readline().split()[1:]
    keys = [
        "largest_included_sphere", "largest_free_sphere",
        "largest_included_sphere_along_free_sphere_path",
    ]
    return dict(zip(keys, values))


def parse_chan(file: FilePath):
    with open(file, "r") as f:
        first_line = f.readline().strip()
        items = first_line.split()[1:]
    total = int(items[0])
    if total == 0:
        values = [0, 0, 0]
    else:
        dims = [int(dim) for dim in items[-total:]]
        dim_counter = Counter(dims)
        values = [dim_counter[d] for d in [1, 2, 3]]
    values.append(sum(values))
    keys = ["num_1d_chan", "num_2d_chan", "num_3d_chan", "num_chan"]
    return dict(zip(keys, values))


def parse_sa(file: FilePath):
    # TODO add chan sa and pocket info
    with open(file, "r") as f:
        first_line = f.readline().strip()
    items = first_line.split()[2:]
    data = dict()
    for i in range(len(items)):
        if i % 2 == 0:
            data[items[i]] = float(items[i + 1])
    sa_data = {k: data[k] for k in ["Unitcell_volume:", "Density:", "ASA_A^2:", "NASA_A^2:", ]}
    return sa_data


def _prepare_one(
        lqg_json_file: FilePath,
        zpp_results_folder: FilePath
):
    zeo_id = os.path.basename(lqg_json_file.replace(".json", ""))
    chan_file = f"{zpp_results_folder}/chan/{zeo_id}.chan"
    res_file = f"{zpp_results_folder}/res/{zeo_id}.res"
    sa_file = f"{zpp_results_folder}/sa/{zeo_id}.sa"
    if any(not os.path.isfile(f) for f in (chan_file, res_file, sa_file)):
        logger.critical(f"not all results are found for: {zeo_id}")
        return
    data = dict()
    data["zeo_id"] = zeo_id
    data.update(parse_res(res_file))
    data.update(parse_sa(sa_file))
    data.update(parse_chan(chan_file))
    return data


def prepare(zpp_results_folder: FilePath, lqg_folder: FilePath, outcsv: FilePath, n_jobs=4):
    lqg_json_files = sorted(glob.glob(f"{lqg_folder}/*.json"))
    assert os.path.isdir(f"{zpp_results_folder}/chan")
    assert os.path.isdir(f"{zpp_results_folder}/sa")
    assert os.path.isdir(f"{zpp_results_folder}/res")
    with joblib_progress("Prepare zpp dataset...", total=len(lqg_json_files)):
        records = Parallel(n_jobs=n_jobs)(
            delayed(_prepare_one)(lqg_json_file, zpp_results_folder) for lqg_json_file in lqg_json_files)
    logger.info(f"organize results for # of zeolites: {len(lqg_json_files)}")
    logger.info(f"# records obtained: {len(records)}")

    df = pd.DataFrame.from_records(records)
    df.to_csv(outcsv, index=False)
    return records


if __name__ == '__main__':
    prepare("../data/results", "../data/PBU_LQG_50k", outcsv="data_zpp_50k.csv")
