import contextlib
import glob
import os.path
import random

import joblib
from joblib import Parallel, delayed
from loguru import logger
from pymatgen.core.structure import Structure
from tqdm import tqdm

from pathlib import Path
from crystalgraph import LQG
from crystalgraph.utils import json_dump, json_load, FilePath

"""
export LQGs from cssr
"""

CSSR_FOLDER = "data/cssr"
LQG_FOLDER = "data/LQG"
PBU_LQG_FOLDER = "data/PBU_LQG"
Path(LQG_FOLDER).mkdir(exist_ok=True)
Path(PBU_LQG_FOLDER).mkdir(exist_ok=True)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



def export_lqg(structure_file: FilePath):
    # logger.warning("writing to `lqg_bu_json` and `lqg_json`, can overwrite!")
    zeo = Structure.from_file(structure_file)
    zeo_id = Path(structure_file).stem

    lqg = LQG.from_structure(zeo)
    d = lqg.as_dict()
    json_dump(d, f"{LQG_FOLDER}/{zeo_id}.json")

    lqg_bu = lqg.bu_contraction()
    d = lqg_bu.as_dict()
    json_dump(d, f"{PBU_LQG_FOLDER}/{zeo_id}.json")


def export_lqg_(json_file: FilePath):
    try:
        export_lqg(json_file)
    except Exception as e:
        logger.error(e.__str__())


def export_lqgs(random_sample=False, k=None, n_jobs=1):
    structure_files = sorted(glob.glob(f"{CSSR_FOLDER}/*.cssr"))
    if random_sample:
        assert k is not None
        random.seed(42)
        structure_files = random.sample(structure_files, k)
    elif k is not None:
        structure_files = structure_files[:k]

    if n_jobs == 1:
        for jf in tqdm(structure_files):
            try:
                export_lqg(jf)
            except Exception as e:
                logger.warning(e.__str__())
                continue
    else:
        with tqdm_joblib(tqdm(desc="parallel export lqgs", total=len(structure_files))) as progress_bar:
            Parallel(n_jobs=n_jobs)(delayed(export_lqg_)(structure_files[i]) for i in range(len(structure_files)))


def load_lqg(json_file: FilePath) -> LQG:
    d = json_load(json_file)
    return LQG.from_dict(d)


if __name__ == '__main__':
    export_lqgs(random_sample=True, k=50000, n_jobs=60)
