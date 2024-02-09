import contextlib
import glob
import os.path
import random

import joblib
from joblib import Parallel, delayed
from loguru import logger
from pymatgen.core.structure import Structure
from tqdm import tqdm

from crystalgraph import LQG
from crystalgraph.utils import json_dump, json_load, FilePath


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


"""
export LQGs from pmg structure json
"""


def export_lqg(json_file: FilePath):
    assert os.path.isdir("lqg_bu_json")
    assert os.path.isdir("lqg_json")
    # logger.warning("writing to `lqg_bu_json` and `lqg_json`, can overwrite!")
    zeo = Structure.from_file(json_file)

    lqg = LQG.from_structure(zeo)
    d = lqg.as_dict()
    output = os.path.basename(json_file)
    json_dump(d, f"lqg_json/{output}")

    lqg_bu = lqg.bu_contraction()
    d = lqg_bu.as_dict()
    output = os.path.basename(json_file)
    json_dump(d, f"lqg_bu_json/{output}")


def export_lqg_(json_file: FilePath):
    try:
        export_lqg(json_file)
    except Exception as e:
        logger.error(e.__str__())


def export_lqgs(random_sample=False, k=None, n_jobs=1):
    pmg_jsons = sorted(glob.glob("pmg_json/*.json"))
    if random_sample:
        assert k is not None
        random.seed(42)
        pmg_jsons = random.sample(pmg_jsons, k)
    elif k is not None:
        pmg_jsons = pmg_jsons[:k]

    if n_jobs == 1:
        for jf in tqdm(pmg_jsons):
            try:
                export_lqg(jf)
            except Exception as e:
                logger.warning(e.__str__())
                continue
    else:
        with tqdm_joblib(tqdm(desc="parallel export lqgs", total=len(pmg_jsons))) as progress_bar:
            Parallel(n_jobs=n_jobs)(delayed(export_lqg_)(pmg_jsons[i]) for i in range(len(pmg_jsons)))


def load_lqg(json_file: FilePath) -> LQG:
    d = json_load(json_file)
    return LQG.from_dict(d)


if __name__ == '__main__':
    export_lqgs(random_sample=True, k=500)
