import glob
import os.path
import random

import tqdm
from loguru import logger
from pymatgen.core.structure import Structure

from crystalgraph import LQG
from crystalgraph.utils import json_dump, json_load, FilePath

"""
export LQGs from pmg structure json
"""


def export_lqg(json_file: FilePath):
    assert os.path.isdir("lqg_bu_json")
    assert os.path.isdir("lqg_json")
    logger.warning("writing to `lqg_bu_json` and `lqg_json`, can overwrite!")
    zeo = Structure.from_file(json_file)

    lqg = LQG.from_structure(zeo)
    d = lqg.as_dict()
    output = os.path.basename(json_file)
    json_dump(d, f"lqg_json/{output}")

    lqg_bu = lqg.bu_contraction()
    d = lqg_bu.as_dict()
    output = os.path.basename(json_file)
    json_dump(d, f"lqg_bu_json/{output}")


def export_lqgs(random_sample=False, k=None):
    pmg_jsons = sorted(glob.glob("pmg_json/*.json"))
    if random_sample:
        assert k is not None
        random.seed(42)
        pmg_jsons = random.sample(pmg_jsons, k)
    elif k is not None:
        pmg_jsons = pmg_jsons[:k]
    for jf in tqdm.tqdm(pmg_jsons):
        try:
            export_lqg(jf)
        except Exception as e:
            logger.warning(e)
            continue


def load_lqg(json_file: FilePath) -> LQG:
    d = json_load(json_file)
    return LQG.from_dict(d)


if __name__ == '__main__':
    export_lqgs(random_sample=True, k=500)
