from typing import Any

import deepchem as dc
import numpy as np
from deepchem.feat.base_classes import UserDefinedFeaturizer


class MyFeaturizer(UserDefinedFeaturizer):
    def _featurize(self, datapoint: Any):
        return datapoint


tasks, datasets, transformers = dc.molnet.load_qm8(
    shard_size=2000, featurizer=MyFeaturizer(None), splitter="random")

print(tasks)