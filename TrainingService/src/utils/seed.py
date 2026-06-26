from __future__ import annotations

import os
import random

import numpy as np

from config.settings import RANDOM_STATE
from utils.decorators import pipeline_step


@pipeline_step("Set global random seed")
def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Seed Python and NumPy for reproducibility
    in classical ML workflows.
    """
    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)