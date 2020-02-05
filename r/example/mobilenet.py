#!/usr/bin/env python3.7
# pylint: skip-file

from typing import List
from typing import Tuple

import functools
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import AnalysisPredictor
from paddle.fluid.core import create_paddle_predictor


def main():
    config: AnalysisConfig = set_config()
    predictor: AnalysisPredictor = create_paddle_predictor(config)

    data, result = parse_data()

    input_names: List[str] = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    shape: Tuple[int] = (1, 3, 300, 300)
    input_data: np.array = data[:-4].astype(np.float32).reshape(shape)
    input_tensor.copy_from_cpu(input_data)

    predictor.zero_copy_run()

    output_names: str = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data: np.array = output_tensor.copy_to_cpu()


def set_config() -> AnalysisConfig:
    config = AnalysisConfig("")
    config.set_model("model/__model__", "model/__params__")
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_profile()

    return config


def parse_data() -> Tuple[np.array, np.array]:
    """ parse input and output data """
    with open('data/data.txt', 'r') as fr:
        data = np.array([float(_) for _ in fr.read().split()])

    with open('data/result.txt', 'r') as fr:
        result = np.array([float(_) for _ in fr.read().split()])

    return (data, result)


if __name__ == "__main__":
    main()
