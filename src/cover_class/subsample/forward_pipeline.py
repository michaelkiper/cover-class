from typing import Tuple, Dict
import torch
from torch import FloatTensor, Tensor
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split as tts # type: ignore[import]

from cover_class.utils import read_config
from cover_class.subsample.subsampler import convex_hull, kmeans, kmedoids, lhs

def train_test_split(data_matrix: FloatTensor, labels:Tensor, frac_test: float, seed: int=42) -> Tuple[FloatTensor, FloatTensor, Tensor, Tensor]:
    return tts(data_matrix, labels, test_size=frac_test, random_state=seed)

def subsample_from_config(
        config:str|Dict, 
        data_matrix: NDArray[np.float32], 
    ) -> FloatTensor:

    subsample_config:Dict = read_config(config)['subsample']
    method:str = str(subsample_config['selected-method']).lower()
    
    match method:
        case 'convex-hull':
            return convex_hull(data_matrix, **subsample_config[method])
        case 'kmeans':
            return kmeans(data_matrix, **subsample_config[method])
        case 'kmedoids':
            return kmedoids(data_matrix, **subsample_config[method])
        case 'lhs':
            return lhs(data_matrix, **subsample_config[method])
        case _:
            return FloatTensor(torch.from_numpy(data_matrix).to(torch.float32))
    return FloatTensor() # here for mypy