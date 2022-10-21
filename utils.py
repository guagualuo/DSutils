import numpy as np
import pandas as pd


def _random_state(seed) -> np.random.RandomState:
    if seed is None:
        return np.random.RandomState()
    elif isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise RuntimeWarning(f"random seed {seed} not accepted")


def _to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy()
    else:
        raise RuntimeWarning("array is not one of array, DataFrame, Series.")
