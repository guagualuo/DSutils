import numpy as np


def _random_state(seed) -> np.random.RandomState:
    if seed is None:
        return np.random.RandomState()
    elif isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise RuntimeWarning(f"random seed {seed} not accepted")