from typing import Literal, Union, Callable
import numpy as np
import pandas as pd

from utils import _random_state


def _convert_data(y) -> np.ndarray:
    if isinstance(y, list):
        y = [y_.values if isinstance(y_, (pd.Series, pd.DataFrame)) else y_ for y_ in y]
        n_dim = y[0].ndim
        y = np.stack(y, axis=n_dim)
    else:
        y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
    if y.ndim not in [2, 3]:
        raise RuntimeError(f"y has dimension {y.ndim}, does not match shape of " +
                           "(n_samples, n_models) or (n_samples, n_outputs, n_models)")
    return y


def majority_voting(y):
    """
    Majority voting of model predictions

    Parameters
    -----
    y:  {array_like or a list}
        array of shape (n_samples, n_models) or (n_samples, n_outputs, n_models)
        list of array of (n_samples, ) or (n_samples, n_outputs)

    Returns
    -----
    majority_voting of the prediction

    """
    from scipy.stats import mode
    y = _convert_data(y)
    return mode(y, axis=y.ndim-1)[0][..., 0]


def model_averaging(y,
                    method: str = Literal['arithmetic', 'geometric', 'logarithmic', 'harmonic', 'powers'],
                    n_powers: int = 3,
                    ):
    """
    Averaging of model predictions

    Parameters
    -----
    Parameters
    -----
    y:  {array_like or a list}
        array of shape (n_samples, n_models) or (n_samples, n_outputs, n_models)
        list of array of (n_samples, ) or (n_samples, n_outputs)

    method: {'arithmetic', 'geometric', 'logarithmic', 'harmonic', 'powers'}
        Method for averaging

    n_powers: int = 3
        Power in the power averaging

    Returns
    -----
    Averaging of the prediction

    """

    if method not in ['arithmetic', 'geometric', 'logarithmic', 'harmonic', 'powers']:
        raise RuntimeError("method must be in ['arithmetic', 'geometric', 'logarithmic', 'harmonic', 'powers']")

    y = _convert_data(y)
    n_models = y.shape[-1]

    if method in ["geometric", "logarithmic", "harmonic", "powers"] and np.any(y < 0):
        raise RuntimeWarning(f"Non-positive predictions used for {method} averaging!")

    n_dim = y.ndim
    eps = 1e-7
    if method == "arithmetic":
        return np.mean(y, axis=n_dim-1)
    elif method == "geometric":
        return np.prod(y, axis=n_dim-1) ** (1/n_models)
    elif method == "logarithmic":
        return np.expm1(np.mean(np.log1p(y), axis=n_dim-1))
    elif method == "harmonic":
        return 1 / np.mean(1/(y + eps), axis=n_dim-1)
    else:
        return np.mean(y**n_powers, axis=n_dim-1) ** (1/n_powers)


def correlation_weighted_average(y,
                                 correlation: bool = True):
    """
    Inversely proportional to model's correlation with other models weighted averaged model

    y: {array_like or a list}
        array of shape (n_samples, n_models) or (n_samples, n_outputs, n_models)
        list of array of (n_samples, ) or (n_samples, n_outputs)

    correlation: bool = False
        Whether to return the correlations between predictions

    Returns
    -----
    pred: Weighted averaged prediction (n_samples,) or (n_samples, n_outputs)

    corr: Correlation between predictions (n_models, n_models) or (n_models, n_models, n_outputs)

    """
    y = _convert_data(y)
    if y.ndim == 2:
        corr = np.corrcoef(y, rowvar=False)
        np.fill_diagonal(corr, 0)
        w = 1. / np.mean(corr, axis=1)
        w = w / sum(w)
        pred = y @ w

        if correlation:
            np.fill_diagonal(corr, 1)
            return pred, corr
    else:
        corrs = []
        for j in range(y.shape[1]):
            corrs.append(np.corrcoef(y[:, j, :], rowvar=False))
        corr = sum(corrs) / len(corrs)
        np.fill_diagonal(corr, 0)
        w = 1. / np.mean(corr, axis=1)
        w = w / sum(w)
        pred = np.zeros_like(y[..., 0])
        for i, wi in enumerate(w):
            pred += y[..., i] * wi

        if correlation:
            return pred, np.stack(corrs, axis=2)

    return pred


class EnsembleSelection:
    """
    Ensemble selection for model averaging.

    Parameters
    -----
    scorer: Callable takes (y_hold, y_pred)
        compute the score the model INSTEAD OF LOSS

    sampling: {int, float}, default None
        sampling frequency of the models at each step of the forward selection

    replacement: bool, default True
        Sample with replacement allows each model to be selected multiple times

    n_init: int, default 3
        Number of models to select at the initial stage according to the `init_method`

    init_method: {"best", "random"}
        "best": select the best `n_init` models from all models
        "random": randomly select `n_init` models from all models

    max_iter: int, default 100
        Maximal number of forward steps

    random_state: random seed

    Attributes
    -----
    weights_: dict: {index, weight}
        The weights of each model component

    model_seq_: array
        Array of model index been selected

    scores_: array
        Array of ensemble model scores

    References
    -----
    Caruana, R. et al. 2004, "Ensemble selection from libraries of models"

    """

    def __init__(self,
                 scorer: Callable[[np.ndarray, np.ndarray], float],
                 sampling: Union[float, int] = None,
                 replacement: bool = True,
                 n_init: int = 3,
                 init_method: Literal["best", "random"] = "best",
                 max_iter: int = 100,
                 random_state: Union[int, np.random.RandomState] = None,
                 ):
        self.scorer = scorer
        self.sampling = sampling
        self.replacement = replacement
        self.n_init = n_init
        self.init_method = init_method
        self.max_iter = max_iter
        self.random_state = _random_state(random_state)

        self._check_params()

    def _check_params(self):
        if isinstance(self.sampling, int):
            assert self.sampling > 0
            if self.sampling == 1:
                self.sampling = 1.0
        if isinstance(self.sampling, float):
            assert 0 < self.sampling <= 1.0

    def fit(self, y_hold, y_pred):
        rng = self.random_state

        y_pred = _convert_data(y_pred)
        n_models = y_pred.shape[-1]
        avail_models = {i: i for i in range(n_models)}

        # score for each model
        init_scores = [(self.scorer(y_hold, y_pred[..., i]), i) for i in range(n_models)]
        init_scores = sorted(init_scores, key=lambda x: x[0], reverse=True)

        # initial model set
        if self.init_method == "best":
            n_init_set = min(self.n_init, n_models)
            models = [x[1] for x in init_scores[:n_init_set]]
        else:
            models = list(rng.choice(list(avail_models.keys()), self.n_init, replace=False))

        # sample with replacement
        if not self.replacement:
            for j in models:
                avail_models.pop(j)

        baseline = self.scorer(y_hold, np.mean(y_pred[..., models], axis=-1))
        self.scores_ = [baseline]

        # forward selection
        k = 0
        while k < self.max_iter and len(avail_models) > 0:
            # sampling of models
            if self.sampling is not None:
                n_model_samples = self.sampling if isinstance(self.sampling, int) \
                    else int(np.round(self.sampling * len(avail_models)))
                n_model_samples = max(1, min(n_model_samples, len(avail_models)))
                avail_models_ = rng.choice(list(avail_models.keys()), n_model_samples, replace=False)
            else:
                avail_models_ = avail_models.keys()
            scores = [(self.scorer(y_hold, np.mean(y_pred[..., models + [i]], axis=-1)), i)
                      for i in avail_models_]
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            best_score, best_model = scores[0]

            if best_score <= baseline:
                break
            baseline = best_score
            self.scores_.append(best_score)
            models.append(best_model)
            if not self.replacement:
                avail_models.pop(best_model)
            k += 1

        # compute model weights
        from collections import Counter
        weight_counter = Counter(models)
        self.weights_ = np.zeros(n_models)

        self.weights_[list(weight_counter.keys())] = np.array(list(weight_counter.values())) / len(models)
        self.model_seq_ = np.array(models)
        self.scores_ = np.array(self.scores_)

    def predict(self, y_pred):
        y_pred = _convert_data(y_pred)
        return y_pred @ self.weights_

    def fit_predict(self, y_hold, y_hold_pred, y_pred):
        self.fit(y_hold, y_hold_pred)
        return self.predict(y_pred)
