from typing import Union, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mode
from collections import defaultdict
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from utils import _random_state


class VotingEnsemble:
    """
    Ensemble of clustering models with the same number of clusters.
    Support both hard and soft votings.
    Infer correspondence btw cluster labels using the Hungarian algorithm.

    Parameters
    -----
    proba: True for soft voting

    random_state: random state seed

    Attributes
    -----
    n_base_clusters_: number of base clusters

    ensemble_proba_: (n_samples, n_clusters), when proba=True, the ensembled probability for each cluster

    ensemble_partitions: (n_samples, n_models), when proba=False, the cluster label from each of the base model,
        adjusted by label correspondence.

    """
    def __init__(self,
                 proba: bool = True,
                 random_state: Union[int, np.random.RandomState] = None
                 ):
        self.proba = proba
        self.random_state = _random_state(random_state)

    def _proba2label(self, partitions):
        if self.proba:
            return [np.argmax(p, axis=1) for p in partitions]
        else:
            return partitions

    def _map_partition(self, c0, c1, partition):
        if self.proba:
            c1 = np.array([i for _, i in sorted(zip(c0, c1))])
            return partition[:, c1]
        else:
            return pd.Series(partition).replace(list(c1), list(c0)).values

    def ensemble(self, partitions: List[np.ndarray], weights=None):
        """
        Ensemble cluster labels or probabilities.

        Parameter
        -----
        partitions: list of labels or probabilities

        weights: weight for each model, effective only for soft voting.
        """
        self.n_base_clusters_ = len(partitions)
        if weights is None:
            weights = np.ones(len(partitions)) / len(partitions)
        else:
            weights /= np.sum(weights)
        return self._simple_voting(partitions, weights)

    def _simple_voting(self, partitions, weights):
        _labels = self._proba2label(partitions)
        n_partitions = len(_labels)

        if self.proba:
            ensemble_proba = deepcopy(partitions[0]) * weights[0]
        else:
            ensemble_partition = pd.DataFrame()
            ensemble_partition[0] = partitions[0]

        current_label = _labels[0]

        for i in range(1, n_partitions):
            C = contingency_matrix(current_label, _labels[i])
            c0, c1 = linear_sum_assignment(C, maximize=True)
            mapped_partition = self._map_partition(c0, c1, partitions[i])
            if self.proba:
                ensemble_proba += mapped_partition * weights[i]
                current_label = np.argmax(ensemble_proba, axis=1)
            else:
                ensemble_partition[i] = mapped_partition

        if self.proba:
            self.ensemble_proba_ = ensemble_proba
            return np.argmax(ensemble_proba, axis=1)
        else:
            self.ensemble_partition_ = ensemble_partition
            return ensemble_partition.mode(axis=1).iloc[:, 0]


class IVC:
    """
    Iterative Voting Consensus (IVC)

    Similar to KModes, but the cluster center is determined component-wise.

    Parameter
    -----
    n_clusters: number of clusters to form

    init: ["random", "smart"], current only support random

    max_iter: max number of iterations

    n_init: number of different initialisations to run

    random_state: default None

    Attributes
    -----
    n_base_clusters_: number of base clusters

    score_: The best average within-cluster hamming distance of cluster labels among different initialisations

    centers_: The best cluster centers among different initialisations

    fitted_labels_: Fitted labels.

    """

    def __init__(self,
                 n_clusters: int = 7,
                 init: str = 'random',
                 max_iter: int = 100,
                 n_init: int = 10,
                 random_state: Union[int, np.random.RandomState] = None,
                 ):
        self.n_clusters = n_clusters
        self.init = init
        self.random_state = _random_state(random_state)
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, X: np.ndarray):
        """ X is the label matrix of an ensemble of clusters """
        self.n_base_clusters_ = X.shape[1]
        clusters = [np.unique(X[:, i]) for i in range(self.n_base_clusters_)]

        centers_list = []
        labels_list = []
        scores = []

        for i_init in range(self.n_init):
            centers = self._init_centers(clusters)
            # assign cluster label to each sample
            dist = cdist(X, centers, metric='hamming')
            labels = np.argmin(dist, axis=1)

            n_iter = 0
            change_in_centers = 1
            while change_in_centers > 0 and n_iter < self.max_iter:
                n_iter += 1
                last_centers = centers.copy()
                labels, score = self._single_iteration(X, labels, centers)
                change_in_centers = np.linalg.norm(last_centers - centers, ord='fro')

            if n_iter >= self.max_iter and change_in_centers > 0:
                print(f"IVC Not converged after {n_iter} iterations")

            centers_list.append(centers)
            labels_list.append(labels)
            scores.append(score)
        k = np.argmin(scores)
        self.score_ = scores[k]
        self.centers_ = centers_list[k]
        self.fitted_labels_ = labels_list[k]

    def predict(self, X: np.ndarray):
        dist = cdist(X, self.centers_, metric='hamming')
        return np.argmin(dist, axis=1)

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.fitted_labels_

    def _init_centers(self, clusters: List[np.array]):
        assert self.init in ['random', 'smart'], 'Unknown method for initialising cluster centers'
        n_base_clusters = len(clusters)
        centers = np.full(shape=(self.n_clusters, n_base_clusters),
                                fill_value=np.NaN)
        if self.init == "random":
            for j, cluster in enumerate(clusters):
                centers[:, j] = self.random_state.choice(clusters[j], self.n_clusters)
        else:
            raise NotImplementedError
        return centers

    @staticmethod
    def _single_iteration(X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        # update center
        n_clusters = centers.shape[0]
        for i in range(n_clusters):
            Xp = X[labels==i]
            if len(Xp) > 0:
                centers[i, :] = mode(Xp, axis=0)[0]

        # resign each sample to its nearest center
        dist = cdist(X, centers, metric='hamming')
        score = np.mean(np.min(dist, axis=1))  # within cluster error
        labels = np.argmin(dist, axis=1)
        return labels, score


def _bhattacharyya_distance_discrete(p: np.ndarray, q: np.ndarray):
    return -np.log((p**0.5).dot(q**0.5))


class IPC:
    """
    Iterative Probability Consensus (IPC)
    Similar to IVC, but uses probabilities instead of labels from base clusters as features for training.
    The distance used is the Bhattacharyya distance.

    Parameter
    -----
    n_clusters: number of clusters to form

    init: ["random", "smart"], currently only support random

    max_iter: max number of iterations

    n_init: number of different initialisations to run

    random_state: default None

    Attributes
    -----
    num_components_: List of the number of components in each base cluster

    score_: The best average within-cluster hamming distance of cluster labels among different initialisations

    centers_: The best cluster centers among different initialisations

    fitted_labels_: Fitted labels.

    """

    def __init__(self,
                 n_clusters=7,
                 init='random',
                 max_iter=100,
                 n_init: int = 10,
                 tol: float = 0.001,
                 random_state: Union[int, np.random.RandomState] = None,
                 ):
        self.n_clusters = n_clusters
        self.init = init
        self.random_state = _random_state(random_state)
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol

    def fit(self, X: np.ndarray):
        """
        X (n_samples, n_clusters_0 + n_clusters_1 + ...)
        is the concatenated probability matrix of an ensemble of clusters
        """
        self._infer_num_components(X)
        centers_list = []
        labels_list = []
        scores = []

        for i_init in range(self.n_init):
            centers = self._init_centers(X)
            # assign cluster label to each sample
            dist = self._compute_dist(X, centers)
            labels = np.argmin(dist, axis=1)

            n_iter = 0
            change_in_centers = 1
            while change_in_centers > self.tol and n_iter < self.max_iter:
                n_iter += 1
                last_centers = centers.copy()
                labels, score = self._single_iteration(X, labels, centers)
                # print(score)
                change_in_centers = self._compute_dist(last_centers, centers).diagonal().sum()

            if n_iter >= self.max_iter and change_in_centers > self.tol:
                print(f"IPC Not converged after {n_iter} iterations")

            centers_list.append(centers)
            labels_list.append(labels)
            scores.append(score)

        k = np.argmin(scores)
        self.score_ = scores[k]
        self.centers_ = centers_list[k]
        self.fitted_labels_ = labels_list[k]

    def predict(self, X: np.ndarray):
        dist = self._compute_dist(X, self.centers_)
        return np.argmin(dist, axis=1)

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.fitted_labels_

    def _infer_num_components(self, X: np.ndarray):
        # determine number of components in each model
        n_components = []
        mean = X.mean(axis=0)
        psum = 0
        k = 0
        for p in mean:
            psum += p
            k += 1
            if abs(psum - 1) < 1e-10:
                n_components.append(k)
                k = 0
                psum = 0
        self.num_components_ = n_components

    def _compute_dist(self, X: np.ndarray, y: np.ndarray):
        dist = np.zeros((X.shape[0], y.shape[0]))
        i, j = 0, 0
        for k in range(len(self.num_components_)):
            i = j
            j += self.num_components_[k]
            dist += -np.log(X[:, i:j]**0.5 @ y[:, i:j].T**0.5)
        return dist/len(self.num_components_)

    def _init_centers(self, X: np.ndarray):
        assert self.init in ['random', 'smart'], 'Unknown method for initialising cluster centers'

        def _gen_slice():
            p = self.random_state.rand(X.shape[1])
            i, j = 0, 0
            for k in range(len(self.num_components_)):
                i = j
                j += self.num_components_[k]
                p[i:j] = p[i:j] / np.sum(p[i:j])
            return p

        centers = np.full(shape=(self.n_clusters, X.shape[1]),
                                fill_value=np.NaN)
        if self.init == "random":
            for i in range(self.n_clusters):
                centers[i, :] = _gen_slice()
        else:
            raise NotImplementedError
        return centers

    def _single_iteration(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        # update center
        n_clusters = centers.shape[0]
        for i in range(n_clusters):
            Xp = X[labels==i]
            centers[i, :] = np.mean(Xp, axis=0)

        # resign each sample to its nearest center
        dist = self._compute_dist(X, centers)
        score = np.mean(np.min(dist, axis=1))  # within cluster error
        labels = np.argmin(dist, axis=1)
        return labels, score


def _bhattacharyya_distance_gaussian(mu1, mu2, cov1, cov2):
    if isinstance(mu1, (float, int)):
        mu1 = np.array([mu1])
    if isinstance(mu2, (float, int)):
        mu2 = np.array([mu2])
    if isinstance(cov1, (float, int)):
        cov1 = np.array([[cov1]])
    if isinstance(cov2, (float, int)):
        cov2 = np.array([[cov2]])
    mu1 = mu1.reshape(-1, 1)
    mu2 = mu2.reshape(-1, 1)

    res = 1/8 * (mu1 - mu2).T @ np.linalg.solve((cov1 + cov2)/2, mu1 - mu2)
    det12 = np.linalg.det((cov1 + cov2)/2)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    res = res[0] + 0.5*np.log(det12 / np.sqrt(det1 * det2))
    return res


class GMMSE:
    """
    The Gaussian mixture model cluster structure ensemble,
    based on "Probabilistic cluster structure ensemble" by Zhiwen Yu et al (2014)

    A spectral clustering algorithm used for the mixture components. Distance/affinity matrix computed based on
    the Bhattacharyya distance.

    Parameter
    -----
    n_neighbors: number of neighbors to consider forming the attraction matrix

    label_method: the method used to assign label for each sample. Current only support "gmv" means
    first get the label in each Gaussian Mixture model and do a majority vote.

    kwargs: parameters for the spectral clustering object used to extract structure among the mixtures.

    Attributes
    -----
    distance_: The distance matrix for each component of the mixtures

    attraction_: The attraction matrix

    mixture_labels_: The components' id of each mixture model

    labels_: The label for each mixture component

    spectral_cluster_: The fitted SpectralClustering object for mixture components

    """

    def __init__(self,
                 n_neighbors: int,
                 label_method: str = "gmv",
                 **kwargs):
        self.n_neighbors = n_neighbors
        self.label_method = label_method
        self.spectral_cluster_params = kwargs

    def _distance_matrix(self,
                          gms: List[Union[GaussianMixture, BayesianGaussianMixture]]
                         ):
        n_mixtures = 0
        means = []
        covariances = []
        mixture_labels = defaultdict(list)
        for index, gm in enumerate(gms):
            for i in range(gm.n_components):
                mixture_labels[index].append(n_mixtures)
                n_mixtures += 1
                means.append(gm.means_[i])
                covariances.append(gm.covariances_[i])

        distance = np.zeros((n_mixtures, n_mixtures))
        for i in range(n_mixtures):
            for j in range(i+1, n_mixtures):
                d = _bhattacharyya_distance_gaussian(means[i], means[j], covariances[i], covariances[i])
                distance[i, j] = d
                distance[j, i] = d
        self.distance_ = distance
        self.mixture_labels_ = mixture_labels

    def _attraction_matrix(self, ):
        assert hasattr(self, "distance_"), "distance matrix has not been computed"

        n = self.distance_.shape[0]
        zeta = np.zeros(n)
        for i in range(n):
            zeta[i] = np.sum(np.partition(self.distance_[i], self.n_neighbors)[:self.n_neighbors+1]) / self.n_neighbors
        attraction = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                attraction[i, j] = (zeta[i] + zeta[j]) / 2
        np.fill_diagonal(attraction, 0)
        self.attraction_ = attraction

    def fit(self,
            gms: List[Union[GaussianMixture, BayesianGaussianMixture]],
            X):
        self.fit(gms, X)
        return self

    def fit_predict(self,
                    gms: List[Union[GaussianMixture, BayesianGaussianMixture]],
                    X):
        self._distance_matrix(gms)
        self._attraction_matrix()
        d = self.distance_ + self.attraction_
        A = np.exp(-d**2/np.std(d)**2)

        spec_cluster = SpectralClustering(affinity='precomputed', **self.spectral_cluster_params)
        labels = spec_cluster.fit_predict(A)
        self.labels_ = labels
        self.spectral_cluster_ = spec_cluster

        if self.label_method == "gmv":
            return self._gmv(labels, gms, X)
        else:
            raise NotImplementedError(f"{self.label_method} labeling method not implemented")

    def _gmv(self,
            labels,
            gms: List[Union[GaussianMixture, BayesianGaussianMixture]],
            X):

        votes = np.full((X.shape[0], len(gms)), fill_value=np.NaN)
        for index, gm in enumerate(gms):
            mixture_label = gm.predict(X)
            votes[:, index] = labels[np.array(self.mixture_labels_[index])[mixture_label]]

        return mode(votes, axis=1)[0][:, 0]
