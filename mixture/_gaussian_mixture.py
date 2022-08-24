import numpy as np
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky


class RestrictedGaussianMixture(GaussianMixture):
    """ Restricted GaussianMixture model that impose some entries of the means and covariances,
     currently assumes "full" covariance type.

     Parameters
     -----
     imposed_means: np.array (n_components, n_features), non-imposed entries are np.NaN

     imposed_covariances: np.array(n_components, n_features, n_features), non-imposed entries are np.NaN

     """

    def __init__(
            self,
            n_components=1,
            *,
            imposed_means: np.array,
            imposed_covariances: np.array,
            **kwargs,
    ):
        super(RestrictedGaussianMixture, self).__init__(n_components, covariance_type="full", **kwargs)
        self.imposed_means = imposed_means
        self.imposed_covariances = imposed_covariances

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        self.means_[~np.isnan(self.imposed_means)] = self.imposed_means[~np.isnan(self.imposed_means)]

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.covariances_[~np.isnan(self.imposed_covariances)] = self.imposed_covariances[~np.isnan(self.imposed_covariances)]
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = np.sqrt(self.precisions_init)

    def _m_step(self, X, log_resp):
        """
        Modified M-step to project onto the restricted space

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, means_, covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        means_[~np.isnan(self.imposed_means)] = self.imposed_means[~np.isnan(self.imposed_means)]
        covariances_[~np.isnan(self.imposed_covariances)] = self.imposed_covariances[~np.isnan(self.imposed_covariances)]
        self.means_ = means_
        self.covariances_ = covariances_

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
