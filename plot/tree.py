import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble._forest import BaseForest


def plot_importance(forest: BaseForest,
                    k: int = None,
                    **kwargs
                    ):
    """
    Function to plot the feature importance of a forest model

    Parameters
    -----
    forest: A forest object inheritated from BaseForest

    k: int, number of features to plot

    **kwargs: kwargs for the figure

    """

    importances = forest.feature_importances_
    n_features = forest.n_features_in_
    k = n_features if k is None else k

    index = forest.feature_names_in_ if hasattr(forest, "feature_names_in_") \
        else [i for i in range(forest.n_features_in_)]
    columns = [i for i in range(forest.n_estimators)]
    imp = pd.DataFrame({i: forest.estimators_[i].feature_importances_
                        for i in range(forest.n_estimators)},
                       columns=columns, index=index)
    std = imp.std(axis=1) / np.sqrt(forest.n_estimators)

    indices = np.argsort(importances)

    # Plot the feature importances of the forest
    plt.figure(**kwargs)
    plt.title("Feature importances")
    plt.barh(range(k), importances[indices[:k]],
           color="r", xerr=std[indices], align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(k), index[indices[:k]])
    plt.ylim([-1, k])
    plt.show()

from xgboost import XGBClassifier
