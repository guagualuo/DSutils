import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_importance(forest,
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
    n_features = forest.n_features_in_ or forest.n_features_
    k = n_features if k is None else k

    if hasattr(forest, "feature_names_in_"):
        index = np.array(forest.feature_names_in_)
    elif hasattr(forest, "feature_name_"):
        index = np.array(forest.feature_name_)
    else:
        index = [i for i in range(forest.n_features_in_)]
    columns = [i for i in range(forest.n_estimators)]

    if hasattr(forest, "estimators_"):
        imp = pd.DataFrame({i: forest.estimators_[i].feature_importances_
                            for i in range(forest.n_estimators)},
                           columns=columns, index=index)
        std = imp.std(axis=1) / np.sqrt(forest.n_estimators)
    else:
        std = np.zeros_like(importances)

    indices = np.argsort(importances)

    # Plot the feature importances of the forest
    plt.figure(**kwargs)
    plt.title("Feature importances")
    plt.barh(range(k), importances[indices[:k]],
             color="r", xerr=std[indices[:k]], align="center")

    plt.yticks(range(k), index[indices[:k]])
    plt.ylim([-1, k])
    plt.show()


def _compute_shap_values(tree_model,
                         X):
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X)
    if len(shap_values) == 2:
        return shap_values[1]
    else:
        return shap_values


def shap_summary_plot(tree_model,
                      X,
                      **kwargs):
    shap_values = _compute_shap_values(tree_model, X)
    shap.summary_plot(shap_values, X, show=False, **kwargs)


def shap_dependence_plot(tree_model,
                         X: pd.DataFrame,
                         feature_1=None,
                         feature_2=None):
    shap_values = _compute_shap_values(tree_model, X)
    if feature_1 is None and feature_2 is None:
        n_features = X.shape[1]
        ncols = 4
        nrows = n_features // ncols + (n_features % ncols != 0)
        fig, axes = plt.subplots(figsize=(16, 3.5 * nrows), nrows=nrows, ncols=ncols)
        for col, ax in zip(X.columns, np.ravel(axes)):
            shap.dependence_plot(col, shap_values, X, X.columns, ax=ax, show=False)

        plt.tight_layout()
    else:
        shap.dependence_plot(feature_1, shap_values, X, interaction_index=feature_2, show=False)
