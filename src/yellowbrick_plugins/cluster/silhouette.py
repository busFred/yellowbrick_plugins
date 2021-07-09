from overrides import overrides
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn_plugins.cluster import SphericalKMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.utils import check_fitted

__all__ = ["SphericalSilhouetteVisualizer"]


class SphericalSilhouetteVisualizer(SilhouetteVisualizer):
    """Visualize Silhouette performance for SphericalKMeans.

    SpehricalKMeans first transfrom the input with principal component analysis (PCA) and then calculates centroids based on the transformed input. However, `yellowbrick.cluster.KElbowVisualizer` calculates the metrics based on inputs prior to principal component analysis.

    In `SphericalKElbowVisualizer` is designed to fix abovementioned phenomena for `SphericalKMeans`. The `SphericalKElbowVisualizer.fit` method is overridden so that function `self.scoring_metric` can have X argument equla to `self.estimator.preprocess_input(X)`.
    """

    def __init__(self,
                 estimator: SphericalKMeans,
                 ax=None,
                 colors=None,
                 is_fitted="auto",
                 **kwargs):
        super().__init__(estimator, ax, colors, is_fitted, **kwargs)

    @overrides
    def fit(self, X, y=None, **kwargs):
        # TODO: decide to use this method or the score method to draw.
        # NOTE: Probably this would be better in score, but the standard score
        # is a little different and I'm not sure how it's used.

        if not check_fitted(self.estimator, is_fitted_by=self.is_fitted):
            # Fit the wrapped estimator
            self.estimator.fit(X, y, **kwargs)

        # Get the properties of the dataset
        self.n_samples_ = X.shape[0]
        self.n_clusters_ = self.estimator.n_clusters

        # Compute the scores of the cluster
        labels = self.estimator.predict(X)
        input_preprocess = self.estimator.preprocess_input(X)
        self.silhouette_score_ = silhouette_score(input_preprocess, labels)
        self.silhouette_samples_ = silhouette_samples(input_preprocess, labels)

        # Draw the silhouette figure
        self.draw(labels)

        # Return the estimator
        return self
