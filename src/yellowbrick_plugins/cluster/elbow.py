import time
import warnings

from overrides import overrides
from sklearn_plugins.cluster import SphericalKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.exceptions import YellowbrickWarning
from yellowbrick.utils import KneeLocator


class SphericalKElbowVisualizer(KElbowVisualizer):
    """Visualize elbow point for SphericalKMeans.

    SpehricalKMeans first transfrom the input with principal component analysis (PCA) and then calculates centroids based on the transformed input. However, `yellowbrick.cluster.KElbowVisualizer` calculates the metrics based on inputs prior to principal component analysis.

    In `SphericalKElbowVisualizer` is designed to fix abovementioned phenomena for `SphericalKMeans`. The `SphericalKElbowVisualizer.fit` method is overridden so that function `self.scoring_metric` can have X argument equla to `self.estimator.preprocess_input(X)`.
    """

    def __init__(self,
                 estimator: SphericalKMeans,
                 ax=None,
                 k=10,
                 metric="distortion",
                 timings=True,
                 locate_elbow=True,
                 **kwargs):
        super().__init__(estimator, ax, k, metric, timings, locate_elbow,
                         **kwargs)

    @overrides
    def fit(self, X, y=None, **kwargs):

        self.k_scores_ = []
        self.k_timers_ = []
        self.kneedle = None
        self.knee_value = None

        if self.locate_elbow:
            self.elbow_value_ = None
            self.elbow_score_ = None

        for k in self.k_values_:
            # Compute the start time for each  model
            start = time.time()

            # Set the k value and fit the model
            self.estimator.set_params(n_clusters=k)
            self.estimator.fit(X, **kwargs)

            # Append the time and score to our plottable metrics
            self.k_timers_.append(time.time() - start)
            self.k_scores_.append(
                self.scoring_metric(self.estimator.preprocess_input(X),
                                    self.estimator.labels_))

        if self.locate_elbow:
            locator_kwargs = {
                "distortion": {
                    "curve_nature": "convex",
                    "curve_direction": "decreasing",
                },
                "silhouette": {
                    "curve_nature": "concave",
                    "curve_direction": "increasing",
                },
                "calinski_harabasz": {
                    "curve_nature": "concave",
                    "curve_direction": "increasing",
                },
            }.get(self.metric, {})
            elbow_locator = KneeLocator(self.k_values_, self.k_scores_,
                                        **locator_kwargs)
            if elbow_locator.knee is None:
                self.elbow_value_ = None
                self.elbow_score_ = 0
                warning_message = (
                    "No 'knee' or 'elbow' point detected, "
                    "pass `locate_elbow=False` to remove the warning")
                warnings.warn(warning_message, YellowbrickWarning)
            else:
                self.elbow_value_ = elbow_locator.knee
                self.elbow_score_ = self.k_scores_[self.k_values_.index(
                    self.elbow_value_)]

        self.draw()

        return self
