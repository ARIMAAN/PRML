
import numpy as np
from .kmeans import KMeans


class GMM:

    def __init__(self, n_components=3, regularization=1e-6,
                 iterations=100, convergence_tol=1e-6):

        self.n_components = n_components
        self.regularization = regularization
        self.iterations = iterations
        self.convergence_tol = convergence_tol

        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihoods = []

    def component_scores(self, data):

        sample_count, feature_count = data.shape
        log_scores = np.empty((sample_count, self.n_components))

        for component in range(self.n_components):

            covariance = self.covariances[component]
            eigenvalues = np.linalg.eigvalsh(covariance)

            if np.any(eigenvalues <= 0):
                raise ValueError(
                    "Covariance must be positive definite. "
                    "Try increasing regularization."
                )

            log_determinant = np.sum(np.log(eigenvalues))
            inverse_matrix = np.linalg.inv(covariance)

            centered_data = data - self.means[component]

            mahalanobis_distance = np.sum(
                (centered_data @ inverse_matrix) * centered_data,
                axis=1
            )

            log_scores[:, component] = (
                np.log(self.weights[component])
                - 0.5 * (
                    feature_count * np.log(2 * np.pi)
                    + log_determinant
                    + mahalanobis_distance
                )
            )

        return log_scores

    def score_samples(self, data):

        component_log_probs = self.component_scores(data)

        row_maximum = np.max(component_log_probs, axis=1)

        stable_log_sum = np.log(
            np.sum(
                np.exp(component_log_probs - row_maximum[:, None]),
                axis=1
            )
        )

        return row_maximum + stable_log_sum

    def responsibilities(self, data):

        log_probabilities = self.component_scores(data)

        log_probabilities -= np.max(
            log_probabilities, axis=1, keepdims=True
        )

        probability_matrix = np.exp(log_probabilities)

        normalized_probabilities = (
            probability_matrix /
            np.sum(probability_matrix, axis=1, keepdims=True)
        )

        return normalized_probabilities

    def fit(self, data):

        total_samples, feature_count = data.shape

        # Initialize Gaussian components using K-Means
        initializer = KMeans(n_clusters=self.n_components).fit(data)
        cluster_assignments = initializer.predict(data)

        self.means = initializer.centers.copy()

        self.weights = np.zeros(self.n_components)

        self.covariances = np.zeros(
            (self.n_components, feature_count, feature_count)
        )

        self.log_likelihoods = []

        identity_matrix = np.eye(feature_count)

        # Initial parameter estimation
        for component in range(self.n_components):

            cluster_data = data[
                cluster_assignments == component
            ]

            cluster_size = len(cluster_data)

            if cluster_size == 0:
                raise ValueError(
                    "Empty cluster detected during initialization."
                )

            centered_cluster = (
                cluster_data - self.means[component]
            )

            self.weights[component] = (
                cluster_size / total_samples
            )

            self.covariances[component] = (
                centered_cluster.T @ centered_cluster / cluster_size
                + self.regularization * identity_matrix
            )

        # Expectation-Maximization optimization
        for iteration in range(self.iterations):

            # E-step: Calculate membership probabilities
            membership = self.responsibilities(data)

            # M-step: Update Gaussian parameters
            effective_counts = np.sum(membership, axis=0)

            if np.any(effective_counts < 1e-10):
                raise ValueError(
                    "A Gaussian component has insufficient membership."
                )

            self.weights = effective_counts / total_samples

            self.means = (
                membership.T @ data
            ) / effective_counts[:, None]

            for component in range(self.n_components):

                deviation = data - self.means[component]

                weighted_deviation = (
                    membership[:, component, None] * deviation
                )

                self.covariances[component] = (
                    deviation.T @ weighted_deviation
                    / effective_counts[component]
                    + self.regularization * identity_matrix
                )

            # Evaluate average log-likelihood
            current_likelihood = np.mean(
                self.score_samples(data)
            )

            self.log_likelihoods.append(current_likelihood)

            # Check convergence
            if iteration > 0:

                likelihood_change = abs(
                    self.log_likelihoods[-1]
                    - self.log_likelihoods[-2]
                )

                if likelihood_change < self.convergence_tol:
                    break

        return self
