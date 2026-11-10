
import numpy as np


class KMeans:

    def __init__(self, n_clusters=3):

        self.n_clusters = n_clusters
        self.centers = None

    def predict(self, data):

        if self.centers is None:
            raise ValueError("Call fit() before predict().")

        assigned_labels = []

        for sample in data:

            squared_distances = np.sum(
                (self.centers - sample) ** 2,
                axis=1
            )

            closest_cluster = np.argmin(squared_distances)
            assigned_labels.append(closest_cluster)

        return np.asarray(assigned_labels, dtype=int)

    def fit(self, data):

        random_generator = np.random.default_rng(42)

        selected_indices = random_generator.choice(
            data.shape[0],
            size=self.n_clusters,
            replace=False
        )

        self.centers = data[selected_indices].astype(float)

        max_iterations = 100
        tolerance = 1e-6

        for iteration in range(max_iterations):

            # Assign each sample to its nearest centroid
            cluster_labels = self.predict(data)

            updated_centers = self.centers.copy()

            # Recalculate centroids
            for cluster_id in range(self.n_clusters):

                cluster_members = data[
                    cluster_labels == cluster_id
                ]

                if cluster_members.shape[0] != 0:

                    updated_centers[cluster_id] = np.mean(
                        cluster_members,
                        axis=0
                    )

            # Check whether centroid movement is sufficiently small
            movement = np.max(
                np.abs(updated_centers - self.centers)
            )

            self.centers = updated_centers

            if movement < tolerance:
                break

        return self
