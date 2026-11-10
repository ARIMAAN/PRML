
import numpy as np


class KNN:

    def __init__(self, neighbors=5):

        self.neighbors = neighbors
        self.training_data = None
        self.training_labels = None

    def predict(self, samples):

        predicted_classes = []

        for sample in samples:

            # Calculate squared Euclidean distances
            distance_values = np.sum(
                (self.training_data - sample) ** 2,
                axis=1
            )

            # Select indices of the nearest neighbors
            nearest_indices = np.argsort(distance_values)[
                :self.neighbors
            ]

            # Collect labels of selected neighbors
            neighbor_labels = self.training_labels[nearest_indices]

            # Find the most frequently occurring class
            class_values, class_frequencies = np.unique(
                neighbor_labels,
                return_counts=True
            )

            majority_class = class_values[
                np.argmax(class_frequencies)
            ]

            predicted_classes.append(majority_class)

        return np.asarray(predicted_classes)

    def fit(self, features, targets):

        self.training_labels = np.asarray(targets)
        self.training_data = np.asarray(features)

        return self
