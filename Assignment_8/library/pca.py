import numpy as np


class PCA:

    def __init__(self, n_components):

        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, data):

        if not 1 <= self.n_components <= data.shape[1]:
            raise ValueError(
                "n_components must be between 1 and the number of features."
            )
        
        self.mean = data.mean(axis=0)
        
        centered = data - self.mean
        covariance = centered.T @ centered / len(data)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        order = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, order[:self.n_components]]
        
        return self

    def transform(self, data):
        
        return (data - self.mean) @ self.components