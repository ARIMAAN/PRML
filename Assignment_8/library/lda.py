import numpy as np


class LDA:

    def __init__(self, regularization=1e-3):

        self.regularization = regularization
        self.classes = None
        self.means = None
        self.covariance = None
        self.weights = None
        self.bias = None

    def fit(self, data, labels):

        self.classes = np.unique(labels)
        
        sample_count, feature_count = data.shape
        
        means = []
        priors = []
        
        covariance = np.zeros((feature_count, feature_count))

        for label in self.classes:

            samples = data[labels == label]
            mean = samples.mean(axis=0)
            
            means.append(mean)
            priors.append(len(samples) / sample_count)
            
            difference = samples - mean
            covariance += difference.T @ difference

        self.means = np.array(means)
        
        self.covariance = (
            covariance / sample_count 
            + self.regularization * np.eye(feature_count)
        )
        
        self.weights = np.linalg.solve(self.covariance, self.means.T).T
        
        self.bias = (-0.5 * np.sum(self.means * self.weights, axis=1) + np.log(priors))
        
        return self

    def predict(self, data):

        scores = data @ self.weights.T + self.bias
        
        return self.classes[np.argmax(scores, axis=1)]