from numpy.linalg import inv
import numpy as np

class NormalEquation:

    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_optimal(self):
        self.theta = inv(self.dataset.x.T.dot(self.dataset.x)).dot(self.dataset.x.T).dot(self.dataset.y)
        return self.theta

    def cost(self, theta):
        errors = (self.dataset.x.dot(theta) - self.dataset.y) ** 2
        return 0.5 * errors.sum() / errors.shape[0]

    def predict(self, x):
        return np.multiply(x, self.theta.T).sum()
