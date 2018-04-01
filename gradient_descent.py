import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:

    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_optimal(self, alpha):
        self.theta = np.random.randn(self.dataset.number_of_features, 1)
        
        old_cost = 1
        for i in range(0, 201):

            self.temp_theta = np.zeros([self.dataset.number_of_features, 1])

            for column in range(0, self.dataset.number_of_features):
                self.temp_theta[column][0] = self.compute_gradient(column)

            self.theta = self.theta - alpha * \
                            self.temp_theta / self.dataset.number_of_samples

            if (i % 5) == 0:
                cost = self.cost(self.theta)
                cost_variation = (old_cost - cost) / old_cost
                print(i, cost, cost_variation)
                old_cost = cost

        np.set_printoptions(suppress = True)
        print(self.theta)

    def compute_gradient(self, column):
        return (self.dataset.x.dot(self.theta) - self.dataset.y).T.dot(self.dataset.x[:, column:column+1])

    def cost(self, theta):
        errors = (self.dataset.x.dot(theta) - self.dataset.y) ** 2
        return 0.5 * errors.sum() / errors.shape[0]

    def predict(self, x):
        return np.multiply(x, self.theta.T).sum()
