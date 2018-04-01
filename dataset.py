import pandas
import math
import numpy as np
import numpy_indexed as npi

class Dataset:

    def __init__(self, filename, features, target):
        self.filename = filename
        self.features = features
        self.target = target
        self.use_target = not not self.target

    def load(self):
        dataset = pandas.read_csv(self.filename)
        x = dataset.loc[:, self.features]
        self.x = np.array(x)
        self.number_of_samples = self.x.shape[0]
        self.number_of_features = self.x.shape[1]
        if self.use_target:
            y = dataset.loc[:, self.target]
            self.y = np.array(y)[:, None]

    def normalize_x(self, features):
        np_all_features = np.array(self.features)
        np_features_to_normalize = np.array(features)
        columns_to_normalize = npi.indices(np_all_features, np_features_to_normalize)

        for column in columns_to_normalize:
            mean = self.x[:,column:column+1].mean()
            sd = self.x[:,column:column+1].std()
            self.x[:,column:column+1] = (self.x[:,column:column+1] - mean) / sd

    def export_to_csv(self, filename):
        if self.use_target:
            data = np.concatenate((self.x,self.y), axis = 1)
            columns = self.features + [self.target]
        else:
            data = self.x
            columns = self.features

        data_frame = pandas.DataFrame(data = data, columns = columns)

        data_frame.to_csv(filename, index = False)
