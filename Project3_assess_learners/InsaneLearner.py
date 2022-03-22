import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose = False):
        for i in range(0,20):
            self.learners = bl.BagLearner(learner = lrl.LinRegLearner, bags = 20, kwargs={}, boost = False, verbose = False)
        pass
    def author(self):
        return 'chuang405'
    def add_evidence(self, data_x, data_y):
        for i in range(0,20):
            self.learners.add_evidence(data_x, data_y)
    def query(self, points):
        y_values = np.ones((20, points.shape[0]))
        for i in range(0, 20):
            ####
            y_values[i,] = self.learners.query(points)
        return np.mean(y_values, axis = 0)

