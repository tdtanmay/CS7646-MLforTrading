import numpy as np
import random
import RTLearner as rt

class BagLearner(object):
    def __init__(self, learner = rt.RTLearner, kwargs = {}, bags = 10, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs))
        pass
    def author(self):
        return 'chuang405'

    def add_evidence(self, data_x, data_y):
        # slap on 1s column so linear regression finds a constant term
        data_all = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        data_all[:, 0: data_x.shape[1]] = data_x
        data_all[:, -1] = data_y
        for i in range(0, self.bags):
            index = np.random.randint(0, data_all.shape[0], data_all.shape[0])
            data_x_i = data_all[index, :-1]
            data_y_i = data_all[index, -1]
            self.learners[i].add_evidence(data_x_i,data_y_i)


    def query(self, points):
        y_values = np.ones((self.bags, points.shape[0]))
        for i in range(0, self.bags):
            ####
            y_values[i,] = self.learners[i].query(points)
        return np.mode(y_values, axis = 0)

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")


