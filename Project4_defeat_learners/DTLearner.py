""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  

-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import numpy as np


class DTLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leafsize = leaf_size
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "chuang405"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # slap on 1s column so linear regression finds a constant term
        data_all = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        data_all[:, 0: data_x.shape[1]] = data_x
        data_all[:, -1] = data_y

        # build and save the model
        self.tree = self.build_tree(data_all)

    def build_tree(self, data):
        """
        using Decision Tree Algo developed mby Quinlan
        """
        if data.shape[0] <= self.leafsize or np.unique(data[:, -1]).shape[0] == 1:
            return np.array([[-1, data[0][-1], None, None]])
        else:
            correlation_array = []
            for i in range(0, data.shape[1] - 1):
                cor_1 = np.corrcoef(data[:, i].transpose(), data[:, -1].transpose())
                cor_2 = cor_1[0, 1]
                correlation_array.append(cor_2)
            correlation_array = np.abs(correlation_array)
            best_feature = np.argmax(correlation_array)
            splitval = np.median(data[:, best_feature])

            if splitval == np.max(data[:,best_feature]):
                q = np.argmax(data[:,best_feature])
                return np.array([[-1, data[q][-1], None, None]])

            maskleft = data[:, best_feature] <= splitval
            maskright = data[:, best_feature] > splitval
            lefttree = self.build_tree(data[maskleft])
            righttree = self.build_tree(data[maskright])
            root = np.array([best_feature, splitval, 1, lefttree.shape[0] + 1])
            return np.vstack((root,lefttree,righttree))


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        y_array = []
        for i in points:
            y_array.append(self.calltree(i, root=0))
        return np.array(y_array)

    def calltree(self, i, root):
        root_fac = int(self.tree[root,0])
        root_SV = self.tree[root, 1]
        if root_fac == -1:
            return root_SV

        elif i[root_fac] <= root_SV:
            next_root = int(root + self.tree[root, 2])
        else:
            next_root = int(root + self.tree[root, 3])

        return self.calltree(i, next_root)





if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

