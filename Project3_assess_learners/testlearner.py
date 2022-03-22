""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import matplotlib.pyplot as plt
import RTLearner as rt
import BagLearner as bl
import time

  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		   	 		  		  		    	 		 		   		 		  
        sys.exit(1)
    if sys.argv[1] == "Data/Istanbul.csv":
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf, delimiter=',')
        data = data[1:, 1:]
    else:
        inf = open(sys.argv[1])
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
  		  	   		   	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")


    # create a learner and train it  		  	   		   	 		  		  		    	 		 		   		 		  
    #earner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    #learner.add_evidence(train_x, train_y)  # train it
    #print(learner.author())
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		   	 		  		  		    	 		 		   		 		  
    #pred_y = learner.query(train_x)  # get the predictions
    #rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    #print()
    #print("In sample results")
    #print(f"RMSE: {rmse}")
    #c = np.corrcoef(pred_y, y=train_y)
    #print(f"corr: {c[0,1]}")
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		   	 		  		  		    	 		 		   		 		  
    #pred_y = learner.query(test_x)  # get the predictions
    #rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    #print()
    #print("Out of sample results")
    #print(f"RMSE: {rmse}")
    #c = np.corrcoef(pred_y, y=test_y)
    #print(f"corr: {c[0,1]}")

    # Experiment 1. Decision Tree
    rmse_in = []
    rmse_out = []
    for i in range(0, 50):
        DT_tree = dt.DTLearner(leaf_size = i, verbose = False)
        DT_tree.add_evidence(train_x, train_y)
    #for i in range(1, 633):
        #print(DT_tree.tree[i,:])
        pred_y_in = DT_tree.query(train_x)
        rmse = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_in.append(rmse)

        pred_y_out = DT_tree.query(test_x)
        rmse = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_out.append(rmse)
    #print(rmse_in)
    #print(rmse_out)
    plt.figure(1)
    plt.plot(rmse_in)
    plt.plot(rmse_out)
    plt.title('Decision Tree, RMSE VS. Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('rmse')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    #plt.show()
    plt.savefig('Figure1')
    '''
    # 2. Random Tree
    rmse_in = []
    rmse_out = []
    for i in range(0, 50):
        RT_tree = rt.RTLearner(leaf_size=i, verbose=False)
        RT_tree.add_evidence(train_x, train_y)
        #for i in range(1, 633):
        #print(DT_tree.tree[i,:])
        pred_y_in = RT_tree.query(train_x)
        rmse = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_in.append(rmse)

        pred_y_out = RT_tree.query(test_x)
        rmse = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_out.append(rmse)
    print(rmse_in)
    print(rmse_out)
    plt.plot(rmse_in)
    plt.plot(rmse_out)
    plt.title('Random Tree, RMSE VS. Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('rmse')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    plt.show()
    plt.savefig('Figure2')
    '''
    # Experiment2. Baglearner
    rmse_in = []
    rmse_out = []
    for i in range(0,50):
        bag_learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size": i}
                                    ,bags = 50, boost = False, verbose = False)
        bag_learner.add_evidence(train_x, train_y)
        pred_y_in = bag_learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rmse_in.append(rmse)

        pred_y_out = bag_learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rmse_out.append(rmse)
    plt.figure(2)
    plt.plot(rmse_in)
    plt.plot(rmse_out)
    plt.title('Bag Learner, RMSE VS. Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('rmse')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    #plt.show()
    plt.savefig('Figure2')


    # Experiment 3. Decision Tree VS. Random Tree
    # 1. MAE: Using Mean Absolute Error to compare accuracy


    dt_MAE = []
    rt_MAE = []
    for i in range(0,50):
        dt_tree = dt.DTLearner(leaf_size = i, verbose = False)
        dt_tree.add_evidence(train_x, train_y)
        pred_y_out = dt_tree.query(test_x)
        MAE = (abs(test_y - pred_y_out)).sum() / test_y.shape[0]
        dt_MAE.append(MAE)
    for i in range(0,50):
        rt_tree = rt.RTLearner(leaf_size = i, verbose = False)
        rt_tree.add_evidence(train_x, train_y)
        pred_y_out = rt_tree.query(test_x)
        MAE = (abs(test_y - pred_y_out)).sum() / test_y.shape[0]
        rt_MAE.append(MAE)
    print(dt_MAE)
    print(rt_MAE)
    plt.figure(3)
    plt.plot(dt_MAE)
    plt.plot(rt_MAE)
    plt.title('MAE, DecisionTree VS. RandomTree')
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE')
    plt.legend(labels=['DT MAE', 'RT MAE'], loc='best')
    #plt.show()
    plt.savefig('Figure3')



    # 2.Training time: Compare training time DT VS. Rt
    dt_time = []
    rt_time = []
    for i in range(0,50):
        training_start = time.time()
        dt_tree = dt.DTLearner(leaf_size = i, verbose = False)
        dt_tree.add_evidence(train_x, train_y)
        training_end = time.time()
        dt_time.append(training_end - training_start)
    for i in range(0,50):
        training_start = time.time()
        rt_tree = rt.RTLearner(leaf_size = i, verbose = False)
        rt_tree.add_evidence(train_x, train_y)
        training_end = time.time()
        rt_time.append(training_end - training_start)
    plt.figure(4)
    plt.plot(dt_time)
    plt.plot(rt_time)
    plt.title('Training time, DecisionTree VS. RandomTree')
    plt.xlabel('Leaf Size')
    plt.ylabel('time')
    plt.legend(labels=['DT time', 'RT time'], loc='best')
    #plt.show()
    plt.savefig('Figure4')




