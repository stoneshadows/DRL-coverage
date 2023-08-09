import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
from datetime import datetime
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def all_cov2(env_name = 'BipedalWalker-v3',total_num = 1):

    env = gym.make(env_name)
    # dimension = env.observation_space.shape[0]
    dimension = 4
    # print(num_action)
    figures_dir = 'figs' + '/'+'cov12/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    l_dir = './preTrained/'
    num_files = next(os.walk(l_dir))[1]
    # print(num_files)

    grids = [10, 50, 100, 500, 1000]
    # grids = [10]

    cov_dict = {}
    indexs = ('single', 'pairwise 1', 'pairwise 2', 'pairwise 3')
    for g in grids:
        print(g)
        for i,algorithm in enumerate(num_files):
            # print(algorithm)
            dir = l_dir + algorithm + '/'

            all_cov = []
            ############################
            dim_list = []
            for d in range(dimension):
                dim = {}
                for num in range(total_num):
                    test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist().keys()
                    # print(test_state)
                    for test in test_state:
                        test = test.split()

                        dim[test[d]] = 1
                dim_cov = len(dim)/g
                dim_list.append(dim_cov)
            # print(np.mean(dim_list))
    #############################################

            dim2_l1 = []
            for d in range(int(dimension/2)):
                dim2_1 = {}
                for num in range(total_num):
                    test_state = np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g),
                                         allow_pickle=True).tolist().keys()
                    for test in test_state:
                        test = test.split()
                        dim2_1[list(zip(test, test[1:]))[::2][d]] = 1

                dim2_cov1 = len(dim2_1) / (g * g)
                dim2_l1.append(dim2_cov1)
            # print(dim2_l1)

    ###################################################

            dim2_l2 = []
            for d in range(dimension - 1):
                dim2_2 = {}
                for num in range(total_num):
                    test_state = np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g),
                                         allow_pickle=True).tolist().keys()
                    for test in test_state:
                        test = test.split()
                        dim2_2[list(zip(test, test[1:]))[d]] = 1
                dim2_cov2 = len(dim2_2) / (g * g)
                dim2_l2.append(dim2_cov2)
            # print(dim2_l2)

    ###########################################

            dim2_all = []
            for d in range(dimension * (dimension-1)):
                dim2_l = {}
                for num in range(total_num):
                    test_state = np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist().keys()
                    for test in test_state:
                        test = test.split()
                        # print(list(itertools.permutations(test, 2)))
                        dim2_l[list(itertools.permutations(test, 2))[d]] = 1
                dim_covl = len(dim2_l)/(g*g)
                dim2_all.append(dim_covl)
            # print(dim2_all)

            all_cov = ['%.3f' % np.mean(dim_list), '%.3f' % np.mean(dim2_l1), '%.3f' % np.mean(dim2_l2), '%.3f' %  np.mean(dim2_all)]
            cov_dict[algorithm] = all_cov
            df = pd.DataFrame(cov_dict, index=indexs).T
        print(df)
        df.to_csv(figures_dir + 'action_cov12_{}.csv'.format(g))
        for d in range(dimension):
            X = pd.Series(pd.to_numeric(df.iloc[:, d]).tolist())
            X1 = pd.Series([0.847,0.856,0.778])
            X2 = pd.Series([0.015, 0.018, 0.001])
            # X3 = pd.Series([6.86E-20, 1.37E-20,1.82E-20])
            #
            # A1 = pd.Series([0.425, 0.999, 1])
            # A2 = pd.Series([0.033, 0.996, 1])
            # A3 = pd.Series([0.028, 0.365, 0.199])
            # Y = pd.Series([278.22, 269.49, 281.27])
            Y = pd.Series([1.85, 5.12, 2.17])
            for v in ["pearson", "spearman", "kendall"]:
                # print(v, '%.3f' % X.corr(Y, method=v))
                print(v, '%.3f' % X1.corr(Y, method=v))
                print(v, '%.3f' % X2.corr(Y, method=v))
            #     print(v, '%.3f' % X3.corr(Y, method=v))
            # print('#########################################')
            # for v in ["pearson", "spearman", "kendall"]:
            #     print(v, '%.3f' % A1.corr(Y, method=v))
            #     print(v, '%.3f' % A2.corr(Y, method=v))
            #     print(v, '%.3f' % A3.corr(Y, method=v))

                # test_action.append(len(np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                # test_state_action.append(len(np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))

if __name__ == '__main__':
    all_cov2()
    
    
    
    
    
    
    
    
    
    
    
    
    

