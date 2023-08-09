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

def all_cov2(env_name = 'Pendulum-v0',total_num = 6):

    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    # num_action = env.action_space.n
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
                    test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g),
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
                    test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g),
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
                    test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist().keys()
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

        for d in range(4):
            X = pd.Series(pd.to_numeric(df.iloc[:, d]).tolist())
            # X = pd.Series([0.993,0.993,0.942])
            Y = pd.Series([-621.08, -149.79, -186.76])
            # for v in ["pearson", "spearman", "kendall"]:
            for v in ["pearson", "spearman"]:
                print(v, '%.3f' % X.corr(Y, method=v))



        print(df)
        # df.to_csv(figures_dir + 'cov12_{}.csv'.format(g))

                # test_action.append(len(np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                # test_state_action.append(len(np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))

if __name__ == '__main__':
    all_cov2()
    
    
    
    
    
    
    
    
    
    
    
    
    

