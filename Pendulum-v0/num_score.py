import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
from datetime import datetime
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import itertools

def all_cov(env_name = 'Pendulum-v0',total_num = 6,called = 'state'):
    import matplotlib.pyplot as plt
    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    d_a = env.action_space.shape[0]
    # print(num_action)
    figures_dir = 'figs' + '/'
    l_dir = './preTrained/'
    num_files = next(os.walk(l_dir))[1]
    # for i, algorithm in enumerate(num_files):
    #     print(algorithm)
    algorithm = "TD33"
    dir = l_dir + algorithm + '/'

    score_list =[]
    ##################################
    for num in range(0,6):
        score_list.extend(np.load("{}{}/scores.npy".format(dir, num), allow_pickle=True).tolist())
    x = [i for i in range(len(score_list))]
    score_list.sort()
    print(len(score_list))

    plt.plot(x, score_list)

    len_3 = int(len(score_list)* 0.03)
    len_10 = int(len(score_list) * 0.1)
    len_20 = int(len(score_list) * 0.2)

    plt.text(0, score_list[0], ("min", round(score_list[0],1)))
    plt.text(len_3, score_list[len_3], ("3%", round(score_list[len_3], 1)))
    plt.text(len_20, score_list[len_20], ("20%", round(score_list[len_20], 1)))
    plt.text(len_10, score_list[len_10], ("10%", round(score_list[len_10], 1)))
    plt.text(int(len(score_list) - 500), score_list[-1], ("max", round(score_list[-1], 1)))
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("{}_{}".format(env_name,algorithm))
    plt.savefig(figures_dir+algorithm+'.pdf')
    plt.close()
    # plt.show()##################################################

    cov_dict = {}

    indexs = ('3%({})'.format(round(score_list[len_3], 1)), '10%({})'.format(round(score_list[len_10], 1)),
              '20%({})'.format(round(score_list[len_20], 1)),
              'SBCov1','SBCov2','KSCov(%)','SACov(%)','Single','Link','Cross','Mutual')

    g = 10
    for num in range(total_num):
    # for num in range(5, 9):
        print(num)

        dim_list = []
        for d in range(dimension):
            dim = {}
            # for num in range(total_num):
            test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist().keys()
            for test in test_state:
                test = test.split()
                dim[test[d]] = 1
            dim_cov = len(dim) / g
            dim_list.append(dim_cov)
        print("single:", np.mean(dim_list))

        #############################################

        dim2_l1 = []
        for d in range(int(dimension / 2)):  # 每个维度
            dim2_1 = {}  # 每个文件夹中所有状态，重新记录
            # for num in range(total_num):
            test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g),
                                 allow_pickle=True).tolist().keys()
            for test in test_state:
                test = test.split()
                dim2_1[list(zip(test, test[1:]))[::2][d]] = 1

            dim2_cov1 = len(dim2_1) / (g * g)
            dim2_l1.append(dim2_cov1)  # 每个维度
        print("link:", np.mean(dim2_l1))

        #############################################

        dim2_l2 = []
        for d in range(dimension - 1):
            dim2_2 = {}
            # for num in range(total_num):
            test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g),
                                 allow_pickle=True).tolist().keys()
            for test in test_state:
                test = test.split()
                dim2_2[list(zip(test, test[1:]))[d]] = 1
            dim2_cov2 = len(dim2_2) / (g * g)
            dim2_l2.append(dim2_cov2)
        print("cross:", np.mean(dim2_l2))

        ###########################################

        dim2_all = []
        for d in range(dimension * (dimension - 1)):
            dim2_l = {}
            # for num in range(total_num):
            test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist().keys()
            for test in test_state:
                test = test.split()
                # print(list(itertools.permutations(test, 2)))
                dim2_l[list(itertools.permutations(test, 2))[d]] = 1
            dim_covl = len(dim2_l) / (g * g)
            dim2_all.append(dim_covl)
        print("mutual:", np.mean(dim2_all))


        # all_cov = ['%.3f' % np.mean(dim_list), '%.3f' % np.mean(dim2_l1), '%.3f' % np.mean(dim2_l2),
        #            '%.3f' % np.mean(dim2_all)]

        score_l = np.load("{}{}/scores.npy".format(dir, num), allow_pickle=True).tolist()
        score_l.sort()

        # print(score_list)
        # print(max(score_list))
        # print(min(state_list))


        print("test_state:",
              len(np.load("{}{}/test_state_dict_10.npy".format(dir, num), allow_pickle=True).tolist()))
        s_num= len(np.load("{}{}/test_state_dict_10.npy".format(dir, num), allow_pickle=True).tolist())
        # print("test_action:",
        #       len(np.load("{}{}/test_action_dict_100.npy".format(dir, num), allow_pickle=True).tolist()))
        print("test_state_action:",
              len(np.load("{}{}/test_state_action_10.npy".format(dir, num), allow_pickle=True).tolist()))
        s_a_num=len(np.load("{}{}/test_state_action_10.npy".format(dir, num), allow_pickle=True).tolist())
        avg_cov_l = []
        for d in range(dimension):
            test_max_s = np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d]
            test_min_s = np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d]

            train_max_s = np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d]
            train_min_s = np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d]

            # grid_state = discretize(state, state_grid)

            # 每一轮的
            avg_cov_d = (test_max_s - test_min_s) / (train_max_s - train_min_s)
            # print(avg_cov_d)

            avg_cov_l.append(avg_cov_d)
        avg_cover1 = sum(avg_cov_l) / dimension
        avg_cover2 = np.prod(avg_cov_l)
        print("cov1:", avg_cover1)
        print("cov2:", avg_cover2)

        # train_max = np.load("{}{}/max_{}_list.npy".format(dir, num, called))
        # train_min = np.load("{}{}/min_{}_list.npy".format(dir, num, called))
        # print(train_min)
        # print(train_max)
        cov_dict[num] = [sum(i < score_list[len_3] for i in score_l),
                         sum(i < score_list[len_10] for i in score_l),
                         sum(i < score_list[len_20] for i in score_l),'%.3f' % avg_cover1,
                         '%.3f' % avg_cover2,
                         s_num/100, s_a_num/100,
                         '%.3f' % np.mean(dim_list), '%.3f' % np.mean(dim2_l1), '%.3f' % np.mean(dim2_l2),
                    '%.3f' % np.mean(dim2_all)]
        df = pd.DataFrame(cov_dict, index=indexs).T
    print(df)
    # df.to_csv(figures_dir + 'add_exp_{}.csv'.format(algorithm))


if __name__ == '__main__':
    all_cov()
    
    
    
    
    
    
    
    
    
    
    
    
    

