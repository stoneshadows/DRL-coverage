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

def all_cov(env_name = 'BipedalWalker-v3',total_num = 6):

    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    d_a = env.action_space.shape[0]
    # num_action = env.action_space.n
    # print(num_action)
    figures_dir = 'figs' + '/'
    if not os.path.exists(figures_dir):
          os.makedirs(figures_dir)
    l_dir = './preTrained/'
    num_files = next(os.walk(l_dir))[1]
    # print(num_files)

    grids = [10, 50, 100, 500, 1000]
    # grids = [10]
    # keys = ['state', 'action', 'state-action']
    count_dict_state = dict([(k, []) for k in grids])
    count_dict_s_a= dict([(k, []) for k in grids])
    count_dict_a = dict([(k, []) for k in grids])
    cov_dict_a = dict([(k, []) for k in grids])
    cov_dict_state = dict([(k, []) for k in grids])
    cov_dict_s_a= dict([(k, []) for k in grids])
    time_dict = dict([('train_time', []),('test_time', [])])

    for i,algorithm in enumerate(num_files):
        # print(algorithm)
        dir = l_dir + algorithm + '/'
        for g in grids:
            print(g)
            test_state, test_action, test_state_action = [[], [], []]
            train_time = []
            test_time = []
            for num in range(total_num):
                test_state.append(len(np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                test_action.append(len(np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                test_state_action.append(len(np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                train_time.append(np.load("{}{}/Total_training_time.npy".format(dir, num), allow_pickle=True).tolist().total_seconds())
                test_time.append(np.load("{}{}/test_time.npy".format(dir, num),
                                          allow_pickle=True).tolist().total_seconds())


            # print(test_state,test_state_action)
            train_t = int(sum(train_time) / total_num)
            test_t = int(sum(test_time) / total_num)

            avg_state = int(sum(test_state)/total_num)
            avg_action = int(sum(test_action) / total_num)
            avg_state_action = int(sum(test_state_action)/total_num)
            # print(avg_state, avg_state_action)
            cov_s = avg_state / (g ** dimension)
            cov_a = avg_action / (g ** d_a)
            cov_s_a = avg_state_action/(g**(dimension+d_a))
            count_dict_state[g].append(avg_state)
            count_dict_a[g].append(avg_action)
            count_dict_s_a[g].append(avg_state_action)
            cov_dict_state[g].append(cov_s)
            cov_dict_a[g].append(cov_a)
            cov_dict_s_a[g].append(cov_s_a)


        time_dict['train_time'].append(datetime.timedelta(seconds=train_t))
        time_dict['test_time'].append(datetime.timedelta(seconds=test_t))




    df_s = pd.DataFrame(count_dict_state, index=num_files).T
    df_a = pd.DataFrame(count_dict_a, index=num_files).T
    df_sa = pd.DataFrame(count_dict_s_a, index=num_files).T
    df_cov_s = pd.DataFrame(cov_dict_state, index=num_files).T
    df_cov_a = pd.DataFrame(cov_dict_a, index=num_files).T
    df_cov_sa = pd.DataFrame(cov_dict_s_a, index=num_files).T
    df_time = pd.DataFrame(time_dict, index=num_files).T


    df_s.to_csv(figures_dir + 'num_state' + '.csv')
    df_a.to_csv(figures_dir + 'num_action' + '.csv')
    df_sa.to_csv(figures_dir + 'num_state_action' + '.csv')
    df_cov_s.to_csv(figures_dir + 'cov_state' + '.csv')
    df_cov_a.to_csv(figures_dir + 'cov_action' + '.csv')
    df_cov_sa.to_csv(figures_dir + 'cov_state_action' + '.csv')
    df_time.to_csv(figures_dir + 'time' + '.csv')

    print(df_s)
    print(df_sa)
    print(df_cov_s)
    print(df_cov_sa)
    print(df_time)


if __name__ == '__main__':
    all_cov()
    
    
    
    
    
    
    
    
    
    
    
    
    

