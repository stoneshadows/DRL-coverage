import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

state_B = ['hull angle','angular velocity','horizontal speed','vertical speed','position of joints 1','position of joints 2',
           'position of joints 3','position of joints 4','joints angular speed 1','joints angular speed 2','joints angular speed 3',
           'joints angular speed 4', 'legs contact with ground 1', 'legs contact with ground 0', '1 lidar rangefinder measurements',
           '2 lidar rangefinder measurements', '3 lidar rangefinder measurements', '4 lidar rangefinder measurements',
           '5 lidar rangefinder measurements','6 lidar rangefinder measurements', '7 lidar rangefinder measurements', '8 lidar rangefinder measurements',
           '9 lidar rangefinder measurements', '10 lidar rangefinder measurements']
action_B = ['Joint torque 1', 'Joint torque 2', 'Joint torque 3', 'Joint torque 4']

state_L = ['horizontal coordinate','vertical coordinate','horizontal speed','vertical speed','angle','angular speed','first leg','second leg']
action_L = ['main engine', 'left-right engines']

def save_graph(env_name = 'LunarLander-v2',num = 0):


    dir = './PPO_preTrained/' + env_name + '/'

    grids = [10, 50, 100, 500, 1000]
    keys = ['state', 'action', 'state-action']
    count_dict_all = dict([(k, []) for k in keys])
    for g in grids:
        test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        print("test_state_{}: ".format(g),len(test_state))
        vs = 0
        for v in test_state.values():
            vs += v
        count_dict_all['state'].append(vs)


        test_action = np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        print("test_action_{}: ".format(g),len(test_action))
        va = 0
        for v in test_action.values():
            va += v

        count_dict_all['action'].append(va)
        test_state_action = np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        print("state-action_{}: ".format(g),len(test_state_action))
        vsa = 0
        for v in test_state_action.values():
            vsa += v
        count_dict_all['state-action'].append(vsa)
    print(count_dict_all)

def jaccard(p,q):
    c = [v for v in p if v in q]
    return float(len(c))/(len(p)+len(q)-len(c))

def read_trace_state(env_name = 'LunarLander-v2',num = 0):

    dir = './PPO_preTrained/' + env_name + '/'

    # grids = [10, 50, 100, 500, 1000]
    # for g in grids:
    #     trace_state = np.load("{}{}/trace_state_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
    #     print("trace_state_{}: ".format(g),trace_state)
    trace_state = np.load("{}{}/trace_state_50.npy".format(dir, num), allow_pickle=True).tolist()
    test_score = np.load("{}{}/test_scores.npy".format(dir, num), allow_pickle=True).tolist()
    # print(np.argwhere(np.array(test_score) < 100))
    print(jaccard(trace_state[1], trace_state[2]))


    # for v in trace_state[2]:
    #     print(v)
    print(len(trace_state[2]))
    print(len(trace_state[5]))
    # for tra in trace_state:  # 50个路径, 每个路径包含多个状态
    #     print(trace_state)
        # for s in tra:   # 每个路径中的状态, 计算每两个路径中相同的状态个数和总的状态个数, 计算两个路径的状态相似度
        #     print(tra)

if __name__ == '__main__':

    read_trace_state()
    
    
    
    
    
    
    
    
    
    
    
    
    

