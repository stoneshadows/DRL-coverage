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

state_B = ['hull angle','angular velocity','horizontal speed','vertical speed','position of joints 1','position of joints 2',
           'position of joints 3','position of joints 4','joints angular speed 1','joints angular speed 2','joints angular speed 3',
           'joints angular speed 4', 'legs contact with ground 1', 'legs contact with ground 0', '1 lidar rangefinder measurements',
           '2 lidar rangefinder measurements', '3 lidar rangefinder measurements', '4 lidar rangefinder measurements',
           '5 lidar rangefinder measurements','6 lidar rangefinder measurements', '7 lidar rangefinder measurements', '8 lidar rangefinder measurements',
           '9 lidar rangefinder measurements', '10 lidar rangefinder measurements']
action_B = ['Joint torque 1', 'Joint torque 2', 'Joint torque 3', 'Joint torque 4']

state_L = ['horizontal coordinate','vertical coordinate','horizontal speed','vertical speed','angle','angular speed','first leg','second leg']
action_L = ['main engine', 'left-right engines']

def save_graph(env_name = 'LunarLander-v2', num = 9):

    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    num_action = env.action_space.n
    print(num_action)

    dir = './preTrained/' + env_name + '/'

    grids = [10, 50, 100, 500, 1000]
    keys = ['state', 'action', 'state-action']
    count_dict_all = dict([(k, []) for k in keys])

    log_f_name = "./figs/" + env_name + "/cover" + ".csv"
    log_f = open(log_f_name,"w+")
    log_f.write('grids,state,state_cov,state-action,state_action_cov\n')
    for g in grids:
        test_state = np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        print("state_{}: ".format(g),len(test_state))
        vs = 0
        for v in test_state.values():
            vs += v
        count_dict_all['state'].append(vs)


        # test_action = np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        # print("test_action_{}: ".format(g),len(test_action))
        # va = 0
        # for v in test_action.values():
        #     va += v

        # count_dict_all['action'].append(va)
        test_state_action = np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()
        print("state-action_{}: ".format(g),len(test_state_action))
        vsa = 0
        for v in test_state_action.values():
            vsa += v
        count_dict_all['state-action'].append(vsa)

        log_f.write('{},{},{},{},{}\n'.format(g, len(test_state), len(test_state)/(g**dimension), len(test_state_action),len(test_state_action)/(num_action*g**dimension)))
        log_f.flush()

    print(count_dict_all)
    log_f.close()


def avg_cov(env_name = 'LunarLander-v2',total_num = 10, algorithm = 'DQN',called = 'state'):

    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    num_action = env.action_space.n
    print(num_action)
    dir = './preTrained/' + algorithm + '/'

    grids = [10, 50, 100, 500, 1000]
    keys = ['state', 'action', 'state-action']
    count_dict_all = dict([(k, []) for k in grids])
    print()

    log_f_name = "./figs/" + env_name + "/"+ algorithm +"_cover" + ".csv"
    log_f = open(log_f_name,"w+")
    log_f.write('grids,state,state_cov,state-action,state_action_cov\n')
    for g in grids:
        test_state, test_action, test_state_action = [[], [], []]
        for num in range(total_num):
            test_state.append(len(np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
            # test_action.append(len(np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
            test_state_action.append(len(np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
        print(test_state,test_state_action)
        avg_state = int(sum(test_state)/total_num)
        avg_state_action = int(sum(test_state_action)/total_num)
        print(avg_state, avg_state_action)


        log_f.write('{},{},{},{},{}\n'.format(g, avg_state, avg_state/(g**dimension),
                                              avg_state_action,avg_state_action/(num_action*g**dimension)))
        log_f.flush()

    # print(count_dict_all)
    log_f.close()

def all_cov(env_name = 'LunarLander-v2',total_num = 6,called = 'state'):

    env = gym.make(env_name)
    dimension = env.observation_space.shape[0]
    num_action = env.action_space.n
    # print(num_action)
    figures_dir = 'figs' + '/'
    l_dir = './preTrained/'
    num_files = next(os.walk(l_dir))[1]
    # print(num_files)

    grids = [10, 50, 100, 500, 1000]
    # keys = ['state', 'action', 'state-action']
    count_dict_state = dict([(k, []) for k in grids])
    count_dict_s_a= dict([(k, []) for k in grids])
    cov_dict_state = dict([(k, []) for k in grids])
    cov_dict_s_a= dict([(k, []) for k in grids])
    time_dict = dict([('train_time', []),('test_time', [])])

    for i,algorithm in enumerate(num_files):
        # print(algorithm)
        dir = l_dir + algorithm + '/'
        for g in grids:
            test_state, test_action, test_state_action = [[], [], []]
            train_time = []
            test_time = []
            for num in range(total_num):
                test_state.append(len(np.load("{}{}/test_state_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                # test_action.append(len(np.load("{}{}/test_action_dict_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                test_state_action.append(len(np.load("{}{}/test_state_action_{}.npy".format(dir, num, g), allow_pickle=True).tolist()))
                train_time.append(np.load("{}{}/Total_training_time.npy".format(dir, num), allow_pickle=True).tolist().total_seconds())
                test_time.append(np.load("{}{}/test_time.npy".format(dir, num),
                                          allow_pickle=True).tolist().total_seconds())


            # print(test_state,test_state_action)
            train_t = int(sum(train_time) / total_num)
            test_t = int(sum(test_time) / total_num)

            avg_state = int(sum(test_state)/total_num)
            avg_state_action = int(sum(test_state_action)/total_num)
            # print(avg_state, avg_state_action)
            cov_s = avg_state / (2*g ** (dimension-2))
            cov_s_a = avg_state_action/(2*num_action*g**(dimension-2))
            count_dict_state[g].append(avg_state)
            count_dict_s_a[g].append(avg_state_action)
            cov_dict_state[g].append(cov_s)
            cov_dict_s_a[g].append(cov_s_a)
        time_dict['train_time'].append(datetime.timedelta(seconds=train_t))
        time_dict['test_time'].append(datetime.timedelta(seconds=test_t))

    df_s = pd.DataFrame(count_dict_state, index=num_files).T
    df_sa = pd.DataFrame(count_dict_s_a, index=num_files).T
    df_cov_s = pd.DataFrame(cov_dict_state, index=num_files).T
    df_cov_sa = pd.DataFrame(cov_dict_s_a, index=num_files).T
    df_time = pd.DataFrame(time_dict, index=num_files).T


    # df_s.to_csv(figures_dir + 'num_state' + '.csv')
    # df_sa.to_csv(figures_dir + 'num_state_action' + '.csv')
    df_cov_s.to_csv(figures_dir + 'cov_state' + '.csv')
    df_cov_sa.to_csv(figures_dir + 'cov_state_action' + '.csv')
    # df_time.to_csv(figures_dir + 'time' + '.csv')

    print(df_s)
    print(df_sa)
    print(df_cov_s)
    print(df_cov_sa)
    print(df_time)


if __name__ == '__main__':
    all_cov()
    
    
    
    
    
    
    
    
    
    
    
    
    

