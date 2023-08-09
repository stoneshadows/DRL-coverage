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
           'joints angular speed 4', 'legs contact with ground 1', 'legs contact with ground 0', 'lidar rangefinder measurement 1',
           'lidar rangefinder measurement 2', 'lidar rangefinder measurement 3', 'lidar rangefinder measurement 4',
           'lidar rangefinder measurement 5','lidar rangefinder measurement 6', 'lidar rangefinder measurement 7', 'lidar rangefinder measurement 8',
           'lidar rangefinder measurement 9', 'lidar rangefinder measurement 10']
action_B = ['Joint torque 1', 'Joint torque 2', 'Joint torque 3', 'Joint torque 4']

state_L = ['horizontal coordinate','vertical coordinate','horizontal speed','vertical speed','angle','angular speed','first leg','second leg']
action_L = ['main engine', 'left-right engines']

def save_graph(env_name = 'BipedalWalker-v3',  episode = 10, called = 'state', label = state_B, dimension = 24, fig_width = 4,
    fig_height = 6,figsize = (30, 20)):


    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'


    env = gym.make(env_name)
    # make directory for saving figures
    figures_dir = "PPO_figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + env_name + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + called + '.png'

    total_num = episode
    dir = './PPO_preTrained/' + env_name + '/'

    plt.subplots(fig_width, fig_height, figsize = figsize)
    # dimension = env.observation_space.shape[0]
    for d in range(dimension):
        print(d)
        train_max_state, train_min_state, test_max_state, test_min_state = [[], [], [], []]
        for num in range(total_num):
            train_max_state.append(np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d])
            train_min_state.append(np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d])

            # test_max_state.append(np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d])
            # test_min_state.append(np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d])

        lists = [[train_max_state, train_min_state]]
        ls = [train_max_state, train_min_state]
        # labels = ['train_max_state', 'train_min_state', 'test_max_state', 'test_min_state']
        colors = ['g', 'r']
        cs = ['g', 'g', 'r', 'r']
        time = [i + 1 for i in range(len(train_max_state))]

        sns.set_style(style="darkgrid")
        ax = plt.subplot(fig_width, fig_height, 1 + d)

        for i,data in enumerate(lists):
            sns.tsplot(time=time, data=data, color=colors[i], linestyle='')
        for i, data in enumerate(ls):
            sns.tsplot(time=time, data=data, color=cs[i],linestyle='--')

        ax.tick_params(top=False, bottom=False, left=False, right=False) #刻度上的小尖尖
        plt.xticks(time)
        # plt.yticks(time)
        plt.ylabel(label[d])

        ax.grid(color='gray', linestyle='--')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.set_xlabel("Episodes")
        # ax.set_ylabel("Rewards")
        #
    # plt.legend(loc='best', facecolor='blue')

    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)
    # plt.title(env_name)
    plt.show()

if __name__ == '__main__':

    save_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
