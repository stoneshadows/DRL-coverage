import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from random import choice

state_L = ['cart position', 'cart velocity', 'pole angle', 'pole angular velocity']
# action_L = ['main engine', 'left-right engines']

def gen(env_name = 'CartPole-v1',algorithm = 'D3QN', episode = 10, called = 'state', label = state_L, dimension = 4, fig_width = 2,
    fig_height = 2,figsize = (10, 6)):


    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'

    env = gym.make(env_name)
    # make directory for saving figures
    figures_dir = "figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + algorithm + '_fig_' + called + '.png'

    total_num = episode
    dir = './preTrained/' + algorithm + '/'

    avg_cov_list = []
    plt.subplots(fig_width, fig_height, figsize = figsize)
    # dimension = env.observation_space.shape[0]



    for num in range(1):
        gen_s_all = []
        for i in range(0, 1000):
            print(i)
            gen_s = []
            for d in range(dimension):
                train_max_state = np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d]
                train_min_state = np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d]

                test_max_state = np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d]
                test_min_state = np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d]

                x = choice([np.random.uniform(train_min_state, test_min_state), np.random.uniform(test_max_state, train_max_state)])
                gen_s.append(x)
            gen_s_all.append(gen_s)
        np.save("{}_gen_s_all.npy".format(algorithm), gen_s_all)


if __name__ == '__main__':

    gen()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
