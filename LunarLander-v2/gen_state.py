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

def gen(env_name = 'LunarLander-v2',algorithm = 'TD3', episode = 10, called = 'state', label = state_L, dimension = 8, fig_width = 2,
    fig_height = 2,figsize = (10, 6)):
# def gen(env_name='CartPole-v1', algorithm='PPO', episode=10, called='state', dimension=4):

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
            gen_s = []
            for d in range(dimension):
                train_max_state = np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d]
                train_min_state = np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d]

                test_max_state = np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d]
                test_min_state = np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d]

                x = choice([np.random.uniform(train_min_state, test_min_state), np.random.uniform(test_max_state, train_max_state)])
                gen_s.append(x)
            print(gen_s)
            gen_s_all.append(gen_s)
        np.save("D:\RL\TD3-PyTorch-BipedalWalker-v2-master\L_gen_s_all.npy", gen_s_all)

    #         # test_state = np.load("{}{}/test_state_dict.npy".format(dir, num), allow_pickle=True).tolist()
    #         # test_action = np.load("{}{}/test_action_dict.npy".format(dir, num), allow_pickle=True).tolist()
    #         # test_state_action = np.load("{}{}/test_state_action.npy".format(dir, num), allow_pickle=True).tolist()
    #     avg_cov = (sum(test_max_state)/total_num - sum(test_min_state)/total_num) / (sum(train_max_state)/total_num - sum(train_min_state)/total_num)
    #     print(avg_cov)
    #     lists = [[train_max_state, train_min_state],[ test_max_state, test_min_state]]
    #     ls = [train_max_state, train_min_state, test_max_state, test_min_state]
    #     # labels = ['train_max_state', 'train_min_state', 'test_max_state', 'test_min_state']
    #     colors = ['g', 'r']
    #     cs = ['g', 'g', 'r', 'r']
    #     time = [i + 1 for i in range(len(train_max_state))]
    #
    #     sns.set_style(style="darkgrid")
    #     ax = plt.subplot(fig_width, fig_height, 1 + d)
    #
    #     for i,data in enumerate(lists):
    #         sns.tsplot(time=time, data=data, color=colors[i], linestyle='')
    #     for i, data in enumerate(ls):
    #         sns.tsplot(time=time, data=data, color=cs[i],linestyle='--')
    #
    #     ax.tick_params(top=False, bottom=False, left=False, right=False) #刻度上的小尖尖
    #     plt.xticks(time)
    #     # plt.yticks(time)
    #     plt.ylabel(label[d])
    #
    #     plt.text(x=7.5, y=(max(train_max_state)+min(train_min_state))/2, s="avg_cov={}".format(round(avg_cov, 3)), bbox={'facecolor': 'w','edgecolor':'gray', 'alpha': 0.8, 'boxstyle':'round'})
    #
    #     ax.grid(color='gray', linestyle='--')
    #     ax.spines['right'].set_color('none')
    #     ax.spines['top'].set_color('none')
    #     # ax.spines['left'].set_color('none')
    #     # ax.spines['bottom'].set_color('none')
    #     # ax.set_xlabel("Episodes")
    #     # ax.set_ylabel("Rewards")
    #     #
    # # plt.legend(loc='best', facecolor='blue')
    #     avg_cov_list.append(avg_cov)
    # avg_coverage = np.prod(avg_cov_list)
    # print("cov:", avg_coverage)
    # plt.suptitle(env_name + "_" + algorithm + " (SBCov = {})".format(round(avg_coverage, 5)))


    # plt.show()


if __name__ == '__main__':

    gen()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
