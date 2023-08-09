import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


state_L = ['cos(theta)', 'sin(theta)', 'theta dot']
action_L = ['Joint effort']
#figsize = (5, 3) DDPG SAC TD3
def save_graph(env_name = 'Pendulum-v0',algorithm = 'SAC', episode = 6, called = 'state', label = state_L, dimension = 3, fig_width = 1,
    fig_height = 3,figsize = (14, 3)):

    env = gym.make(env_name)
    print(env.observation_space)
    print(env.action_space)

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

    for d in range(dimension):
        print(d)
        train_max_state, train_min_state, test_max_state, test_min_state = [[], [], [], []]
        for num in range(total_num):
            train_max_state.append(np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d])
            train_min_state.append(np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d])

            test_max_state.append(np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d])
            test_min_state.append(np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d])

            # test_state = np.load("{}{}/test_state_dict.npy".format(dir, num), allow_pickle=True).tolist()
            # test_action = np.load("{}{}/test_action_dict.npy".format(dir, num), allow_pickle=True).tolist()
            # test_state_action = np.load("{}{}/test_state_action.npy".format(dir, num), allow_pickle=True).tolist()
        avg_cov = (sum(test_max_state)/total_num - sum(test_min_state)/total_num) / (sum(train_max_state)/total_num - sum(train_min_state)/total_num)
        print(avg_cov)
        lists = [[train_max_state, train_min_state],[ test_max_state, test_min_state]]
        ls = [train_max_state, train_min_state, test_max_state, test_min_state]
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

        plt.text(x=4.5, y=(max(train_max_state)+min(train_min_state))/2, s="avg_cov={}".format(round(avg_cov, 3)), bbox={'facecolor': 'w','edgecolor':'gray', 'alpha': 0.8, 'boxstyle':'round'})

        ax.grid(color='gray', linestyle='--')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.set_xlabel("Episodes")
        # ax.set_ylabel("Rewards")
        #
    # plt.legend(loc='best', facecolor='blue')
        avg_cov_list.append(avg_cov)
    avg_coverage1 = sum(avg_cov_list)/ dimension
    avg_coverage2 = np.prod(avg_cov_list)
    print("cov1:", avg_coverage1)
    print("cov2:", avg_coverage2)
    plt.suptitle(env_name + "_" + algorithm + " (SBCov1 = {}, SBCov2 = {})".format(round(avg_coverage1, 3), round(avg_coverage2, 3)))
    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)

    plt.show()

if __name__ == '__main__':

    save_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
