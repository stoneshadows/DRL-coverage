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
# fig_width = 4,fig_height = 6,figsize = (30,20)
#fig_width = 2,fig_height = 2,figsize = (10, 6)):

def save_graph(env_name = 'BipedalWalker-v3',algorithm = 'SAC', episode = 6, called = 'state', label = state_B, dimension = 24, fig_width = 4,
    fig_height = 6,figsize = (30, 20)):

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

        plt.text(x=7.5, y=(max(train_max_state)+min(train_min_state))/2, s="avg_cov={}".format(round(avg_cov, 3)), bbox={'facecolor': 'w','edgecolor':'gray', 'alpha': 0.8, 'boxstyle':'round'})

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
    plt.suptitle(env_name + "_" + algorithm + " (ABCov1 = {}, ABCov2 = {})".format(round(avg_coverage1, 3), round(avg_coverage2, 5)))
    # plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)

    plt.show()

if __name__ == '__main__':

    save_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
