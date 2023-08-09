import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def save_graph(env_name = 'BipedalWalker-v2', episode = 3, called = 'state', dimension = 24, fig_width = 4,
    fig_height = 6,figsize = (20, 10)):


    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
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
        train_max_state, train_min_state, test_max_state, test_min_state = [[] for i in range(total_num + 1)]
        for num in range(total_num):
            train_max_state.append(np.load("{}{}/max_{}_list.npy".format(dir, num, called))[d])
            train_min_state.append(np.load("{}{}/min_{}_list.npy".format(dir, num, called))[d])

            test_max_state.append(np.load("{}{}/test_max_{}.npy".format(dir, num, called))[d])
            test_min_state.append(np.load("{}{}/test_min_{}.npy".format(dir, num, called))[d])

            test_state = np.load("{}{}/test_state_dict.npy".format(dir, num), allow_pickle=True).tolist()
            test_action = np.load("{}{}/test_action_dict.npy".format(dir, num), allow_pickle=True).tolist()
            test_state_action = np.load("{}{}/test_state_action_dict.npy".format(dir, num), allow_pickle=True).tolist()

        lists = [[train_max_state, train_min_state], [test_max_state, test_min_state]]
        ls = [train_max_state, train_min_state, test_max_state, test_min_state]
        # labels = ['train_max_state', 'train_min_state', 'test_max_state', 'test_min_state']
        colors = ['g', 'r']
        cs = ['g', 'g','r','r']
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
        # plt.ylabel('Y-axis')

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

        # lists = [train_max_state, train_min_state, test_max_state, test_min_state]
        # labels = ['train_max_state', 'train_min_state', 'test_max_state', 'test_min_state']
        # all_runs = []
        # for list in lists:
        #     dict = {}
        #     dict['episode'] = [i for i in range(1,total_num+1)]
        #     dict['reward'] = list
        #     data = pd.DataFrame(dict)
        #     all_runs.append(data)
        #     # print(all_runs)
        #
        # colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']
        #
        # # make directory for saving figures
        # figures_dir = "PPO_figs"
        # if not os.path.exists(figures_dir):
        #     os.makedirs(figures_dir)
        #
        # # make environment directory for saving figures
        # figures_dir = figures_dir + '/' + env_name + '/'
        # if not os.path.exists(figures_dir):
        #     os.makedirs(figures_dir)
        #
        # fig_num = len(next(os.walk(figures_dir))[2])
        #
        # fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + str(fig_num) + '.png'
        #
        # ax = plt.gca()
        #
        # if plot_avg:
        #     # average all runs
        #     df_concat = pd.concat(all_runs)
        #     df_concat_groupby = df_concat.groupby(df_concat.index)
        #     data_avg = df_concat_groupby.mean()
        #
        #     # smooth out rewards to get a smooth and a less smooth (var) plot lines
        #     data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        #     data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        #
        #     data_avg.plot(kind='line', x='episode' , y='reward_smooth',ax=ax,color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        #     data_avg.plot(kind='line', x='episode' , y='reward_var',ax=ax,color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)
        #
        #     # keep only reward_smooth in the legend and rename it
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)
        #
        #
        # else:
        #     for i, run in enumerate(all_runs):
        #         # smooth out rewards to get a smooth and a less smooth (var) plot lines
        #         run[labels[i]] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        #         run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        #
        #         # plot the lines
        #         run.plot(kind='line', x='episode', y=labels[i],ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        #         run.plot(kind='line', x='episode', y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)
        #
        #     # keep alternate elements (reward_smooth_i) in the legend
        #     handles, labels = ax.get_legend_handles_labels()
        #     new_handles = []
        #     new_labels = []
        #     for i in range(len(handles)):
        #         if(i%2 == 0):
        #             new_handles.append(handles[i])
        #             new_labels.append(labels[i])
        #     ax.legend(new_handles, new_labels, loc=2)
        #
        # ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2)
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.set_xlabel("Episodes")
        # ax.set_ylabel("Rewards")
        #
        # plt.title(env_name)
        #
        # fig = plt.gcf()
        # fig.set_size_inches(fig_width, fig_height)
        # plt.savefig(fig_save_path)
        # print("figure saved at : ", fig_save_path)
        #
        # plt.show()


if __name__ == '__main__':

    save_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
