import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def save_graph():

    env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
    # env_name = 'BipedalWalker-v2'


    fig_num = 0     #### change this to prevent overwriting figures in same env_name folder

    plot_avg = True   # plot average of all runs; else plot all runs separately

    fig_width = 10
    fig_height = 6


    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']


    # make directory for saving figures
    figures_dir = "figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + env_name + 'reward_fig.png'

    # get number of log files in directory

    log_dir = "logs" + '/' + env_name + '/'

    current_num_files = next(os.walk(log_dir))
    print(current_num_files)

    all_runs = []
    labels = []
    for log_f_name in current_num_files:
        # label = log_f_name.split("_")[3]
        data = pd.read_csv("./{}/{}".format(log_dir,log_f_name))
        data = pd.DataFrame(data)
        all_runs.append(data)
        # labels.append(label)


    ax = plt.gca()

    if plot_avg:
        # average all runs
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='episode' , y='reward_smooth',ax=ax,color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='episode' , y='reward_var',ax=ax,color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)

        # keep only reward_smooth in the legend and rename it
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)


    else:
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run[labels[i]] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='episode', y=labels[i],ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='episode', y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)

        # keep alternate elements (reward_smooth_i) in the legend
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i%2 == 0):
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2)

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    plt.title(env_name, fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    # plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)

    plt.show()
##################################################################################################################

def avg_all_graph():
    # env_name = 'CartPole-v1'
    env_name = 'LunarLander-v2'
    # env_name = 'BipedalWalker-v2'
    # env_name = 'BipedalWalker-v2'

    fig_num = 0  #### change this to prevent overwriting figures in same env_name folder

    plot_avg = True  # plot average of all runs; else plot all runs separately

    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = sns.color_palette("deep")
    # colors = ['aliceblue', 'skyblue', 'deepskyblue', 'orange', 'tomato', 'red']

    # make directory for saving figures
    figures_dir = "figs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + 'reward_fig.png'

    # get number of log files in directory

    l_dir = "logs" + '/'
    num_files = next(os.walk(l_dir))[1]
    keys = ["(1) avg. episode reward over all of training", "(2) avg. episode reward over last 100 episodes"]
    count_dict = dict([(k, []) for k in keys])
    for i,file in enumerate(num_files):
        log_dir = "logs" + '/'+ file + '/'
        current_num_files = next(os.walk(log_dir))[2]
        label = file



        all_runs = []
        for log_f_name in current_num_files:
            data = pd.read_csv("./{}/{}".format(log_dir, log_f_name))
            data = pd.DataFrame(data)
            all_runs.append(data)

        sns.set_style(style="darkgrid")
        ax = plt.gca()
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        count_dict[keys[0]].append(np.mean(data_avg['reward']))
        count_dict[keys[1]].append(np.mean(data_avg['reward'][-100:]))

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang',
                                                               min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang',
                                                            min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='episode', y='reward_smooth', ax=ax, color=colors[i], linewidth=linewidth_smooth,
                      alpha=alpha_smooth, label=label)
        data_avg.plot(kind='line', x='episode', y='reward_var', ax=ax, color=colors[i], linewidth=linewidth_var,
                      alpha=alpha_var, label=label)

    # keep only reward_smooth in the legend and rename it
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i%2 == 0):
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=1)
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc=1)

    df = pd.DataFrame(count_dict, index=new_labels).T
    print(df)
    df.to_csv(figures_dir + 'avg_rewards' + '.csv')

    # ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")


    plt.title(env_name)
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)

    plt.show()


if __name__ == '__main__':
    avg_all_graph()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
