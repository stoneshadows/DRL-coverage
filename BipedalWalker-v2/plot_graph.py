import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def avg_all_graph():
    # env_name = 'CartPole-v1'
    # env_name = 'Pendulum-v0'
    env_name = 'BipedalWalker-v3'
    # env_name = 'BipedalWalker-v2'

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

    # make directory for saving figures
    figures_dir = "figs/"
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
        label = file
        current_num_files = next(os.walk(log_dir))[2]

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
    # df.to_csv(figures_dir + 'avg_rewards' + '.csv')

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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
