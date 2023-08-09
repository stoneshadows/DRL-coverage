import os
import glob
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gym
# import roboschool

# import pybullet_envs

from PPO import PPO
import copy

def jaccard(p,q):
    c = [v for v in p if v in q]
    return float(len(c))/(len(p)+len(q)-len(c))

def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

#################################### Testing ###################################
from random import choice

def test():

    print("============================================================================================")

    ################## hyperparameters ##################



    # env_name = "BipedalWalker-v3"
    # has_continuous_action_space = True
    # max_ep_len = 1500
    # action_std = 0.1


    env_name = "LunarLander-v2"
    has_continuous_action_space = False
    max_ep_len = 1500           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving


    # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_acti  on_space = True
    # max_ep_len = 1000           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving


    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames


    total_test_episodes = 100    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    ##########################################################################################
    num = 0
    dir = './PPO_preTrained/' + env_name + '/'

    # max_action_list = np.load("{}{}/max_action_list.npy".format(dir, num))
    # min_action_list = np.load("{}{}/min_action_list.npy".format(dir, num))
    max_state_list = np.load("{}{}/max_state_list.npy".format(dir, num))
    min_state_list = np.load("{}{}/min_state_list.npy".format(dir, num))

    grids = [10, 50, 100, 500, 1000]
    # grids = [50]
    state_grid_list = []
    action_grid_list = []
    for g in grids:
        state_bins = [g for i in range(env.observation_space.shape[0])]
        # action_bins = [g for i in range(env.action_space.shape[0])]
        state_grid_list.append(create_uniform_grid(min_state_list, max_state_list, bins=state_bins))
        # action_grid_list.append(create_uniform_grid(min_action_list, max_action_list, bins=action_bins))

    test_state_dict = [{} for i in range(len(grids))]
    test_action_dict = [{} for i in range(len(grids))]
    test_state_action_dict = [{} for i in range(len(grids))]
    keys = ['state', 'action', 'state-action']
    count_dict = dict([(k, []) for k in keys])
    max_list = [0 for i in range(env.observation_space.shape[0])]
    min_list = [0 for i in range(env.observation_space.shape[0])]
    max_temp = [0 for i in range(env.observation_space.shape[0])]
    min_temp = [0 for i in range(env.observation_space.shape[0])]
    # max_action_list = [0 for i in range(env.action_space.shape[0])]
    # min_action_list = [0 for i in range(env.action_space.shape[0])]
    # max_action_temp = [0 for i in range(env.action_space.shape[0])]
    # min_action_temp = [0 for i in range(env.action_space.shape[0])]
    ###########################################################################################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num


    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, num)

    ppo_agent.load(checkpoint_path)

    test_running_reward = 0
    start_time = datetime.now().replace(microsecond=0)
    scores = []
    tr_s = [[] for i in range(len(grids))]
    j1 = [0 for i in range(len(grids))]
    j2 = [0 for i in range(len(grids))]
    we_s1 = [[] for i in range(len(grids))]
    we_s2 = [[] for i in range(len(grids))]

    init_state1 = [{} for i in range(len(grids))]
    init_state2 = [{} for i in range(len(grids))]
    init_state3 = [{} for i in range(len(grids))]

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        n = 0
        ss = [[] for i in range(len(grids))]  # 每个回合状态序列
        scor = [[] for i in range(len(grids))]
        state = env.reset()

        # state = choice(np.load("B_gen_s_all.npy"))

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # ###########
            # for j, val in enumerate(state):
            #     if val >= max_temp[j]:
            #         max_list[j] = val
            #     elif val < min_temp[j]:
            #         min_list[j] = val
            #     else:
            #         max_list[j] = max_temp[j]
            #         min_list[j] = min_temp[j]
            # min_temp = copy.deepcopy(min_list)
            # ############

            for i, state_grid in enumerate(state_grid_list):
                grid_state = discretize(state, state_grid)
                ss[i].append(grid_state)   # 对于一个状态有5种粒度，ss = [ss[10], ss[50], ss[100], ss[500], ss[1000]]

                # sa.append((state, action))
                if '{}'.format(grid_state) in test_state_dict[i]:
                    test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数

                else:
                    test_state_dict[i]['{}'.format(grid_state)] = 1
                # print(len(test_state_dict[i]))
                # 每次回合如何观测字典长度变化？长度变化大说明覆盖变化大，测试用例更有效，所以要记录每个回合的初始状态及其对应的长度变化值和得分


                # np.save("{}{}/test_state_dict_{}.npy".format(dir, num, grids[i]), test_state_dict[i])
                # count_dict['state'].append(len(test_state_dict[i]))
            #
            # # for i, action_grid in enumerate(action_grid_list):
            # #     grid_action = discretize(action, action_grid)
            # #     if '{}'.format(grid_action) in test_action_dict:
            # #         test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
            # #     else:
            # #         test_action_dict[i]['{}'.format(grid_action)] = 1
            # #     np.save("{}{}/test_action_dict_{}.npy".format(dir,num,grids[i]), test_action_dict[i])
            # #     count_dict['action'].append(len(test_action_dict[i]))
            #
            # for i in range(len(grids)):
            #     grid_state = discretize(state, state_grid_list[i])
            #     # grid_action = discretize(action, action_grid_list[i])
            #     grid_action = action
            #     if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
            #         test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
            #     else:
            #         test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
            #     np.save("{}{}/test_state_action_{}.npy".format(dir,num,grids[i]), test_state_action_dict[i])
            #
            #     count_dict['state-action'].append(len(test_state_action_dict[i]))
            # ###########################################################################################
            state = next_state
            n += 1
            # print(n)
            ep_reward += reward
            # if render:
            #     env.render()
            #     time.sleep(frame_delay)
            if done:
                break
        # clear buffer
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        scores.append(round(ep_reward, 2))
        print(0, scores)
        # print("rewards:", ep_reward)
        # np.save("{}{}/test_scores.npy".format(directory, num), scores)
        # #######################################################################################

        # # np.save("{}{}/test_min_action.npy".format(dir,num), min_action_list)
        # np.save("{}{}/test_max_state.npy".format(dir,num), max_list)
        # np.save("{}{}/test_min_state.npy".format(dir,num), min_list)
        # ####################################################################################

        # 伪代码
        # jac = jaccard(ss[1], ss[2])
        # if jac > 0:
        #     # 重新开始一个回合，选择一个初始状态，计算和之前所有状态的jaccard距离，直到jac值为0
        # else: # jac == 0
        #     # 得到测试用例，也就是初始状态（状态路径覆盖增大），记录这些初始状态的得分和随机选取的作比较，测试策略

        # if ep_reward < 50:
        #     ee += 1
        # print("reward<50:", ee)

#######################################################################################
        for i in range(len(grids)):
            p = 0
            for s in ss[i]:
                if s not in tr_s[i]:  # 判断回合中的每个状态是否在总的状态池中
                    p += 1
            # p为回合中新增加的状态数，p值和初始状态组成字典进行排序
            tr_s[i].extend(ss[i])      # 每个回合 的状态序列
            # init_state[i]['{}'.format(ss[i][0])] = p
            init_state1[i][ep] = p   # 运行50个回合，挑出覆盖最多的30个序号
            # init_state2[i][ep] = ss[i][0]
            # init_state3[i][ep] = ep_reward   # 选这50个序号的得分

            sort = sorted(init_state1[i].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 按照p的值给初始状态排序
            print(sort)  # [(1, 500), (2, 176), (12, 98), (3, 91), (4, 65), (9, 36), (10, 28), (8, 19), (5, 11), (11, 10), (14, 9), (13, 8), (7, 6), (6, 3)]
            for j in sort:   # 如果得分相同，序号如何排序
                scor[i].append(round(scores[j[0] - 1], 2))  # 按序号从大到小提取得分，也就是按照覆盖准则从大到小来排序得分，解决每个粒度问题，每个粒度对应新的得分序列,累计平均得分
                                                            # 新增状态个数是在上一个回合的基础上新增，所以不适合排序，状态个数覆盖与得分的关系
            print(grids[i], scor[i])

            # rolling_mean = pd.Series(scor[i]).rolling(50, min_periods=1, center=True).mean()
            # plt.plot(scor[i])
            # plt.show()



            # 以ss[i][0]为初始状态的状态序列，有多少没有出现过，即覆盖了多少新的状态。提取初始状态, 记录对应的奖励值，也就是每个回合的rewards
            # 按照覆盖状态个数的多少，给状态排序，与对应回合奖励值的关系。
            # print(init_state1[i])
            # print(init_state2[i])
            # print(init_state3[i])
            #
            # jac_sum1 = 0
            # jac_sum2 = 0
            # for ts in tr_s[i]:
            #     trace1 = list(zip(ss[i], ss[i][1:]))
            #     trace2 = list(zip(ts, ts[1:]))
            #
            #
            #     jac_tr = jaccard(trace1, trace2)   # 状态轨迹（也就是状态对）的相似度比较
            #     jac_s = jaccard(ss[i], ts)  # 每个回合的状态的相似度比较
            #
            #     jac_sum1 += jac_tr
            #     jac_sum2 += jac_s
            # print(jac_sum1)  # 总的相似度
            # if jac_sum1 == 1:  # 与自己这个路径相似性为1，与之前的路径相似性均为0
            #     j1[i] += 1
            #     print("transition初始状态({}):".format(grids[i]), j1[i])
            #     we_s1[i].append(ep_reward)
            #
            # if jac_sum2 == 1:
            #     j2[i] += 1
            #     print("state初始状态({}):".format(grids[i]), j2[i])
            #     we_s2[i].append(ep_reward)


#############################################################################################
                    # print(round (ep_reward, 2))
                # else:
                #     print(ss[i][0])  # 初始状态


            # print(len(ss[i]))
            # print(len(tr_s[i]))
            # np.save("{}{}/trace_state_{}.npy".format(dir, num, grids[i]), tr_s[i])
        # print(tr_s)

        print('Episode, States: {},{} \t\t Reward: {}'.format(ep, n, round(ep_reward, 2)))
        ep_reward = 0
    env.close()
    print(np.mean(scores))
    # print("============================================================================================")
    # avg_test_reward = test_running_reward / total_test_episodes
    # avg_test_reward = round(avg_test_reward, 2)
    # print("average test reward : " + str(avg_test_reward))
    #
    # print("============================================================================================")
    # end_time = datetime.now().replace(microsecond=0)
    # total_time = end_time - start_time
    # np.save("{}{}/test_time".format(dir, num), total_time)
    # print("Started training at (GMT) : ", start_time)
    # print("Finished training at (GMT) : ", end_time)
    # print("Total training time  : ", total_time)
    # print("============================================================================================")


if __name__ == '__main__':

    test()
