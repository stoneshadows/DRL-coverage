import os
import glob
import time
from datetime import datetime


import numpy as np

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

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 1000
    # action_std = None


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


    total_test_episodes = 100     # total num of testing episodes

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
    num = 1
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
    tr_sa = [[] for i in range(len(grids))]
    sa = []
    j1 = [0 for i in range(len(grids))]
    j2 = [0 for i in range(len(grids))]
    j3 = [0 for i in range(len(grids))]
    we_s1 = [[] for i in range(len(grids))]
    we_s2 = [[] for i in range(len(grids))]
    we_s3 = [[] for i in range(len(grids))]

    for ep in range(1, total_test_episodes+1):  # 100个回合
        ep_reward = 0
        n = 0
        ss = [[] for i in range(len(grids))]
        s_a = [[] for i in range(len(grids))]
        state = env.reset()
        # state = choice(np.load("B_gen_s_all.npy"))

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # ##############################################################################################
            #
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

            # 离散化状态
            for i, state_grid in enumerate(state_grid_list):
                grid_state = discretize(state, state_grid)
                ss[i].append(grid_state)
                # sa.append((state, action))

                if '{}'.format(grid_state) in test_state_dict:
                    test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_dict[i]['{}'.format(grid_state)] = 1

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

            # 记录离散化的状态和动作
            for i in range(len(grids)):
                grid_state = discretize(state, state_grid_list[i])
                # grid_action = discretize(action, action_grid_list[i])
                grid_action = action
                s_a[i].append((grid_state, grid_action))  # 每个回合的状态动作对序列

                if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1

                # np.save("{}{}/test_state_action_{}.npy".format(dir,num,grids[i]), test_state_action_dict[i])
                #
                # count_dict['state-action'].append(len(test_state_action_dict[i]))
            # ###########################################################################################
            state = next_state
            n += 1
            # print(n)
            ep_reward += reward
            # 整个回合的奖励值

            # if render:
            #     env.render()
            #     time.sleep(frame_delay)
            if done:
                break
        # clear buffer

        ppo_agent.buffer.clear()
        # test_running_reward += ep_reward
        scores.append(ep_reward)  # 每个回合的总得分的列表
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
        for i, state_grid in enumerate(state_grid_list):
            tr_s[i].append(ss[i])  #

            tr_sa[i].append(s_a[i])  # 所有回合轨迹

            jac_sum1 = 0
            jac_sum2 = 0
            jac_sum3 = 0

            for ts in tr_s[i]:
                trace1 = list(zip(ss[i], ss[i][1:]))
                trace2 = list(zip(ts, ts[1:]))      # 每个回合组成的状态对

                jac_tr = jaccard(trace1, trace2)   # 状态轨迹（也就是状态对）的相似度比较
                jac_s = jaccard(ss[i], ts)          # 每个回合的状态的相似度比较

                jac_sum1 += jac_tr
                jac_sum2 += jac_s
            # print(jac_sum)  # 总的相似度
            # if jac_sum1 == 1:  # 与自己这个路径相似性为1，与之前的路径相似性均为0
            #     j1[i] += 1
            #     print("transition初始状态({}):".format(grids[i]), j1[i])
            #     we_s1[i].append(ep_reward)
            #
            if jac_sum2 == 1:
                j2[i] += 1
                print("state初始状态({}):".format(grids[i]), j2[i])
                we_s2[i].append(ep_reward)

            for tsa in tr_sa[i]:
                jac_sa = jaccard(s_a[i], tsa)
                jac_sum3 += jac_sa

            if jac_sum3 == 1:
                j3[i] += 1
                print("state-action初始状态({}):".format(grids[i]), j3[i])
                we_s3[i].append(ep_reward)
                # print(ss[i][0])  # 初始状态

            # np.save("{}{}/we_scores3_{}.npy".format(directory, num, grids[i]), we_s3[i])
            # np.save("{}{}/we_scores1_{}.npy".format(directory, num, grids[i]), we_s1[i])
            # np.save("{}{}/we_scores2_{}.npy".format(directory, num, grids[i]), we_s2[i])

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
    print(np.mean(scores))   # 最终平均得分：每个回合数的得分求和再取平均
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
