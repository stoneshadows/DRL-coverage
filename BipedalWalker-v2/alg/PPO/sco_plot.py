import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gym
import pickle
import os
import seaborn as sns

env_name = "LunarLander-v2"
total_num = 6
directory = "./PPO_preTrained" + '/' + env_name + '/'

for num in range(total_num):
    scores = np.load("{}{}/test_scores.npy".format(directory, num))
    grids = [10, 50, 100, 500, 1000]
    for g in grids:
        print("grids:", g)
        we_scores1 = np.load("{}{}/we_scores1_{}.npy".format(directory, num, g), allow_pickle=True).tolist()
        we_scores2 = np.load("{}{}/we_scores2_{}.npy".format(directory, num, g), allow_pickle=True).tolist()
        we_scores3 = np.load("{}{}/we_scores3_{}.npy".format(directory, num, g), allow_pickle=True).tolist()
        rad_scores = random.sample(list(scores), len(we_scores1))
        select_rewards = [np.mean(we_scores1), np.mean(we_scores2), np.mean(we_scores3), np.mean(rad_scores), np.mean(scores), len(we_scores1),len(we_scores2),len(we_scores3)]
        np.save("{}{}/select_rewards_{}.npy".format(directory, num, g), select_rewards)

        print("select_tr:", np.mean(we_scores1))
        print("select_s:", np.mean(we_scores2))
        print("select_sa:", np.mean(we_scores3))
        print("total:", np.mean(scores))
        print("random:", np.mean(rad_scores))
        print(len(we_scores1))
        print(len(we_scores2))
        print(len(we_scores3))
        print(len(scores))

# 一个算法多个回合求平均得到的最终结果，不同的粒度g

for g in grids:
    all_rewards = []
    for num in range(total_num):
        all_rewards.append(np.load("{}{}/select_rewards_{}.npy".format(directory, num, g)))
    # print(all_rewards)

    final_rewards = sum(all_rewards) / total_num
    print("grids:", g, final_rewards)
# rolling_mean = pd.Series(scores).rolling(50, min_periods=1, center=True).mean()
# rad_mean = pd.Series(rad_scores).rolling(50, min_periods=1, center=True).mean()
# r_mean1 = pd.Series(we_scores1).rolling(50, min_periods=1, center=True).mean()
# r_mean2 = pd.Series(we_scores2).rolling(50, min_periods=1, center=True).mean()
#
# # plt.plot(scores)
# # plt.plot(rad_scores)
# # plt.plot(we_scores1)
# # plt.plot(we_scores2)
# #############
# plt.plot(rolling_mean)
# plt.plot(rad_mean)
# plt.plot(r_mean1)
# plt.plot(r_mean2)
#
# plt.show()
# ENV_NAME = 'CartPole-v0'
#
# file = "behavior_" + ENV_NAME+".pkl"
# with open(os.path.join(file), "rb") as f:
#     data = pickle.load(f)
#
# x1 = data["mean"]
# x2 = data["std"]
# # print(x1)
# # x1=np.load('../D3QN/200/scores.npy')
# # x2=np.load('../D3QN/200-1/scores.npy')
# x = [x1, x2]
# print(x2)
# time = range(len(x1))
#
# sns.set(style="darkgrid", font_scale=1.5)
# sns.tsplot(time=time, data=x, color="r", condition="behavior_cloning")
#
# plt.ylabel("Reward")
# plt.xlabel("Iteration Number")
# plt.title("Imitation Learning")
#
# plt.show()
#
# # max_list = np.load('./max_list.npy')
# # print(max_list)
# # min_list = np.load('./min_list.npy')
# # print(min_list)
# # state_dict = np.load('./state_dict.npy', allow_pickle=True).tolist()
# # print(len(state_dict))
#
# # max_list = np.load('./300/max_list.npy')
# # print(max_list)
# # min_list = np.load('./300/min_list.npy')
# # print(min_list)
# # state_dict = np.load('./300/state_dict.npy', allow_pickle=True).tolist()
# # print(len(state_dict))
#
# max_list = np.load('./200/max_list.npy')
# print(max_list)
# min_list = np.load('./200/min_list.npy')
# print(min_list)
# state_action_dict = np.load('./200/state_action_dict.npy', allow_pickle=True).tolist()
# print("action:", len(state_action_dict))
# state_dict = np.load('./200/state_dict.npy', allow_pickle=True).tolist()
# print(len(state_dict))

test_state_dict = np.load('./PPO_preTrained/LunarLander-v2/0/test_state_dict_1000.npy', allow_pickle=True).tolist()
print("test_state:", len(test_state_dict))

rank_index = []
rank_value = []
for v in sorted(test_state_dict, key=test_state_dict.__getitem__, reverse=True):
    rank_index.append(v)
    rank_value.append(test_state_dict[v])

rank = rank_index[0:len(test_state_dict)]
rank_v = rank_value[0:len(test_state_dict)]

plt.bar(range(len(test_state_dict)),rank_v)
plt.show()
# #
# count = 0
# for t in test_state_dict.keys():
#     if t in rank:
#         count += 1
# cov = count/len(test_state_dict)
# print(count)
# print("cov:", cov)
#
# test_s_dict = np.load('./test_s_dict.npy', allow_pickle=True).tolist()
# print(len(test_s_dict))
# test_max_list = np.load('./test_max_list.npy')
# print(test_max_list)
# test_min_list = np.load('./test_min_list.npy')
# print(test_min_list)
#
# # test_max_list = np.load('./test_max_list.npy')
# # print(test_max_list)
# # test_min_list = np.load('./test_min_list.npy')
# # print(test_min_list)
# # test_state_dict = np.load('./test_state_dict.npy', allow_pickle=True).tolist()
# # print(len(test_state_dict))
#
# env_name = "BipedalWalker-v2"
# scores = np.load("./preTrained/{}/one/scores.npy".format(env_name))


# plt.xlabel('Episode')
# plt.ylabel('Average Reward')
# plot_scores(scores,label="reward")
# plt.show()


# def getdata():
#     basecond = [[18, 20, 19, 18, 13, 4, 1],
#                 [20, 17, 12, 9, 3, 0, 0],
#                 [20, 20, 20, 12, 5, 3, 0]]
#
#     cond1 = [[18, 19, 18, 19, 20, 15, 14],
#              [19, 20, 18, 16, 20, 15, 9],
#              [19, 20, 20, 20, 17, 10, 0],
#              [20, 20, 20, 20, 7, 9, 1]]
#
#     cond2 = [[20, 20, 20, 20, 19, 17, 4],
#              [20, 20, 20, 20, 20, 19, 7],
#              [19, 20, 20, 19, 19, 15, 2]]
#
#     cond3 = [[20, 20, 20, 20, 19, 17, 12],
#              [18, 20, 19, 18, 13, 4, 1],
#              [20, 19, 18, 17, 13, 2, 0],
#              [19, 18, 20, 20, 15, 6, 0]]
#
#     return basecond, cond1, cond2, cond3
#
# data = getdata()
# fig = plt.figure()
# xdata = np.array([0, 1, 2, 3, 4, 5, 6])/5
# linestyle = ['-', '--', ':', '-.']
# color = ['r', 'g', 'b', 'k']
# label = ['algo1', 'algo2', 'algo3', 'algo4']
#
# for i in range(4):
#     sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])
#
# plt.ylabel("Success Rate", fontsize=25)
# plt.xlabel("Iteration Number", fontsize=25)
# plt.title("Awesome Robot Performance", fontsize=30)
# plt.show()