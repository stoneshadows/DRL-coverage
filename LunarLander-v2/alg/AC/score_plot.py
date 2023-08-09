import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
import pickle
import os
import seaborn as sns

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
#
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
#
# test_state_dict = np.load('./test_state_dict.npy', allow_pickle=True).tolist()
# print("test_state:", len(test_state_dict))
#
# rank_index = []
# rank_value = []
# for v in sorted(state_dict, key=state_dict.__getitem__, reverse=True):
#     rank_index.append(v)
#     rank_value.append(state_dict[v])
#
# rank = rank_index[0:len(test_state_dict)]
# rank_v = rank_value[0:len(test_state_dict)]
#
# plt.bar(range(len(test_state_dict)),rank_v)
# plt.show()
#
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

# test_max_list = np.load('./test_max_list.npy')
# print(test_max_list)
# test_min_list = np.load('./test_min_list.npy')
# print(test_min_list)
# test_state_dict = np.load('./test_state_dict.npy', allow_pickle=True).tolist()
# print(len(test_state_dict))


scores = np.load('./preTrained/LunarLander-v2/2/scores.npy')
print(scores)
rolling_mean = pd.Series(scores).rolling(100, min_periods=1, center=True).mean()
plt.plot(scores)
plt.plot(rolling_mean)

plt.show()
# print(np.mean(scores))
# p_scores(scores)
#

# plt.xlabel('Episode')
# plt.ylabel('Average Reward')
# plot_scores(scores,label="reward")
# plt.show()