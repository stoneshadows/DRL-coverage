# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from SAC import SAC
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym
import os
import copy
from datetime import datetime
def prob2action(action_array):
    max_i = 0
    max_value = action_array[0]
    for i in range(1, len(action_array)):
        if action_array[i] > max_value:
            max_i = i
            max_value = action_array[i]
    return max_i
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

EPISODES = 100
random_steps = 1000
max_episode_length = 200
train_interval = 1
policy_delay = 1
random.seed(0)
# load scenario from script

num = 5
env = gym.make('BipedalWalker-v3')
env_name = 'BipedalWalker-v3'
directory = './preTrained/' + env_name + '/'

# result_path = directory + str(num) + '/results/'
# model_path = directory + str(num) + '/models/'
eval_env = gym.make(env_name)
eval_env.seed(100)
rewards, ma_rewards = [], []
##########################################################################################

max_action_list = np.load("{}{}/max_action_list.npy".format(directory, num))
min_action_list = np.load("{}{}/min_action_list.npy".format(directory, num))
max_state_list = np.load("{}{}/max_state_list.npy".format(directory, num))
min_state_list = np.load("{}{}/min_state_list.npy".format(directory, num))

grids = [10, 50, 100, 500, 1000]
state_grid_list = []
action_grid_list = []
for g in grids:
    state_bins = [g for i in range(env.observation_space.shape[0])]
    action_bins = [g for i in range(env.action_space.shape[0])]
    state_grid_list.append(create_uniform_grid(min_state_list, max_state_list, bins=state_bins))
    action_grid_list.append(create_uniform_grid(min_action_list, max_action_list, bins=action_bins))

test_state_dict = [{} for i in range(len(grids))]
test_action_dict = [{} for i in range(len(grids))]
test_state_action_dict = [{} for i in range(len(grids))]
keys = ['state', 'action', 'state-action']
count_dict = dict([(k, []) for k in keys])
max_list = [0 for i in range(env.observation_space.shape[0])]
min_list = [0 for i in range(env.observation_space.shape[0])]
max_temp = [0 for i in range(env.observation_space.shape[0])]
min_temp = [0 for i in range(env.observation_space.shape[0])]
max_action_list = [0 for i in range(env.action_space.shape[0])]
min_action_list = [0 for i in range(env.action_space.shape[0])]
max_action_temp = [0 for i in range(env.action_space.shape[0])]
min_action_temp = [0 for i in range(env.action_space.shape[0])]
###########################################################################################
start_time = datetime.now().replace(microsecond=0)
obs_shape = env.observation_space.shape
n_actions = 4
action_shape = (n_actions, )
agent = SAC(observation_shape=obs_shape, action_shape=action_shape, clip_value=None, alpha=0.2,
            gamma=0.99, tau=0.005, batch_size=256, memory_max=int(1000000), actor_lr=1e-4, critic_lr=1e-4)

sess = tf.Session()
agent.initialize(sess)
writer = tf.summary.FileWriter("logs/", sess.graph)
writer.add_graph(sess.graph)
saver = tf.train.Saver()

ckpt_path = directory + str(num) + '/' + 'trained_SAC_model'

# saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))  # sy
model_file=tf.train.latest_checkpoint(directory + str(num) + '/')
# graph = tf.get_default_graph()
saver.restore(sess, model_file)  # 恢复模型，测试阶段

Rewards = []

obs_ = None
t = 0
train_step = 0
avg = 0
avg_train_time = 0
avg_update_time = 0
noise = False
log_f = open("log.txt","w+")
for episode in range(1, EPISODES + 1):
    episode_reward = 0
    episode_train_time = 0
    episode_update_time = 0
    start = True
    t_1 = t
    for _ in range(max_episode_length):
        t += 1

        if start == True:
            obs = env.reset()
            start = False
        else:
            obs = obs_
        # time1 = time.time()
        if t >= random_steps:
            action = agent.step(obs).reshape(n_actions, )
        else:
            action = env.action_space.sample()

        ##############################################################################################
        for k, va in enumerate(action):
            if va >= max_action_temp[k]:
                max_action_list[k] = va
            elif va < min_action_temp[k]:
                min_action_list[k] = va
            else:
                max_action_list[k] = max_action_temp[k]
                min_action_list[k] = min_action_temp[k]
        max_action_temp = copy.deepcopy(max_action_list)
        min_action_temp = copy.deepcopy(min_action_list)
        ###########
        for j, val in enumerate(obs):
            if val >= max_temp[j]:
                max_list[j] = val
            elif val < min_temp[j]:
                min_list[j] = val
            else:
                max_list[j] = max_temp[j]
                min_list[j] = min_temp[j]
        max_temp = copy.deepcopy(max_list)
        min_temp = copy.deepcopy(min_list)
        ############
        for i, state_grid in enumerate(state_grid_list):
            grid_state = discretize(obs, state_grid)
            if '{}'.format(grid_state) in test_state_dict[i]:
                test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
            else:
                test_state_dict[i]['{}'.format(grid_state)] = 1
            np.save("{}{}/test_state_dict_{}.npy".format(directory, num, grids[i]), test_state_dict[i])
            count_dict['state'].append(len(test_state_dict[i]))

        for i, action_grid in enumerate(action_grid_list):
            grid_action = discretize(action, action_grid)
            if '{}'.format(grid_action) in test_action_dict[i]:
                test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
            else:
                test_action_dict[i]['{}'.format(grid_action)] = 1
            np.save("{}{}/test_action_dict_{}.npy".format(directory, num, grids[i]), test_action_dict[i])
            count_dict['action'].append(len(test_action_dict[i]))

        for i in range(len(grids)):
            grid_state = discretize(obs, state_grid_list[i])
            grid_action = discretize(action, action_grid_list[i])
            if '{},{}'.format(grid_state, grid_action) in test_state_action_dict[i]:
                test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
            else:
                test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
            np.save("{}{}/test_state_action_{}.npy".format(directory, num, grids[i]), test_state_action_dict[i])

            count_dict['state-action'].append(len(test_state_action_dict[i]))
        ###########################################################################################

        obs_, reward, terminal, info = env.step(action)

        episode_reward += reward
        agent.store(obs, action, reward, obs_, terminal)

        if terminal == True:
            break
    # print(t - t_1)
    if len(agent.memory.obs) >= agent.batch_size:
        # for train_i in range(t - t_1):
        #     if train_i % (policy_delay * train_interval) == 0:
        #         time3 = time.time()
        #         agent.train_batch(update_policy=True)
        #         time4 = time.time()
        #         agent.update_target_net()
        #         time5 = time.time()
        #         episode_train_time += time4 - time3
        #         episode_update_time += time5 - time4
        #
        #         # print(time4 - time3)
        #     elif train_i % train_interval == 0:
        #         agent.train_batch(update_policy=False)
        train_step += 1
    # log_f.write('{},{}\n'.format(episode, episode_reward))
    # log_f.flush()
    avg += episode_reward / 10

    avg_train_time += episode_train_time / 10 / (t - t_1) * train_interval
    avg_update_time += episode_update_time / 10 / (t - t_1) * train_interval

    Rewards.append(episode_reward)
    if episode % 10 == 0:
        # noise *= 0.9

        print("Episode %d average reward: %.3f " % (episode, avg))
        avg = 0
        avg_train_time = 0
        avg_update_time = 0

    if episode % 100 == 0:
        checkpoint = 'check_point_episode_%d' % episode
        # saver.save(sess, checkpoint)

    #######################################################################################
    np.save("{}{}/test_max_action.npy".format(directory, num), max_action_list)
    np.save("{}{}/test_min_action.npy".format(directory, num), min_action_list)
    np.save("{}{}/test_max_state.npy".format(directory, num), max_list)
    np.save("{}{}/test_min_state.npy".format(directory, num), min_list)


print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
total_time = end_time - start_time
np.save("{}{}/test_time".format(directory, num), total_time)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", total_time)
print("============================================================================================")



