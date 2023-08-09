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


EPISODES = 1000
random_steps = 10000
max_episode_length = 2000
train_interval = 1
policy_delay = 1
load_model = False   # 加载模型测试阶段
random.seed(0)
# load scenario from script


env = gym.make('BipedalWalker-v3')

###################################################################
env_name = 'BipedalWalker-v3'
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

log_f_name = log_dir + '/AC_' + env_name + "_log_" + str(run_num) + ".csv"
log_f = open(log_f_name, "w+")
log_f.write('episode,reward\n')

directory = './preTrained/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

num = len(next(os.walk(directory))[1])

if not os.path.exists(directory + str(num) + '/'):
    os.makedirs(directory + str(num) + '/')

max_list = [0 for i in range(env.observation_space.shape[0])]
min_list = [0 for i in range(env.observation_space.shape[0])]
max_temp = [0 for i in range(env.observation_space.shape[0])]
min_temp = [0 for i in range(env.observation_space.shape[0])]
max_action_list = [0 for i in range(env.action_space.shape[0])]
min_action_list = [0 for i in range(env.action_space.shape[0])]
max_action_temp = [0 for i in range(env.action_space.shape[0])]
min_action_temp = [0 for i in range(env.action_space.shape[0])]
scores = []
#######################################################################################
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
ckpt_path = directory + str(num) + '/' +'trained_SAC_model'
Rewards = []
if load_model == True:
    saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))  # sy
    graph = tf.get_default_graph()
    saver.restore(sess, ckpt_path)  # 恢复模型，测试阶段
obs_ = None
t = 0
train_step = 0
avg = 0
avg_train_time = 0
avg_update_time = 0
noise = False
# log_f = open("log.txt","w+")
start_time = datetime.now().replace(microsecond=0)
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

        #######################################################################
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
        ##########
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
        ##########################################################################

        # time2 = time.time()
        # rand = random.uniform(0, 1)
        # if rand < noise:
        #     action = env.action_space.sample()
        obs_, reward, terminal, info = env.step(action)



        # print(time2 - time1)
        episode_reward += reward
        agent.store(obs, action, reward, obs_, terminal)



        if terminal == True:
            break
    # print(t - t_1)
    if len(agent.memory.obs) >= agent.batch_size:
        for train_i in range(t - t_1):
            if train_i % (policy_delay * train_interval) == 0:
                time3 = time.time()
                agent.train_batch(update_policy=True)
                time4 = time.time()
                agent.update_target_net()
                time5 = time.time()
                episode_train_time += time4 - time3
                episode_update_time += time5 - time4

                # print(time4 - time3)
            elif train_i % train_interval == 0:
                agent.train_batch(update_policy=False)
        train_step += 1
    # log_f.write('{},{}\n'.format(episode, episode_reward))
    # log_f.flush()
    avg += episode_reward / 10

    avg_train_time += episode_train_time / 10 / (t - t_1) * train_interval
    avg_update_time += episode_update_time / 10 / (t - t_1) * train_interval

    Rewards.append(episode_reward)
    if episode % 10 == 0:
        # noise *= 0.9

        print("Episode %d average reward: %.3f  average training time: %.4f  %.4f" % (episode, avg, avg_train_time, avg_update_time))
        avg = 0
        avg_train_time = 0
        avg_update_time = 0

    if episode % 100 == 0:
        checkpoint = 'check_point_episode_%d' % episode
        # saver.save(sess, checkpoint)

    ###############################################################################
    scores.append(episode_reward)
    np.save("{}{}/scores.npy".format(directory, num), scores)
    np.save("{}{}/max_action_list.npy".format(directory, num), max_action_list)
    np.save("{}{}/min_action_list.npy".format(directory, num), min_action_list)
    np.save("{}{}/max_state_list.npy".format(directory, num), max_list)
    np.save("{}{}/min_state_list.npy".format(directory, num), min_list)
    log_f.write('{},{}\n'.format(episode, episode_reward))
    log_f.flush()
# log_f.close()
####################################################################################
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
training_time = end_time - start_time
np.save("{}{}/Total_training_time".format(directory, num), training_time)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", training_time)
print("============================================================================================")

saver.save(sess, ckpt_path)
# Rewards = np.array(Rewards)
# divide = 200
# plt.plot(Rewards)
# plt.show()

# for i in range(1):
#     start = True
#     for _ in range(max_episode_length):
#         t += 1
#
#         if start == True:
#             obs = env.reset()
#             start = False
#         else:
#             obs = obs_
#         # time1 = time.time()
#         action = agent.step(obs).reshape(n_actions, )
#         print(action)
#         # time2 = time.time()
#         obs_, reward, terminal, info = env.step(action)
#         if terminal:
#             break
#         env.render()
#         # time.sleep(0.02)
#
#

