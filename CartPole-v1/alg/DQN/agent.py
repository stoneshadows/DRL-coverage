import copy
import os
import pickle
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer

import os
import gym
import copy
from datetime import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # 返回索引值


def jaccard(p,q):
    c = [v for v in p if v in q]
    return float(len(c))/(len(p)+len(q)-len(c))

env_name = 'CartPole-v1'
env = gym.make(env_name)
print(env.action_space)

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

directory = 'preTrained'
if not os.path.exists(log_dir):
    os.makedirs(directory)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

directory = directory+ '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)

# state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(100, 100))
################################################

def DeepQNetwork(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return q_net


class Agent:
    def __init__(self, lr, discount_factor, num_actions, epsilon, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.update_rate = 100
        self.step_counter = 0
        self.buffer = ReplayBuffer(1000000, input_dims)
        self.q_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)
        self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)

    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def train(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_rate == 0:
            self.q_target_net.set_weights(self.q_net.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted = self.q_net(state_batch)
        q_next = self.q_target_net(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.discount_factor*q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val
        self.q_net.train_on_batch(state_batch, q_target)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.step_counter += 1
        return state_batch ###########################################################

    def train_model(self, env, num_episodes, graph):

        num = len(next(os.walk(directory))[1])
        if not os.path.exists(directory + str(num) + '/'):
            os.makedirs(directory + str(num) + '/')

        run_num = len(next(os.walk(log_dir))[2])
        log_f_name = log_dir + '/DQN_' + env_name + "_log_" + str(run_num) + ".csv"

        log_f = open(log_f_name, "w+")
        log_f.write('episode,reward\n')

        # grids = [10, 50, 100, 500, 1000]
        # state_grid_list = []
        # action_grid_list = []
        # for g in grids:
        #     state_bins = [g for i in range(env.observation_space.shape[0])]
        #     # action_bins = [g for i in range(env.action_space.shape[0])]
        #     state_grid_list.append(
        #         create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=state_bins))

        # test_state_dict = [{} for i in range(len(grids))]
        # test_action_dict = [{} for i in range(len(grids))]
        # test_state_action_dict = [{} for i in range(len(grids))]

        max_list = [0 for i in range(env.observation_space.shape[0])]
        min_list = [0 for i in range(env.observation_space.shape[0])]
        max_temp = [0 for i in range(env.observation_space.shape[0])]
        min_temp = [0 for i in range(env.observation_space.shape[0])]
        # max_action_list = [0 for i in range(env.action_space.shape[0])]
        # min_action_list = [0 for i in range(env.action_space.shape[0])]
        # max_action_temp = [0 for i in range(env.action_space.shape[0])]
        # min_action_temp = [0 for i in range(env.action_space.shape[0])]

        scores, episodes, avg_scores, obj, std = [], [], [], [], []
        goal = 195
        # f = 0
        # txt = open("saved_networks.txt", "w")
        start_time = datetime.now().replace(microsecond=0)

        for e in range(num_episodes):
            done = False
            score = 0.0
            state = env.reset()
            while not done:
                action = self.policy(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, new_state, done)
                state = new_state
                self.train()
                if type(self.train()) is np.ndarray:
                    for st in self.train():
                        for j, val in enumerate(st):
                            if val >= max_temp[j]:
                                max_list[j] = val
                            elif val < min_temp[j]:
                                min_list[j] = val
                            else:
                                max_list[j] = max_temp[j]
                                min_list[j] = min_temp[j]
                        max_temp = copy.deepcopy(max_list)
                        min_temp = copy.deepcopy(min_list)

                        ################
                        # for k, va in enumerate(action):
                        #     if va >= max_action_temp[k]:
                        #         max_action_list[k] = va
                        #     elif va < min_action_temp[k]:
                        #         min_action_list[k] = va
                        #     else:
                        #         max_action_list[k] = max_action_temp[k]
                        #         min_action_list[k] = min_action_temp[k]
                        # max_action_temp = copy.deepcopy(max_action_list)
                        # min_action_temp = copy.deepcopy(min_action_list)

                        # ############
                        # for i, state_grid in enumerate(state_grid_list):
                        #     grid_state = discretize(state, state_grid)
                        #     if '{}'.format(grid_state) in test_state_dict:
                        #         test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                        #     else:
                        #         test_state_dict[i]['{}'.format(grid_state)] = 1
                        #     np.save("{}{}/train_state_dict_{}.npy".format(directory, num, grids[i]), test_state_dict[i])
                        #     count_dict['state'].append(len(test_state_dict[i]))

                        # for i, action_grid in enumerate(action_grid_list):
                        #     grid_action = discretize(action, action_grid)
                        #     if '{}'.format(grid_action) in test_action_dict:
                        #         test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
                        #     else:
                        #         test_action_dict[i]['{}'.format(grid_action)] = 1
                        #     np.save("{}{}/test_action_dict_{}.npy".format(dir,num,grids[i]), test_action_dict[i])
                        #     count_dict['action'].append(len(test_action_dict[i]))

                        # for i in range(len(grids)):
                        #     grid_state = discretize(state, state_grid_list[i])
                        #     # grid_action = discretize(action, action_grid_list[i])
                        #     grid_action = action
                        #     if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                        #         test_state_action_dict[i][
                        #             '{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                        #     else:
                        #         test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
                        #     np.save("{}{}/train_state_action_{}.npy".format(directory, num, grids[i]),
                        #             test_state_action_dict[i])

                            # count_dict['state-action'].append(len(test_state_action_dict[i]))
                        ###########################################################################################

            log_f.write('{},{}\n'.format(e, score))
            log_f.flush()
            scores.append(score)
            np.save("{}{}/scores.npy".format(directory, num), scores)
            # np.save("{}{}/max_action_list.npy".format(directory, num), max_action_list)
            # np.save("{}{}/min_action_list.npy".format(directory, num), min_action_list)
            np.save("{}{}/max_state_list.npy".format(directory, num), max_list)
            np.save("{}{}/min_state_list.npy".format(directory, num), min_list)
            #########################################################################
            obj.append(goal)
            episodes.append(e)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(e, num_episodes, score, self.epsilon,
                                                                             avg_score))
            #################################################################################################
            if avg_score >= 195 and score >= 200:
                self.q_net.save((directory + str(num) + '/'))
                self.q_net.save_weights(("{}{}/net_weights.h5".format(directory, num)))
                ########################################################################################################
                # f += 1
                print("Network saved")

        # txt.close()
        log_f.close()
        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('MountainCar_Train.png')

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        training_time = end_time - start_time
        np.save("{}{}/Total_training_time".format(directory, num), training_time)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", training_time)
        print("============================================================================================")

    def test(self, num, env, num_episodes, file_type, file, graph):

        dir = './preTrained/' + env_name + '/'

        # max_action_list = np.load("{}{}/max_action_list.npy".format(dir, num))
        # min_action_list = np.load("{}{}/min_action_list.npy".format(dir, num))
        # max_state_list = np.load("{}{}/max_state_list.npy".format(dir, num))
        # min_state_list = np.load("{}{}/min_state_list.npy".format(dir, num))

        grids = [10, 50, 100, 500, 1000]
        state_grid_list = []
        action_grid_list = []
        # for g in grids:
        #     state_bins = [g for i in range(env.observation_space.shape[0])]
        #     # action_bins = [g for i in range(env.action_space.shape[0])]
        #     state_grid_list.append(create_uniform_grid(min_state_list, max_state_list, bins=state_bins))
        #     # action_grid_list.append(create_uniform_grid(min_action_list, max_action_list, bins=action_bins))

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
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.q_net.load_weights(file)
        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 195
        score = 0.0

        start_time = datetime.now().replace(microsecond=0)

        tr_s = [[] for i in range(len(grids))]
        j1 = [0 for i in range(len(grids))]
        j2 = [0 for i in range(len(grids))]
        we_s1 = [[] for i in range(len(grids))]
        we_s2 = [[] for i in range(len(grids))]

        for e in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0
            n = 0
            ss = [[] for i in range(len(grids))]
            while not done:
                # env.render()
                action = self.policy(state)
                new_state, reward, done, _ = env.step(action)
                ##############################################################################################

                ###########
                # for j, val in enumerate(state):
                #     if val >= max_temp[j]:
                #         max_list[j] = val
                #     elif val < min_temp[j]:
                #         min_list[j] = val
                #     else:
                #         max_list[j] = max_temp[j]
                #         min_list[j] = min_temp[j]
                # max_temp = copy.deepcopy(max_list)
                # min_temp = copy.deepcopy(min_list)
                ############
                for i, state_grid in enumerate(state_grid_list):
                    grid_state = discretize(state, state_grid)
                    ss[i].append(grid_state)


                    if '{}'.format(grid_state) in test_state_dict:
                        test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                    else:
                        test_state_dict[i]['{}'.format(grid_state)] = 1
                    # np.save("{}{}/test_state_dict_{}.npy".format(dir, num, grids[i]), test_state_dict[i])
                    # count_dict['state'].append(len(test_state_dict[i]))

                # for i, action_grid in enumerate(action_grid_list):
                #     grid_action = discretize(action, action_grid)
                #     if '{}'.format(grid_action) in test_action_dict:
                #         test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
                #     else:
                #         test_action_dict[i]['{}'.format(grid_action)] = 1
                #     np.save("{}{}/test_action_dict_{}.npy".format(dir,num,grids[i]), test_action_dict[i])
                #     count_dict['action'].append(len(test_action_dict[i]))

                # for i in range(len(grids)):
                #     grid_state = discretize(state, state_grid_list[i])
                #     # grid_action = discretize(action, action_grid_list[i])
                #     grid_action = action
                #     if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                #         test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                #     else:
                #         test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
                #     np.save("{}{}/test_state_action_{}.npy".format(dir, num, grids[i]), test_state_action_dict[i])
                #
                #     count_dict['state-action'].append(len(test_state_action_dict[i]))
                ###########################################################################################
                episode_score += reward
                state = new_state
                n += 1
            score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(e)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            #######################################################################################
            # np.save("{}{}/test_max_action.npy".format(dir,num), max_action_list)
            # np.save("{}{}/test_min_action.npy".format(dir,num), min_action_list)
            # np.save("{}{}/test_max_state.npy".format(dir, num), max_list)
            # np.save("{}{}/test_min_state.npy".format(dir, num), min_list)
            ####################################################################################
            np.save("{}{}/test_scores.npy".format(directory, num), scores)

            for i, state_grid in enumerate(state_grid_list):
                tr_s[i].append(ss[i])
                jac_sum1 = 0
                jac_sum2 = 0
                for ts in tr_s[i]:
                    trace1 = list(zip(ss[i], ss[i][1:]))
                    trace2 = list(zip(ts, ts[1:]))

                    jac_tr = jaccard(trace1, trace2)  # 状态轨迹（也就是状态对）的相似度比较
                    jac_s = jaccard(ss[i], ts)  # 每个回合的状态的相似度比较

                    jac_sum1 += jac_tr
                    jac_sum2 += jac_s
                # print(jac_sum)  # 总的相似度
                if jac_sum1 == 1:
                    j1[i] += 1
                    print("transition初始状态({}):".format(grids[i]), j1[i])
                    we_s1[i].append(episode_score)

                if jac_sum2 == 1:
                    j2[i] += 1
                    print("state初始状态({}):".format(grids[i]), j2[i])
                    we_s2[i].append(episode_score)
                    # print(ss[i][0])  # 初始状态
                np.save("{}{}/we_scores1_{}.npy".format(directory, num, grids[i]), we_s1[i])
                np.save("{}{}/we_scores2_{}.npy".format(directory, num, grids[i]), we_s2[i])
            ##############################################################################################
            print("Episode, States: {},{} \t\t Reward: {}".format(e, n, round(episode_score)))
        print(np.mean(scores))

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        total_time = end_time - start_time
        np.save("{}{}/test_time".format(dir, num), total_time)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", total_time)
        print("============================================================================================")

        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('MountainCar_Test.png')

        env.close()
