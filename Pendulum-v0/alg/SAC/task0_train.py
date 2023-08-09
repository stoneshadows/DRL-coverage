#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-29 12:59:22
LastEditor: JiangJi
LastEditTime: 2021-05-06 16:58:01
Discription: 
Environment: 
'''
import sys,os
# curr_path = os.path.dirname(__file__)
# parent_path = os.path.dirname(curr_path)
# sys.path.append(parent_path)  # add current terminal path to sys.path


import gym
import torch
import datetime

from SAC.env import NormalizedActions
from SAC.agent import SAC
from common.utils import save_results, make_dir
from common.plot import plot_rewards
import copy
import numpy as np
# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
from datetime import datetime
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

class SACConfig:
    def __init__(self) -> None:
        self.algo = 'SAC'
        self.env = 'Pendulum-v0'
        # self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        # self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 600
        self.train_steps = 500
        self.eval_eps = 50
        self.eval_steps = 500
        self.gamma = 0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2
        self.value_lr  = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size  = 128
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make("Pendulum-v0"))
    env.seed(seed)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    agent = SAC(state_dim,action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    ####################################################################
    env_name = 'Pendulum-v0'
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

    num = len(next(os.walk(directory))[1]) - 1

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
    #####################################################################
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    scores = []
    from datetime import datetime
    start_time = datetime.now().replace(microsecond=0)

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            action = agent.policy_net.get_action(state)
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
            for j, val in enumerate(state):
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
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            if done:
                break
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        ###############################################################################
        scores.append(ep_reward)
        np.save("{}{}/scores.npy".format(directory, num), scores)
        np.save("{}{}/max_action_list.npy".format(directory, num), max_action_list)
        np.save("{}{}/min_action_list.npy".format(directory, num), min_action_list)
        np.save("{}{}/max_state_list.npy".format(directory, num), max_list)
        np.save("{}{}/min_state_list.npy".format(directory, num), min_list)
        log_f.write('{},{}\n'.format(i_ep, ep_reward))
        log_f.flush()
    log_f.close()
    ####################################################################################
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    training_time = end_time - start_time
    np.save("{}{}/Total_training_time".format(directory, num), training_time)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", training_time)
    print("============================================================================================")
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward
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
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        for i_step in range(cfg.eval_steps):
            action = agent.policy_net.get_action(state)
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
            for j, val in enumerate(state):
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
                grid_state = discretize(state, state_grid)
                if '{}'.format(grid_state) in test_state_dict:
                    test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_dict[i]['{}'.format(grid_state)] = 1
                np.save("{}{}/test_state_dict_{}.npy".format(directory, num, grids[i]), test_state_dict[i])


            for i, action_grid in enumerate(action_grid_list):
                grid_action = discretize(action, action_grid)
                if '{}'.format(grid_action) in test_action_dict:
                    test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_action_dict[i]['{}'.format(grid_action)] = 1
                np.save("{}{}/test_action_dict_{}.npy".format(directory, num, grids[i]), test_action_dict[i])


            for i in range(len(grids)):
                grid_state = discretize(state, state_grid_list[i])
                grid_action = discretize(action, action_grid_list[i])
                if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
                np.save("{}{}/test_state_action_{}.npy".format(directory, num, grids[i]), test_state_action_dict[i])

            ###########################################################################################
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        #######################################################################################
        np.save("{}{}/test_max_action.npy".format(directory, num), max_action_list)
        np.save("{}{}/test_min_action.npy".format(directory, num), min_action_list)
        np.save("{}{}/test_max_state.npy".format(directory, num), max_list)
        np.save("{}{}/test_min_state.npy".format(directory, num), min_list)
        ####################################################################################

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time
    np.save("{}{}/test_time".format(directory, num), total_time)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", total_time)
    print("============================================================================================")
    print('Complete evaling！')
    return rewards, ma_rewards

if __name__ == "__main__":
    env_name = 'Pendulum-v0'
    directory = './preTrained/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # num = len(next(os.walk(directory))[1]) ########################################################## train
    num = 5 ##eval

    if not os.path.exists(directory + str(num) + '/'):
        os.makedirs(directory + str(num) + '/')
    result_path = directory + str(num) + '/results/'
    model_path = directory + str(num) + '/models/'
    cfg = SACConfig()
    # # train
    # env,agent = env_agent_config(cfg,seed=1)
    # rewards, ma_rewards = train(cfg, env, agent)
    # make_dir(result_path, model_path)
    # agent.save(path=model_path)
    # save_results(rewards, ma_rewards, tag='train', path=result_path)
    # plot_rewards(rewards, ma_rewards, tag="train",
    #              algo=cfg.algo, path=result_path)

    ##############eval
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=result_path)




