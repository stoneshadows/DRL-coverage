#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-09-16 01:31:33
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
# curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
# parent_path = os.path.dirname(curr_path) # 父路径
# sys.path.append(parent_path) # 添加父路径到系统路径sys.path

import datetime
import gym
import torch

from DDPG.env import NormalizedActions, OUNoise
from DDPG.agent import DDPG
from common.utils import save_results,make_dir
from common.plot import plot_rewards, plot_rewards_cn

import copy
import numpy as np
# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
from datetime import datetime
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG' # 算法名称
        self.env = 'Pendulum-v0' # 环境名称
        # self.result_path = curr_path+"/outputs/" + self.env + \
        #     '/'+curr_time+'/results/'  # 保存结果的路径
        # self.model_path = curr_path+"/outputs/" + self.env + \
        #     '/'+curr_time+'/models/'  # 保存模型的路径
        self.train_eps = 600 # 训练的回合数
        self.eval_eps = 50 # 测试的回合数
        self.gamma = 0.99 # 折扣因子
        self.critic_lr = 1e-3 # 评论家网络的学习率
        self.actor_lr = 1e-4 # 演员网络的学习率
        self.memory_capacity = 8000 
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2 # 软更新参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make(cfg.env)) 
    env.seed(seed) # 随机种子
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,cfg)
    return env,agent

def train(cfg, env, agent):
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
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = [] # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    scores = []

    from datetime import datetime
    start_time = datetime.now().replace(microsecond=0)

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)

            #######################################################################
            for k, va in enumerate([action]):
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
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if (i_ep+1)%10 == 0:
            print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1, cfg.train_eps, ep_reward))
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
    print('完成训练！')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = [] # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
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
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            ##############################################################################################
            for k, va in enumerate([action]):
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
                grid_action = discretize([action], action_grid)
                if '{}'.format(grid_action) in test_action_dict:
                    test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_action_dict[i]['{}'.format(grid_action)] = 1
                np.save("{}{}/test_action_dict_{}.npy".format(directory, num, grids[i]), test_action_dict[i])

            for i in range(len(grids)):
                grid_state = discretize(state, state_grid_list[i])
                grid_action = discretize([action], action_grid_list[i])
                if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
                np.save("{}{}/test_state_action_{}.npy".format(directory, num, grids[i]), test_state_action_dict[i])

            ###########################################################################################
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        #######################################################################################
        np.save("{}{}/test_max_action.npy".format(directory, num), max_action_list)
        np.save("{}{}/test_min_action.npy".format(directory, num), min_action_list)
        np.save("{}{}/test_max_state.npy".format(directory, num), max_list)
        np.save("{}{}/test_min_state.npy".format(directory, num), min_list)
        ####################################################################################
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.eval_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
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
    num = 5  ##eval

    if not os.path.exists(directory + str(num) + '/'):
        os.makedirs(directory + str(num) + '/')
    result_path = directory + str(num) + '/results/'
    model_path = directory + str(num) + '/models/'
    cfg = DDPGConfig()
    # # 训练
    # env,agent = env_agent_config(cfg,seed=1)
    # rewards, ma_rewards = train(cfg, env, agent)
    # make_dir(result_path, model_path)
    # agent.save(path=model_path)
    # save_results(rewards, ma_rewards, tag='train', path=result_path)
    # # plot_rewards_cn(rewards, ma_rewards, tag="train", env = cfg.env, algo=cfg.algo, path=result_path)
    # plot_rewards(rewards, ma_rewards, tag="train",
    #              algo=cfg.algo, path=result_path)
    # 测试
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag = 'eval',path = result_path)
    # plot_rewards_cn(rewards,ma_rewards,tag = "eval",env = cfg.env,algo = cfg.algo,path=result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=result_path)
    
