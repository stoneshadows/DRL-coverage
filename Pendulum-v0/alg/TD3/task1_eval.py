 #!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-23 20:36:23
LastEditor: JiangJi
LastEditTime: 2021-04-28 10:14:33
Discription: 
Environment: 
'''
import sys,os
# curr_path = os.path.dirname(__file__)
# parent_path=os.path.dirname(curr_path)
# sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import numpy as np

import copy


from TD3.agent import TD3
from common.plot import plot_rewards
from common.utils import save_results,make_dir

from datetime import datetime
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class TD3Config:
	def __init__(self) -> None:
		self.algo = 'TD3'
		self.env = 'Pendulum-v0'
		self.seed = 0
		# self.result_path = curr_path+"/results/" +self.env+'/'+curr_time+'/results/'  # path to save results
		# self.model_path = curr_path+"/results/" +self.env+'/'+curr_time+'/models/'  # path to save models
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.99 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.noise_clip = 0.5  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(env_name,agent, seed, eval_episodes=50):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	rewards,ma_rewards =[],[]
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
	for i_episode in range(eval_episodes):
		ep_reward = 0
		state, done = eval_env.reset(), False
		while not done:
			# eval_env.render()
			action = agent.choose_action(np.array(state))
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
				count_dict['state'].append(len(test_state_dict[i]))

			for i, action_grid in enumerate(action_grid_list):
			    grid_action = discretize(action, action_grid)
			    if '{}'.format(grid_action) in test_action_dict:
			        test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
			    else:
			        test_action_dict[i]['{}'.format(grid_action)] = 1
			    np.save("{}{}/test_action_dict_{}.npy".format(directory, num, grids[i]), test_action_dict[i])
			    count_dict['action'].append(len(test_action_dict[i]))

			for i in range(len(grids)):
				grid_state = discretize(state, state_grid_list[i])
				grid_action = discretize(action, action_grid_list[i])
				if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
					test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
				else:
					test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
				np.save("{}{}/test_state_action_{}.npy".format(directory, num, grids[i]), test_state_action_dict[i])

				count_dict['state-action'].append(len(test_state_action_dict[i]))
			###########################################################################################
			state, reward, done, _ = eval_env.step(action)
			ep_reward += reward

		#######################################################################################
		np.save("{}{}/test_max_action.npy".format(directory,num), max_action_list)
		np.save("{}{}/test_min_action.npy".format(directory,num), min_action_list)
		np.save("{}{}/test_max_state.npy".format(directory, num), max_list)
		np.save("{}{}/test_min_state.npy".format(directory, num), min_list)
		####################################################################################
		print(f"Episode:{i_episode+1}, Reward:{ep_reward:.3f}")
		rewards.append(ep_reward)
		# 计算滑动窗口的reward
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

	return rewards,ma_rewards

if __name__ == "__main__":

	num = 0
	cfg  = TD3Config()
	env = gym.make(cfg.env)
	env.seed(cfg.seed) # Set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	td3= TD3(state_dim,action_dim,max_action,cfg)
	####################################################
	env_name = 'Pendulum-v0'
	directory = './preTrained/' + env_name + '/'
	result_path = directory + str(num) + '/results/'
	model_path = directory + str(num) + '/models/'
	td3.load(model_path)
	rewards,ma_rewards = eval(cfg.env,td3,cfg.seed)
	make_dir(result_path,model_path)
	save_results(rewards,ma_rewards,tag='eval',path=result_path)
	plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=result_path)