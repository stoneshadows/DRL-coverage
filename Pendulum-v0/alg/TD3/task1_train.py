# import sys,os
# curr_path = os.path.dirname(__file__)
# parent_path=os.path.dirname(curr_path)
# sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import numpy as np
import datetime


from TD3.agent import TD3
from common.plot import plot_rewards
from common.utils import save_results,make_dir
# from datetime import datetime
import os
import copy
import numpy as np

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
	

class TD3Config:
	def __init__(self) -> None:
		# env_name = 'Pendulum-v0'
		# directory = './preTrained/' + env_name + '/'
		# num = len(next(os.walk(directory))[1])
		#
		# if not os.path.exists(directory + str(num) + '/'):
		# 	os.makedirs(directory + str(num) + '/')

		self.algo = 'TD3'
		self.env = 'Pendulum-v0'
		self.seed = 0
		# self.result_path = curr_path + directory + str(num) + '/'  # path to save results
		# self.model_path = curr_path+ directory + str(num) + '/' # path to save models
		self.start_timestep = 25e3 # Time steps initial random policy is used
		self.start_ep = 50 # Episodes initial random policy is used
		self.eval_freq = 10 # How often (episodes) we evaluate
		self.train_eps = 600
		self.max_timestep = 100000 # Max time steps to run environment
		self.expl_noise = 0.1 # Std of Gaussian exploration noise
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.9 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.noise_clip = 0.3  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(env,agent, seed, eval_episodes=10):
	eval_env = gym.make(env)
	eval_env.seed(seed + 100)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			# eval_env.render()
			action = agent.choose_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

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
	rewards = []
	ma_rewards = [] # moveing average reward
	scores = []

	from datetime import datetime
	start_time = datetime.now().replace(microsecond=0)
	for i_ep in range(int(cfg.train_eps)):
		ep_reward = 0
		ep_timesteps = 0
		state, done = env.reset(), False
		while not done:
			ep_timesteps += 1
			# Select action randomly or according to policy
			if i_ep < cfg.start_ep:
				action = env.action_space.sample()
			else:
				action = (
					agent.choose_action(np.array(state))
					+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
				).clip(-max_action, max_action)
			# Perform action
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
			done_bool = float(done) if ep_timesteps < env._max_episode_steps else 0
			# Store data in replay buffer
			agent.memory.push(state, action, next_state, reward, done_bool)
			state = next_state
			ep_reward += reward
			# Train agent after collecting sufficient data

			if i_ep+1 >= cfg.start_ep:
				agent.update()
		print(f"Episode:{i_ep+1}/{cfg.train_eps}, Step:{ep_timesteps}, Reward:{ep_reward:.3f}")
		rewards.append(ep_reward)
		# 计算滑动窗口的reward
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

	return rewards, ma_rewards


if __name__ == "__main__":
	env_name = 'Pendulum-v0'
	directory = './preTrained/' + env_name + '/'
	num = len(next(os.walk(directory))[1])

	if not os.path.exists(directory + str(num) + '/'):
		os.makedirs(directory + str(num) + '/')
	cfg  = TD3Config()
	result_path = directory + str(num) + '/results/'
	model_path = directory + str(num) + '/models/'

	env = gym.make(cfg.env)
	env.seed(cfg.seed) # Set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	agent = TD3(state_dim,action_dim,max_action,cfg)
	rewards,ma_rewards = train(cfg,env,agent)
	make_dir(result_path,model_path)
	agent.save(path=model_path)
	save_results(rewards,ma_rewards,tag='train',path=result_path)
	plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=result_path)


		
