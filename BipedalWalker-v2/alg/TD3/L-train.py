import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import copy
import os
from datetime import datetime
def train():
    ######### Hyperparameters #########
    env_name = "LunarLanderContinuous-v2"      #BipedalWalker-v2 (800 episodes);LunarLanderContinuous-v2 (1500 episodes);RoboschoolWalker2d-v1 (lr=0.002, 1400 episodes);HalfCheetah-v1 (lr=0.002, 1400 episodes)
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################
    
    env = gym.make(env_name)
    # print(env.action_space.low)
    # print(env.action_space.high)
    # print(env.action_space.shape[0])
    # print(env.observation_space.shape[0])
    # print(env.observation_space.high)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0

    log_dir = "TD3_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    log_f_name = log_dir + '/TD3_' + env_name + "_log_" + str(run_num) + ".csv"
    log_f = open(log_f_name,"w+")
    log_f.write('episode,reward\n')

    ####################################################################

    directory = './preTrained/' + env_name + '/'
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
    #####################################################################
    scores = []
    start_time = datetime.now().replace(microsecond=0)
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
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
            ##########################################################################
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            avg_reward += reward
            ep_reward += reward

            # if episode is done then update policy:
            if done or t == (max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        ###############################################################################
        scores.append(ep_reward)
        np.save("{}{}/scores.npy".format(directory, num), scores)
        np.save("{}{}/max_action_list.npy".format(directory,num), max_action_list)
        np.save("{}{}/min_action_list.npy".format(directory,num), min_action_list)
        np.save("{}{}/max_state_list.npy".format(directory,num), max_list)
        np.save("{}{}/min_state_list.npy".format(directory,num), min_list)
        ####################################################################################
        # logging updates:
        log_f.write('{},{}\n'.format(episode,ep_reward))
        log_f.flush()
        ep_reward = 0
        
        # if avg reward > 300 then save and stop traning:
        if (avg_reward/log_interval) >= 240:
            print("########## Solved! ###########")
            name = filename + '_solved'
            policy.save(directory + str(num) + '/', name)
            log_f.close()
            break


        if episode > 500:
            policy.save(directory + str(num) + '/', filename)
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    training_time = end_time - start_time
    np.save("{}{}/Total_training_time".format(directory, num), training_time)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", training_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()
