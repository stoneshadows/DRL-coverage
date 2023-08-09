import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

# import pybullet_envs

from PPO import PPO
import copy

def train():

    env_name = "Pendulum-v0"   # BipedalWalker-v2,Pendulum-v0,LunarLanderContinuous-v2


    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(200000)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)


    ## Note : print/log frequencies should be > than max_ep_len

    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = len(next(os.walk(log_dir))[2])

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    directory = 'PPO_preTrained' + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
    #######################################################
    num = len(next(os.walk(directory))[1])

    if not os.path.exists(directory + str(num) + '/'):
        os.makedirs(directory + str(num) + '/')
    ######################################################
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    ####################################################################
    max_list = [0 for i in range(env.observation_space.shape[0])]
    min_list = [0 for i in range(env.observation_space.shape[0])]
    max_temp = [0 for i in range(env.observation_space.shape[0])]
    min_temp = [0 for i in range(env.observation_space.shape[0])]
    # max_action_list = [0 for i in range(env.action_space.shape[0])]
    # min_action_list = [0 for i in range(env.action_space.shape[0])]
    # max_action_temp = [0 for i in range(env.action_space.shape[0])]
    # min_action_temp = [0 for i in range(env.action_space.shape[0])]
    #####################################################################

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,reward\n')


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    scores = []

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    # training loop
    # while i_episode <= max_episode:
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            #######################################################################
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

            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode,  time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)

            # break; if the episode is over
            if done:
                break

        ###############################################################################
        scores.append(current_ep_reward)
        np.save("{}{}/scores.npy".format(directory, num), scores)
        # np.save("{}{}/max_action_list.npy".format(directory, num), max_action_list)
        # np.save("{}{}/min_action_list.npy".format(directory, num), min_action_list)
        np.save("{}{}/max_state_list.npy".format(directory, num), max_list)
        np.save("{}{}/min_state_list.npy".format(directory, num), min_list)
        ####################################################################################

        log_f.write('{},{}\n'.format(i_episode, current_ep_reward))
        log_f.flush()

        print_running_reward += current_ep_reward

        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
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
    
    
    
    
    
    
    
    
