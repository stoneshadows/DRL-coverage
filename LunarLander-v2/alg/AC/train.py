from test import test
from model import ActorCritic
import torch
import torch.optim as optim
import gym
from datetime import datetime
import os
import copy
import numpy as np
def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543
    
    torch.manual_seed(random_seed)
    env_name = 'LunarLander-v2'
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr,betas)

    ep_reward = 0
    running_reward = 0
    ####################################################################
    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)


    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    log_f_name = log_dir + '/AC_' + env_name + "_log_" + str(run_num) + ".csv"
    log_f = open(log_f_name,"w+")
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
    # max_action_list = [0 for i in range(env.action_space.shape[0])]
    # min_action_list = [0 for i in range(env.action_space.shape[0])]
    # max_action_temp = [0 for i in range(env.action_space.shape[0])]
    # min_action_temp = [0 for i in range(env.action_space.shape[0])]
    #####################################################################
    scores = []
    start_time = datetime.now().replace(microsecond=0)
    for i_episode in range(0, 10000):
        state = env.reset()
        for t in range(10000):
            action = policy(state)
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
            next_state, reward, done, _ = env.step(action)
            state = next_state
            policy.rewards.append(reward)
            running_reward += reward
            ep_reward += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()
        
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        ###############################################################################
        scores.append(ep_reward)
        np.save("{}{}/scores.npy".format(directory, num), scores)
        # np.save("{}{}/max_action_list.npy".format(directory,num), max_action_list)
        # np.save("{}{}/min_action_list.npy".format(directory,num), min_action_list)
        np.save("{}{}/max_state_list.npy".format(directory,num), max_list)
        np.save("{}{}/min_state_list.npy".format(directory,num), min_list)
        log_f.write('{},{}\n'.format(i_episode,ep_reward))
        log_f.flush()
        ep_reward = 0
        ####################################################################################

        if running_reward > 4000:
            torch.save(policy.state_dict(), directory + str(num) + '/'+'LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            print("########## Solved! ##########")
            # test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            break
        
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
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
