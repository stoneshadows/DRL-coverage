from model import ActorCritic
import torch
import gym
import copy
import numpy as np

from PIL import Image
from datetime import datetime
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def test(n_episodes=100):
    num = 5
    env = gym.make('LunarLander-v2')
    env_name = 'LunarLander-v2'
    policy = ActorCritic()
    lr = 0.02
    betas = (0.9, 0.999)
    directory = './preTrained/' + env_name + '/'
    policy.load_state_dict(torch.load(directory + str(num) + '/'+'LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1])))
    ##########################################################################################


    # max_action_list = np.load("{}{}/max_action_list.npy".format(dir, num))
    # min_action_list = np.load("{}{}/min_action_list.npy".format(dir, num))
    max_state_list = np.load("{}{}/max_state_list.npy".format(directory, num))
    min_state_list = np.load("{}{}/min_state_list.npy".format(directory, num))

    grids = [10, 50, 100, 500, 1000]
    state_grid_list = []
    action_grid_list = []
    for g in grids:
        state_bins = [g for i in range(env.observation_space.shape[0])]
        # action_bins = [g for i in range(env.action_space.shape[0])]
        state_grid_list.append(create_uniform_grid(min_state_list, max_state_list, bins=state_bins))
        # action_grid_list.append(create_uniform_grid(min_action_list, max_action_list, bins=action_bins))

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
    render = True
    save_gif = False

    start_time = datetime.now().replace(microsecond=0)
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            ##############################################################################################
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
            ############
            for i, state_grid in enumerate(state_grid_list):
                grid_state = discretize(state, state_grid)
                if '{}'.format(grid_state) in test_state_dict:
                    test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_dict[i]['{}'.format(grid_state)] = 1
                # np.save("{}{}/test_state_dict_{}.npy".format(directory, num, grids[i]), test_state_dict[i])
                count_dict['state'].append(len(test_state_dict[i]))

            # for i, action_grid in enumerate(action_grid_list):
            #     grid_action = discretize(action, action_grid)
            #     if '{}'.format(grid_action) in test_action_dict:
            #         test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
            #     else:
            #         test_action_dict[i]['{}'.format(grid_action)] = 1
            #     np.save("{}{}/test_action_dict_{}.npy".format(directory, num, grids[i]), test_action_dict[i])
            #     count_dict['action'].append(len(test_action_dict[i]))

            for i in range(len(grids)):
                grid_state = discretize(state, state_grid_list[i])
                grid_action = action
                if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
                # np.save("{}{}/test_state_action_{}.npy".format(directory, num, grids[i]), test_state_action_dict[i])

                count_dict['state-action'].append(len(test_state_action_dict[i]))
            ###########################################################################################
            next_state, reward, done, _ = env.step(action)
            state = next_state
            running_reward += reward
            # if render:
            #      env.render()
            #      if save_gif:
            #          img = env.render(mode = 'rgb_array')
            #          img = Image.fromarray(img)
            #          img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        #######################################################################################
        # np.save("{}{}/test_max_action.npy".format(directory,num), max_action_list)
        # np.save("{}{}/test_min_action.npy".format(directory,num), min_action_list)
        # np.save("{}{}/test_max_state.npy".format(directory,num), max_list)
        # np.save("{}{}/test_min_state.npy".format(directory,num), min_list)
        ####################################################################################

        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time
    # np.save("{}{}/test_time".format(directory, num), total_time)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", total_time)
    print("============================================================================================")
            
if __name__ == '__main__':
    test()
