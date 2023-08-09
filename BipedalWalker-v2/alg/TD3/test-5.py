# import roboschool, gym
import gym
from TD3 import TD3
import copy
import numpy as np
from random import choice
from PIL import Image
from datetime import datetime
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

# bins = [100 for i in range(env.observation_space.shape[0])]
# state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=bins)
def jaccard(p,q):
    c = [v for v in p if v in q]
    return float(len(c))/(len(p)+len(q)-len(c))

def test():

    env_name = "BipedalWalker-v3"  # BipedalWalker-v2;LunarLanderContinuous-v2
    env = gym.make(env_name)
    random_seed = 0
    n_episodes = 100
    lr = 0.002
    max_timesteps = 1500
    render = True
    save_gif = False
    ##########################################################################################
    num = 5
    dir = './preTrained/' + env_name + '/'

    max_action_list = np.load("{}{}/max_action_list.npy".format(dir, num))
    min_action_list = np.load("{}{}/min_action_list.npy".format(dir, num))
    max_state_list = np.load("{}{}/max_state_list.npy".format(dir, num))
    min_state_list = np.load("{}{}/min_state_list.npy".format(dir, num))

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

    filename = "TD3_{}_{}".format(env_name, random_seed)
    filename += '_solved'
    directory = "./preTrained/{}/{}/".format(env_name, num)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    policy.load_actor(directory, filename)

    start_time = datetime.now().replace(microsecond=0)

    scores = []
    tr_s = [[] for i in range(len(grids))]
    sa = []
    j1 = [0 for i in range(len(grids))]
    j2 = [0 for i in range(len(grids))]
    we_s1 = [[] for i in range(len(grids))]
    we_s2 = [[] for i in range(len(grids))]

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        ss = [[] for i in range(len(grids))]
        n = 0
        state = env.reset()
        # state = choice(np.load("L_gen_s_all.npy"))
        for t in range(1, max_timesteps + 1):
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)

    # ##############################################################################################
    #         for k, va in enumerate(action):
    #             if va >= max_action_temp[k]:
    #                 max_action_list[k] = va
    #             elif va < min_action_temp[k]:
    #                 min_action_list[k] = va
    #             else:
    #                 max_action_list[k] = max_action_temp[k]
    #                 min_action_list[k] = min_action_temp[k]
    #         max_action_temp = copy.deepcopy(max_action_list)
    #         min_action_temp = copy.deepcopy(min_action_list)
    #         ###########
    #         for j, val in enumerate(state):
    #             if val >= max_temp[j]:
    #                 max_list[j] = val
    #             elif val < min_temp[j]:
    #                 min_list[j] = val
    #             else:
    #                 max_list[j] = max_temp[j]
    #                 min_list[j] = min_temp[j]
    #         max_temp = copy.deepcopy(max_list)
    #         min_temp = copy.deepcopy(min_list)
    #         ############
            for i, state_grid in enumerate(state_grid_list):
                grid_state = discretize(state, state_grid)
                ss[i].append(grid_state)

                if '{}'.format(grid_state) in test_state_dict:
                    test_state_dict[i]['{}'.format(grid_state)] += 1  # 出现过的状态，并记录出现次数
                else:
                    test_state_dict[i]['{}'.format(grid_state)] = 1
    #             np.save("{}{}/test_state_dict_{}.npy".format(dir, num, grids[i]), test_state_dict[i])
    #
    #
    #         for i, action_grid in enumerate(action_grid_list):
    #             grid_action = discretize(action, action_grid)
    #             if '{}'.format(grid_action) in test_action_dict:
    #                 test_action_dict[i]['{}'.format(grid_action)] += 1  # 出现过的状态，并记录出现次数
    #             else:
    #                 test_action_dict[i]['{}'.format(grid_action)] = 1
    #             np.save("{}{}/test_action_dict_{}.npy".format(dir, num, grids[i]), test_action_dict[i])
    #
    #
    #         for i in range(len(grids)):
    #             grid_state = discretize(state, state_grid_list[i])
    #             grid_action = discretize(action, action_grid_list[i])
    #             if '{},{}'.format(grid_state, grid_action) in test_state_action_dict:
    #                 test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] += 1  # 出现过的状态，并记录出现次数
    #             else:
    #                 test_state_action_dict[i]['{},{}'.format(grid_state, grid_action)] = 1
    #             np.save("{}{}/test_state_action_{}.npy".format(dir, num, grids[i]), test_state_action_dict[i])
    #
    #
    #         ###########################################################################################
            state = next_state
            n += 1
            ep_reward += reward
            # if render:
            #     env.render()

            #     if save_gif:
            #          img = env.render(mode = 'rgb_array')
            #          img = Image.fromarray(img)
            #          img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        # print(np.mean(ep_reward))
        scores.append(ep_reward)
        np.save("{}/test_scores.npy".format(directory), scores)
        ########################################################################################
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
            # print(jac_sum1)  # 总的相似度
            if jac_sum1 == 1:
                j1[i] += 1
                print("transition初始状态({}):".format(grids[i]), j1[i])
                we_s1[i].append(ep_reward)

            if jac_sum2 == 1:
                j2[i] += 1
                print("state初始状态({}):".format(grids[i]), j2[i])
                we_s2[i].append(ep_reward)
                # print(ss[i][0])  # 初始状态
            np.save("{}/we_scores1_{}.npy".format(directory,  grids[i]), we_s1[i])
            np.save("{}/we_scores2_{}.npy".format(directory,  grids[i]), we_s2[i])

        print('Episode, States: {},{} \t\t Reward: {}'.format(ep, n, round(ep_reward, 2)))

        # #######################################################################################
        # np.save("{}{}/test_max_action.npy".format(dir,num), max_action_list)
        # np.save("{}{}/test_min_action.npy".format(dir,num), min_action_list)
        # np.save("{}{}/test_max_state.npy".format(dir,num), max_list)
        # np.save("{}{}/test_min_state.npy".format(dir,num), min_list)
        # ####################################################################################

        # print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
    env.close()
    print(np.mean(scores))
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time
    np.save("{}{}/test_time".format(dir, num), total_time)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", total_time)
    print("============================================================================================")
                
if __name__ == '__main__':
    test()
    
    
    
