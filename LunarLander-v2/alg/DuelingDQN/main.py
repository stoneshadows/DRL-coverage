

from agent import Agent
import gym


env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
train = 0
test = 1
num_episodes = 100
graph = True

env_name = 'LunarLander-v2'
num = 5
file_type = 'tf'
directory = 'preTrained' + '/' + env_name + '/'
file = directory + str(num) + '/'

duelingdqn_agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=4, epsilon=1.0, batch_size=64, input_dim=[8])

if train and not test:
    duelingdqn_agent.train_model(env, num_episodes, graph)
else:
    duelingdqn_agent.test(num, env, num_episodes, file_type, file, graph)
