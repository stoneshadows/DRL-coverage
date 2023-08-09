

from agent import Agent
import gym


env = gym.make("CartPole-v1")
spec = gym.spec("CartPole-v1")
train = 1
test = 0
num_episodes = 1000
graph = True

env_name = 'CartPole-v1'
num = 100
file_type = 'tf'
directory = 'preTrained' + '/' + env_name + '/'
file = directory + str(num) + '/'

dqn_agent = Agent(lr=0.001, discount_factor=0.99, num_actions=2, epsilon=1.0, batch_size=64, input_dims=4)

if train and not test:
    dqn_agent.train_model(env, num_episodes, graph)
else:
    dqn_agent.test(num, env, num_episodes, file_type, file, graph)
