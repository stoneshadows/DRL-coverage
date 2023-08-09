# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()


import numpy as np
from copy import copy
# import tensorflow.contrib.layers as layers
# import tensorflow.keras.layers as layers
class Actor():
    def __init__(self, n_actions, name='actor'):
        self.n_actions = n_actions
        self.name = name

    def __call__(self, obs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = obs
            x = tf.compat.v1.layers.dense(x, 400, activation='relu')
            x = tf.compat.v1.layers.dense(x, 300, activation='relu')
            mu = tf.compat.v1.layers.dense(x, self.n_actions)
            log_sigma = tf.compat.v1.layers.dense(x, self.n_actions)
            sigma = tf.exp(log_sigma)
            distribution = tf.distributions.Normal(mu, sigma)
            e = distribution.sample()
            action = tf.nn.tanh(e)
            action_mean = tf.nn.tanh(mu)
            log_prob_e = distribution.log_prob(e)
            log_prob_action = log_prob_e - tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
            return action, log_prob_action, action_mean

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class Q_function():
    def __init__(self, name='Q_function'):
        self.name = name

    def __call__(self, obs, action):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1)
            x = tf.compat.v1.layers.dense(x, 400, activation='relu')
            x = tf.compat.v1.layers.dense(x, 300, activation='relu')
            x = tf.compat.v1.layers.dense(x, 1)
            return x

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class V_function():
    def __init__(self, name='V_function'):
        self.name = name

    def __call__(self, obs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = obs
            x = tf.compat.v1.layers.dense(x, 400, activation='relu')
            x = tf.compat.v1.layers.dense(x, 300, activation='relu')
            x = tf.compat.v1.layers.dense(x, 1, )
            return x

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class Memory():
    def __init__(self, size):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.obs_ = []
        self.terminals = []
        self.full = False
        self.node = 0
        self.size = size

    def append(self, obs, action, reward, obs_, terminal):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_.append(obs_)
        self.terminals.append(terminal)

    def replace(self, obs, action, reward, obs_, terminal, index):
        self.obs[index] = obs
        self.actions[index] = action
        self.rewards[index] = reward
        self.obs_[index] = obs_
        self.terminals[index] = terminal

    def store(self, obs, action, reward, obs_, terminal):
        if self.full == False:
            self.append(obs, action, reward, obs_, terminal)
            if len(self.obs) == self.size:
                self.full = True
        else:
            self.replace(obs, action, reward, obs_, terminal, self.node)
            self.node += 1
            if self.node >= self.size:
                self.node = 0

    def sample(self, batch_size):
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_obs_ = []
        batch_terminals = []
        batch_ids = np.random.randint(0, len(self.obs), size=batch_size)
        for i in batch_ids:
            batch_obs.append(self.obs[i])
            batch_actions.append(self.actions[i])
            batch_rewards.append(self.rewards[i])
            batch_obs_ .append(self.obs_[i])
            batch_terminals.append(self.terminals[i])
        return np.array(batch_obs), np.array(batch_actions), np.array(batch_rewards), np.array(batch_obs_), np.array(batch_terminals)




def minimize_and_clip(optimizer, objective, var_list, clip_val=0.5):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

class SAC():
    def __init__(self, observation_shape, action_shape, clip_value=None, alpha=1,
                 gamma=0.99, tau=0.001, batch_size=100, memory_max=10000, actor_lr=1e-2, critic_lr=1e-2):
        # Inputs.
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs')
        self.obs_next_ph = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs_')
        self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
        self.reward_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.action_ph = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')



        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.alpha = tf.Variable(alpha, name='alpha', dtype=tf.float32)
        self.target_entropy = - tf.constant(action_shape[-1], dtype=tf.float32)
        self.q1 = Q_function(name='q1')
        self.q2 = Q_function(name='q2')
        self.actor = Actor(action_shape[-1])
        self.value = V_function()
        self.action, self.log_prob_action, self.action_mean = self.actor(self.obs_ph)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.critic_with_actor1 = self.q1(self.obs_ph, self.action)
        self.critic_with_actor2 = self.q2(self.obs_ph, self.action)
        self.target_value = copy(self.value)
        self.target_value.name = 'target_v'


        # Memory
        self.memory = Memory(size=memory_max)

        # Loss
        self.q_obs_action1 = self.q1(self.obs_ph, self.action)
        self.q_obs_action2 = self.q2(self.obs_ph, self.action)
        self.q_min = tf.minimum(self.q_obs_action1, self.q_obs_action2)
        self.target_V = self.q_min - self.alpha * self.log_prob_action
        self.V_loss = tf.reduce_mean(tf.square(self.value(self.obs_ph) - self.target_V))


        self.target_Q = self.reward_ph + gamma * (1 - self.terminals_ph) * self.target_value(self.obs_next_ph)
        self.Q1_loss = tf.reduce_mean(tf.square(self.q1(self.obs_ph, self.action_ph) - self.target_Q))
        self.Q2_loss = tf.reduce_mean(tf.square(self.q2(self.obs_ph, self.action_ph) - self.target_Q))
        self.P_min = tf.minimum(self.critic_with_actor1, self.critic_with_actor2)
        self.P_loss = - tf.reduce_mean(self.P_min - self.alpha * self.log_prob_action)
        self.alpha_loss = tf.reduce_mean( - self.alpha * (self.log_prob_action + self.target_entropy))
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
        self.actor_train = minimize_and_clip(optimizer=self.optimizer, objective=self.P_loss,
                                             var_list=self.actor.trainable_vars(), clip_val=clip_value)
        self.q1_train = minimize_and_clip(optimizer=self.optimizer, objective=self.Q1_loss,
                                          var_list=self.q1.trainable_vars(), clip_val=clip_value)
        self.q2_train = minimize_and_clip(optimizer=self.optimizer, objective=self.Q2_loss,
                                          var_list=self.q2.trainable_vars(), clip_val=clip_value)
        self.v_train = minimize_and_clip(optimizer=self.optimizer, objective=self.V_loss,
                                         var_list=self.value.trainable_vars(), clip_val=clip_value)
        self.alpha_train = minimize_and_clip(optimizer=self.optimizer, objective=self.alpha_loss,
                                             var_list=[self.alpha], clip_val=clip_value)
        # Update
        self.update_target_value = []
        length_value = len(self.target_value.vars())
        for i in range(length_value):
            self.update_target_value.append(
                tf.assign(self.target_value.vars()[i], (1 - self.tau) * self.target_value.vars()[i] + self.tau * self.value.vars()[i]))


    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs, evaluate=False):

        shape = (1, ) + obs.shape
        obs = obs.reshape(shape)
        feed_dict = {self.obs_ph : obs}
        if evaluate:
            action = self.sess.run(self.action_mean)
        else:
            action = self.sess.run(self.action, feed_dict=feed_dict)
        return action

    def store(self, obs, action, reward, obs_, terminal):

        self.memory.store(obs, action, reward, obs_, terminal)

    def update_target_net(self):
        self.sess.run(self.update_target_value)


    def train_batch(self, update_policy):
        batch = self.memory.sample(self.batch_size)

        feed_dict = {self.obs_ph : batch[0],
                            self.action_ph : batch[1],
                            self.reward_ph : batch[2].reshape(self.batch_size, 1),
                            self.obs_next_ph : batch[3],
                            self.terminals_ph : batch[4].reshape(self.batch_size, 1)}
        self.sess.run(self.v_train, feed_dict=feed_dict)
        self.sess.run(self.q1_train, feed_dict=feed_dict)
        self.sess.run(self.q2_train, feed_dict=feed_dict)
        if update_policy:
            feed_dict_actor = {self.obs_ph : batch[0]}
            self.sess.run([self.actor_train, self.alpha_train], feed_dict=feed_dict_actor)


# self.sess.run([self.critic1.vars()[0][0][[0]], self.critic2.vars()[0][0][[0]],
#                self.target_critic1.vars()[0][0][[0]], self.target_critic2.vars()[0][0][[0]],
#                self.actor.vars()[0][0][[0]], self.target_actor.vars()[0][0][[0]]])