import tensorflow as tf
import numpy as np
from copy import copy
import tensorflow.contrib.layers as layers

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Actor():
    def __init__(self, n_actions, name='actor'):
        self.n_actions = n_actions
        self.name = name

    def __call__(self, obs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = obs
            x = layers.fully_connected(x, num_outputs=400, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=300, activation_fn=tf.nn.relu)
            mu = layers.fully_connected(x, num_outputs=self.n_actions, activation_fn=None)
            log_sigma = layers.fully_connected(x, num_outputs=self.n_actions, activation_fn=None)
            log_sigma = tf.clip_by_value(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
            sigma = tf.exp(log_sigma)
            distribution = tf.distributions.Normal(mu, sigma)
            e = distribution.sample()
            action = tf.nn.tanh(e)
            log_prob_e = distribution.log_prob(e)
            log_prob_action = log_prob_e - tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
            mean_action = tf.nn.tanh(mu)
            return action, log_prob_action, mean_action

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
            x = layers.fully_connected(x, num_outputs=400, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=300, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=1, activation_fn=None)
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
            x = layers.fully_connected(x, num_outputs=400, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=300, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=1, activation_fn=None)
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
    def __init__(self, observation_shape, action_shape, clip_value=0.5, alpha=0.2,
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
        self.target_q1 = copy(self.q1)
        self.target_q1.name = 'target_q1'
        self.target_q2 = copy(self.q2)
        self.target_q2.name = 'target_q2'
        self.actor = Actor(action_shape[-1])
        self.action, self.log_prob_action, self.mean_action = self.actor(self.obs_ph)
        self.action_next, self.log_prob_action_next, _ = self.actor(self.obs_next_ph)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.critic_with_actor1 = self.q1(self.obs_ph, self.action)
        self.critic_with_actor2 = self.q2(self.obs_ph, self.action)

        # Memory
        self.memory = Memory(size=memory_max)

        # Loss
        self.q1_obs_action = self.q1(self.obs_ph, self.action_ph)
        self.q2_obs_action = self.q2(self.obs_ph, self.action_ph)
        self.q1_target = self.target_q1(self.obs_next_ph, self.action_next)
        self.q2_target = self.target_q2(self.obs_next_ph, self.action_next)
        self.q_min_target = tf.minimum(self.q1_target, self.q1_target)
        self.soft_q_target = self.q_min_target - self.alpha * self.log_prob_action_next

        self.target_Q = self.reward_ph + gamma * (1 - self.terminals_ph) * self.soft_q_target
        self.Q1_loss = tf.reduce_mean(tf.square(self.q1_obs_action - self.target_Q))
        self.Q2_loss = tf.reduce_mean(tf.square(self.q2_obs_action - self.target_Q))

        self.q_min = tf.minimum(self.critic_with_actor1, self.critic_with_actor2)
        self.P_loss = - tf.reduce_mean(self.q_min - self.alpha * self.log_prob_action)

        self.alpha_loss = tf.reduce_mean( - self.alpha * (self.log_prob_action + self.target_entropy))
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)

        self.actor_train = minimize_and_clip(optimizer=self.optimizer, objective=self.P_loss,
                                             var_list=self.actor.trainable_vars(), clip_val=clip_value)
        self.q1_train = minimize_and_clip(optimizer=self.optimizer, objective=self.Q1_loss,
                                          var_list=self.q1.trainable_vars(), clip_val=clip_value)
        self.q2_train = minimize_and_clip(optimizer=self.optimizer, objective=self.Q2_loss,
                                          var_list=self.q2.trainable_vars(), clip_val=clip_value)
        self.alpha_train = minimize_and_clip(optimizer=self.optimizer, objective=self.alpha_loss,
                                       var_list=[self.alpha], clip_val=clip_value)

        # grads = tape.gradient(self.alpha_loss, variable)
        # self.alpha_train = self.optimizer.apply_gradients(zip(grads, variable))
        # Update
        self.update_target_q1 = []
        length_q1 = len(self.target_q1.vars())
        for i in range(length_q1):
            self.update_target_q1.append(
                tf.assign(self.target_q1.vars()[i], (1 - self.tau) * self.target_q1.vars()[i] + self.tau * self.q1.vars()[i]))
        self.update_target_q2 = []
        length_q2 = len(self.target_q2.vars())
        for i in range(length_q2):
            self.update_target_q2.append(
                tf.assign(self.target_q2.vars()[i],
                          (1 - self.tau) * self.target_q2.vars()[i] + self.tau * self.q2.vars()[i]))


    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs, evaluate=False):
        shape = (1,) + obs.shape
        obs = obs.reshape(shape)
        feed_dict = {self.obs_ph: obs}
        if evaluate:
            action = self.sess.run(self.mean_action, feed_dict=feed_dict)
        else:
            action = self.sess.run(self.action, feed_dict=feed_dict)
        return action

    def store(self, obs, action, reward, obs_, terminal):

        self.memory.store(obs, action, reward, obs_, terminal)

    def update_target_net(self):
        self.sess.run(self.update_target_q1)
        self.sess.run(self.update_target_q2)


    def train_batch(self, update_policy):
        batch = self.memory.sample(self.batch_size)

        feed_dict = {self.obs_ph : batch[0],
                     self.action_ph : batch[1],
                     self.reward_ph : batch[2].reshape(self.batch_size, 1),
                     self.obs_next_ph : batch[3],
                     self.terminals_ph : batch[4].reshape(self.batch_size, 1)}
        self.sess.run(self.q1_train, feed_dict=feed_dict)
        self.sess.run(self.q2_train, feed_dict=feed_dict)
        if update_policy:
            feed_dict_actor = {self.obs_ph : batch[0]}
            self.sess.run([self.actor_train, self.alpha_train], feed_dict=feed_dict_actor)


# self.sess.run([self.critic1.vars()[0][0][[0]], self.critic2.vars()[0][0][[0]],
#                self.target_critic1.vars()[0][0][[0]], self.target_critic2.vars()[0][0][[0]],
#                self.actor.vars()[0][0][[0]], self.target_actor.vars()[0][0][[0]]])