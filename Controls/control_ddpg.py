import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from Utility import utility as ut

""" 2016 DDPG """


class DDPG(object):
    def __init__(self, sysid, config):
        self.sysid = sysid
        self.config = config
        self.seed = config.seed

        self.tf_data_type = tf.float64
        self.np_data_type = np.float64

        self.x_dim = sysid.x_est_dim
        self.x_min = sysid.x_est_min
        self.x_max = sysid.x_est_max

        self.u_dim = sysid.plant.u_dim
        self.u_min = sysid.plant.u_min
        self.u_max = sysid.plant.u_max

        self.actor_node_num = config.DDPG_actor_node_num
        self.actor_act_fcn = config.DDPG_actor_activation_fcn
        self.actor_learning_rate = config.DDPG_actor_learning_rate
        self.actor_target_update_parameter = config.DDPG_actor_target_update_parameter

        self.critic_node_num = config.DDPG_critic_node_num
        self.critic_act_fcn = config.DDPG_critic_activation_fcn
        self.critic_learning_rate = config.DDPG_critic_learning_rate
        self.critic_target_update_parameter = config.DDPG_critic_target_update_parameter

        self.batch_size = config.DDPG_batch_size
        self.discount_factor = config.DDPG_discount_factor
        self.buffer_size = config.DDPG_buffer_size
        self.replay_buffer = ut.ReplayBuffer(buffer_size=self.buffer_size, seed=self.seed)

        # Set up Ornstein-Uhlenbeck process for exploration
        self.ou_theta = config.DDPG_ou_linear_coefficient
        self.ou_sigma = config.DDPG_ou_diffusion
        self.noise = np.zeros(self.u_dim, dtype=self.np_data_type)

        # Other parameters
        self.value_max = config.DDPG_value_max if config.DDPG_value_max is not None else 1e5

        # Set up actor
        self.actor = self._set_up_actor(0)
        self.actor_target = self._set_up_actor(1)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_learning_rate)

        # Set up critic
        self.critic = self._set_up_critic(0)
        self.critic_target = self._set_up_critic(1)
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
        self.critic_loss_fcn = keras.losses.MeanSquaredError()

    def control(self, x):
        # Descaled in, descaled out
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = self.actor(scaled_x.reshape(1, -1), training=False).numpy().squeeze()
        self.noise += -self.ou_theta*self.noise \
                      + self.ou_sigma*tf.random.normal([self.u_dim], 0, 1, dtype=self.tf_data_type)
        scaled_u = np.clip(scaled_u + self.noise, -1, 1)
        u = ut.zero_mean_descale(scaled_u, self.u_min, self.u_max)
        return u

    def control_without_exploration(self, x):
        # Descaled in, descaled out
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = self.actor(scaled_x.reshape(1, -1), training=False).numpy().squeeze()
        u = ut.zero_mean_descale(scaled_u, self.u_min, self.u_max)
        return u

    def save_data_to_buffer(self, x, u, c, xp, is_terminal):
        # Descaled in, save the scaled one
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = ut.zero_mean_scale(u, self.u_min, self.u_max)
        scaled_xp = ut.zero_mean_scale(xp, self.x_min, self.x_max)
        self.replay_buffer.add(scaled_x, scaled_u, c, scaled_xp, is_terminal)

    def update_critic(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return self._update_critic(len(self.replay_buffer.buffer))
        else:
            return self._update_critic_with_fixed_batch_size()

    def update_actor(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return self._update_actor(len(self.replay_buffer.buffer))
        else:
            return self._update_actor_with_fixed_batch_size()

    def update_critic_target(self):
        c_nn_weight = self.critic.get_weights()
        ct_nn_weight = self.critic_target.get_weights()
        for k in range(len(c_nn_weight)):
            ct_nn_weight[k] += self.critic_target_update_parameter*(c_nn_weight[k] - ct_nn_weight[k])
        self.critic_target.set_weights(ct_nn_weight)

    def update_actor_target(self):
        a_nn_weight = self.actor.get_weights()
        at_nn_weight = self.actor_target.get_weights()
        for k in range(len(a_nn_weight)):
            at_nn_weight[k] += self.actor_target_update_parameter*(a_nn_weight[k] - at_nn_weight[k])
        self.actor_target.set_weights(at_nn_weight)

    def save_controller(self, directory, name):
        control_parameters = [self.sysid, self.config, self.replay_buffer,
                              self.actor.get_weights(), self.actor_target.get_weights(),
                              self.critic.get_weights(), self.critic_target.get_weights()]
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'wb') as handle:
            pickle.dump(control_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_controller(self, directory, name):
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'rb') as handle:
            control_parameters = pickle.load(handle)
        self.__init__(control_parameters[0], control_parameters[1])
        self.replay_buffer = control_parameters[2]
        self.actor.set_weights(control_parameters[3])
        self.actor_target.set_weights(control_parameters[4])
        self.critic.set_weights(control_parameters[5])
        self.critic_target.set_weights(control_parameters[6])

    def _update_critic(self, batch_size):
        x_b, u_b, c_b, xp_b, t_b = self.replay_buffer.sample(batch_size=batch_size)
        t_idx_b = tf.cast(tf.where(t_b, tf.fill(tf.shape(t_b), 1.), tf.fill(tf.shape(t_b), 0.)), self.tf_data_type)
        t_idx_b = tf.reshape(t_idx_b, [-1, 1])

        xu_b = tf.concat([x_b, u_b], 1)
        up_b = self.actor_target(xp_b, training=False)
        xup_b = tf.concat([xp_b, up_b], 1)
        target_b = c_b + self.discount_factor*self.critic_target(xup_b, training=False)*(1 - t_idx_b)
        target_b = tf.clip_by_value(target_b, c_b, self.value_max)
        with tf.GradientTape() as tape:
            predicted_b = self.critic(xu_b, training=True)
            loss_value = self.critic_loss_fcn(target_b, predicted_b)
        grads = tape.gradient(loss_value, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss_value.numpy()

    def _update_critic_with_fixed_batch_size(self):
        # Fixed batch size
        x_b, u_b, c_b, xp_b, t_b = self.replay_buffer.sample(batch_size=self.batch_size)
        t_idx_b = tf.cast(tf.where(t_b, tf.fill(tf.shape(t_b), 1.), tf.fill(tf.shape(t_b), 0.)), self.tf_data_type)
        t_idx_b = tf.reshape(t_idx_b, [-1, 1])

        xu_b = tf.concat([x_b, u_b], 1)
        up_b = self.actor_target(xp_b, training=False)
        xup_b = tf.concat([xp_b, up_b], 1)
        target_b = c_b + self.discount_factor*self.critic_target(xup_b, training=False)*(1 - t_idx_b)
        target_b = tf.clip_by_value(target_b, c_b, self.value_max)
        loss_value = self._train_critic(xu_b, target_b).numpy()
        return loss_value

    @tf.function
    def _train_critic(self, xu_b, target_b):
        with tf.GradientTape() as tape:
            predicted_b = self.critic(xu_b, training=True)
            loss_value = self.critic_loss_fcn(target_b, predicted_b)
        grads = tape.gradient(loss_value, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss_value

    def _update_actor(self, batch_size):
        x_b, _, _, _, _ = self.replay_buffer.sample(batch_size=batch_size)
        with tf.GradientTape() as tape:
            u_b = self.actor(x_b, training=True)
            xu_b = tf.concat([x_b, u_b], 1)
            loss_value = self.critic(xu_b, training=False)  # Minimization
        grads = tape.gradient(loss_value, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return np.mean(loss_value.numpy())

    def _update_actor_with_fixed_batch_size(self):
        # Fixed batch size
        x_b, _, _, _, _ = self.replay_buffer.sample(batch_size=self.batch_size)
        loss_value = np.mean(self._train_actor(x_b).numpy())
        return loss_value

    @tf.function
    def _train_actor(self, x_b):
        with tf.GradientTape() as tape:
            u_b = self.actor(x_b, training=True)
            xu_b = tf.concat([x_b, u_b], 1)
            loss_value = self.critic(xu_b, training=False)  # Minimization
        grads = tape.gradient(loss_value, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss_value

    def _set_up_actor(self, idx):
        actor = keras.Sequential(name="Actor" + str(idx))
        actor.add(keras.Input(shape=(self.x_dim,), dtype=self.tf_data_type))
        for layer_idx, node_num in enumerate(self.actor_node_num):
            actor.add(keras.layers.Dense(node_num, activation=self.actor_act_fcn[layer_idx],
                                         dtype=self.tf_data_type, name="Actor" + str(idx) + str(layer_idx)))
        # actor.summary()
        return actor

    def _set_up_critic(self, idx):
        critic = keras.Sequential(name="Critic" + str(idx))
        critic.add(keras.Input(shape=(self.x_dim + self.u_dim,), dtype=self.tf_data_type))
        for layer_idx, node_num in enumerate(self.critic_node_num):
            critic.add(keras.layers.Dense(node_num, activation=self.critic_act_fcn[layer_idx],
                                          dtype=self.tf_data_type, name="Critic" + str(idx) + str(layer_idx)))
        # critic.summary()
        return critic




