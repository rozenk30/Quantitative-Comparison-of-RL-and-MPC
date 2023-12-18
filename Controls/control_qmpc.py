####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################

"""
Firstly written by Tae Hoon Oh 2022.09 (oh.taehoon.4i@kyoto-u.ac.jp)
"""

import pickle
import casadi as ca
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from Utility import utility as ut


class QMPC(object):
    def __init__(self, sysid, config):
        self.config = config
        self.sysid = sysid
        self.seed = config.seed
        tf.random.set_seed(self.seed)
        self.decimal_point = config.decimal_point

        self.data_type = tf.float64
        self.dynamic_model = self.sysid.dynamic_model
        self.observe_model = self.sysid.observe_model
        self.abstract_stage_cost = config.mpc_abstract_stage_cost
        self.abstract_stage_constraint = config.mpc_abstract_stage_constraint
        self.abstract_terminal_constraint = config.mpc_abstract_terminal_constraint
        self.initial_control = config.mpc_initial_control

        self.x_dim = sysid.x_est_dim
        self.x_min = sysid.x_est_min
        self.x_max = sysid.x_est_max

        self.u_dim = sysid.plant.u_dim
        self.u_min = sysid.plant.u_min
        self.u_max = sysid.plant.u_max

        self.g_dim = config.QMPC_g_dim
        self.h_dim = config.QMPC_h_dim

        self.prediction_horizon = config.QMPC_prediction_horizon
        self.scale_grad = 2./(self.x_max - self.x_min)
        self.u_guess = np.zeros((self.u_dim, self.prediction_horizon + 1))

        self.critic_node_num = config.QMPC_critic_node_num
        self.critic_act_fcn = config.QMPC_critic_activation_fcn
        self.critic_learning_rate = config.QMPC_critic_learning_rate
        self.critic_target_update_parameter = config.QMPC_critic_target_update_parameter

        self.batch_size = config.QMPC_batch_size
        self.discount_factor = config.QMPC_discount_factor
        self.buffer_size = config.QMPC_buffer_size
        self.replay_buffer = ut.ReplayBuffer(seed=self.seed, buffer_size=self.buffer_size)

        # Set up Ornstein-Uhlenbeck process for exploration
        self.noise_std = config.QMPC_noise_std

        # Other parameters
        self.value_max = config.QMPC_value_max if config.QMPC_value_max is not None else 1e5

        # Set up critic
        self.critic = self._set_up_critic(0)
        self.critic_target = self._set_up_critic(1)
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
        self.critic_loss_fcn = keras.losses.MeanSquaredError()

    def control(self, x):
        u_now = self.control_without_exploration(x)
        scaled_u = ut.zero_mean_scale(u_now, self.u_min, self.u_max) + self.noise_std*np.random.randn(self.u_dim)
        scaled_u = np.clip(scaled_u, -1, 1)
        u = np.round(ut.zero_mean_descale(scaled_u, self.u_min, self.u_max), self.decimal_point)
        return u

    def control_without_exploration(self, x):
        # Descaled in descaled out
        ini_x, ini_u = self._guess_generation(x)

        ca_x = ca.SX.sym('x', self.x_dim)
        ca_u = ca.SX.sym('u', self.u_dim)
        ca_xu = ca.vertcat(ca_x, ca_u)
        ca_x_d = ut.zero_mean_descale(ca_x, self.x_min, self.x_max)
        ca_u_d = ut.zero_mean_descale(ca_u, self.u_min, self.u_max)

        next_ca_x = self.dynamic_model(ca_x_d, ca_u_d, for_casadi=True)
        next_ca_x = ut.zero_mean_scale(next_ca_x, self.x_min, self.x_max)
        step = ca.Function('Step', [ca_x, ca_u], [next_ca_x], ['x', 'u'], ['x_plus'])

        # Constraint
        stage_constraint = self._stage_constraint(ca_x_d, ca_u_d, for_casadi=True)
        terminal_constraint = self._terminal_constraint(ca_x_d, ca_u_d, for_casadi=True)
        stage_constraint_func = ca.Function('g', [ca_x, ca_u], [stage_constraint], ['x', 'u'], ['gc'])
        terminal_constraint_func = ca.Function('h', [ca_x, ca_u], [terminal_constraint], ['x', 'u'], ['hc'])

        # Cost
        stage_cost = self._stage_cost(ca_x_d, ca_u_d, for_casadi=True)
        terminal_cost = self._neural_network_casadi(ca_xu, self.critic_target.get_weights())
        stage_cost_func = ca.Function('L', [ca_x, ca_u], [stage_cost], ['x', 'u'], ['Lc'])
        terminal_cost_func = ca.Function('Q', [ca_x, ca_u], [terminal_cost], ['x', 'u'], ['Vc'])

        # Start with an empty NLP
        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        cost = 0

        # For extracting x and u information
        x_plot, u_plot = [], []

        # Initial state
        xk = ca.MX.sym('x' + str(0), self.x_dim)
        w.append(xk)
        lbw = np.append(lbw, -np.ones((self.x_dim, 1)))
        ubw = np.append(ubw, np.ones((self.x_dim, 1)))
        w0 = np.append(w0, ut.zero_mean_scale(ini_x[:, 0], self.x_min, self.x_max))
        x_plot.append(xk)

        # Initial state constraint
        g.append(xk - ut.zero_mean_scale(ini_x[:, 0], self.x_min, self.x_max))
        lbg = np.append(lbg, np.zeros((self.x_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.x_dim, 1)))

        for k in range(self.prediction_horizon):
            # Input
            uk = ca.MX.sym('u' + str(k), self.u_dim)
            w.append(uk)
            lbw = np.append(lbw, -np.ones((self.u_dim, 1)))
            ubw = np.append(ubw, np.ones((self.u_dim, 1)))
            w0 = np.append(w0, ut.zero_mean_scale(ini_u[:, k], self.u_min, self.u_max))
            u_plot.append(uk)

            # Add stage cost
            cost += (self.discount_factor**k)*stage_cost_func(xk, uk)

            # Stage constraint
            g.append(stage_constraint_func(xk, uk))
            lbg = np.append(lbg, -np.inf*np.ones((self.g_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.g_dim, 1)))

            # Next state
            xk_predict = step(xk, uk)
            xk = ca.MX.sym('x' + str(k+1), self.x_dim)
            w.append(xk)
            lbw = np.append(lbw, -np.ones((self.x_dim, 1)))
            ubw = np.append(ubw, np.ones((self.x_dim, 1)))
            w0 = np.append(w0, ut.zero_mean_scale(ini_x[:, k+1], self.x_min, self.x_max))
            x_plot.append(xk)

            # Next state constraint
            g.append(xk - xk_predict)
            lbg = np.append(lbg, np.zeros((self.x_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.x_dim, 1)))

        uk = ca.MX.sym('u' + str(self.prediction_horizon), self.u_dim)
        w.append(uk)
        lbw = np.append(lbw, -np.ones((self.u_dim, 1)))
        ubw = np.append(ubw, np.ones((self.u_dim, 1)))
        w0 = np.append(w0, ut.zero_mean_scale(ini_u[:, self.prediction_horizon], self.u_min, self.u_max))
        u_plot.append(uk)

        # Add terminal cost
        cost += (self.discount_factor**self.prediction_horizon)*terminal_cost_func(xk, uk)

        # Terminal constraint
        g.append(terminal_constraint_func(xk, uk))
        lbg = np.append(lbg, -np.inf*np.ones((self.h_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.h_dim, 1)))

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)

        # Create an NLP solver
        prob = {'f': cost, 'x': w, 'g': g}

        # "linear_solver": "ma27"
        # opts = {'print_time': True, 'ipopt': {'print_level': 5}}
        opts = {'print_time': False, "ipopt": {'print_level': 0}}
        # opts = {'ipopt': {'tol': 1e-6}, 'ipopt': {'acceptable_tol': 1e-4}, 'ipopt': {'print_level': 5}}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)  # solver = ca.nlpsol('solver', 'sqpmethod', prob)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        obj_cost = sol['f']
        constraint_value = sol['g']

        x_opt, u_opt = trajectories(sol['x'])
        x_opt, u_opt = x_opt.full(), u_opt.full()

        descale_x_opt = ut.zero_mean_descale(np.transpose(x_opt), self.x_min, self.x_max)  # matrix
        descale_u_opt = ut.zero_mean_descale(np.transpose(u_opt), self.u_min, self.u_max)  # matrix
        control_value = descale_u_opt[0, :]
        control_value = np.round(control_value, self.decimal_point)
        return control_value

    def save_data_to_buffer(self, x, u, c, xp, is_terminal):
        # Descaled in, save the scaled one
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = ut.zero_mean_scale(u, self.u_min, self.u_max)
        scaled_xp = ut.zero_mean_scale(xp, self.x_min, self.x_max)
        self.replay_buffer.add(scaled_x, scaled_u, c, scaled_xp, is_terminal)

    def pre_update_critic_by_model(self, number):  # ### ##########################################
        self.replay_buffer.reset()
        for k in range(number):
            print('pre-updating critic, number is:', k)
            x = ut.zero_one_descale(np.random.random(self.x_dim), self.x_min, self.x_max)
            for kk in range(60):  # #########################
                u = ut.zero_one_descale(np.random.random(self.u_dim), self.u_min, self.u_max)
                c = self._stage_cost(x, u, for_casadi=False)
                xp = self.dynamic_model(x, u, for_casadi=False)
                self.save_data_to_buffer(x, u, np.array([c]), xp, False)
                x = xp
                self.update_critic()
        self.replay_buffer.reset()  # ######

    def update_critic(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return self._update_critic(len(self.replay_buffer.buffer))
        else:
            return self._update_critic_with_fixed_batch_size()

    def update_critic_target(self):
        c_nn_weight = self.critic.get_weights()
        ct_nn_weight = self.critic_target.get_weights()
        for k in range(len(c_nn_weight)):
            ct_nn_weight[k] += self.critic_target_update_parameter*(c_nn_weight[k] - ct_nn_weight[k])
        self.critic_target.set_weights(ct_nn_weight)

    def save_controller(self, directory, name):
        control_parameters = [self.sysid, self.config, self.replay_buffer,
                              self.critic.get_weights(), self.critic_target.get_weights()]
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'wb') as handle:
            pickle.dump(control_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_controller(self, directory, name):
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'rb') as handle:
            control_parameters = pickle.load(handle)
        self.__init__(control_parameters[0], control_parameters[1])
        self.replay_buffer = control_parameters[2]
        self.critic.set_weights(control_parameters[3])
        self.critic_target.set_weights(control_parameters[4])

    def _guess_generation(self, current_state):
        guess_x = np.zeros((self.x_dim, self.prediction_horizon + 1))  # Trajectory
        guess_u = np.zeros((self.u_dim, self.prediction_horizon + 1))  # Trajectory
        guess_x[:, 0] = current_state
        for k in range(self.prediction_horizon):
            guess_u[:, k] = np.clip(guess_u[:, k], self.u_min, self.u_max)
            guess_x[:, k+1] = self.dynamic_model(guess_x[:, k], guess_u[:, k], for_casadi=False)  # Calculate by numpy
            guess_x[:, k+1] = np.clip(guess_x[:, k+1], self.x_min, self.x_max)
        guess_u[:, -1] = self.initial_control(guess_x[:, -1])
        return guess_x, guess_u

    def _update_critic(self, batch_size):
        x_b, u_b, c_b, xp_b, t_b = self.replay_buffer.sample(batch_size=batch_size)
        t_idx_b = tf.cast(tf.where(t_b, tf.fill(tf.shape(t_b), 1.), tf.fill(tf.shape(t_b), 0.)), self.data_type)
        t_idx_b = tf.reshape(t_idx_b, [-1, 1])

        xu_b = tf.concat([x_b, u_b], 1)
        q_b = np.zeros((xp_b.shape[0], 1))
        for k in range(xp_b.shape[0]):
            q_b[k, :] = self._optimize_critic_target(xp_b[k].numpy())
        q_b = tf.convert_to_tensor(q_b, dtype=self.data_type)
        target_b = c_b + self.discount_factor*q_b*(1 - t_idx_b)
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
        t_idx_b = tf.cast(tf.where(t_b, tf.fill(tf.shape(t_b), 1.), tf.fill(tf.shape(t_b), 0.)), self.data_type)
        t_idx_b = tf.reshape(t_idx_b, [-1, 1])

        xu_b = tf.concat([x_b, u_b], 1)
        q_b = np.zeros((xp_b.shape[0], 1))
        for k in range(xp_b.shape[0]):
            q_b[k, :] = self._optimize_critic_target(xp_b[k].numpy())
        q_b = tf.convert_to_tensor(q_b, dtype=self.data_type)
        target_b = c_b + self.discount_factor * q_b * (1 - t_idx_b)
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

    def _set_up_critic(self, idx):
        critic = keras.Sequential(name="Critic" + str(idx))
        critic.add(keras.Input(shape=(self.x_dim + self.u_dim,), dtype=self.data_type))
        for layer_idx, node_num in enumerate(self.critic_node_num):
            critic.add(keras.layers.Dense(node_num, _activation=self.critic_act_fcn[layer_idx],
                                          dtype=self.data_type, name="Critic" + str(idx) + str(layer_idx)))
        # critic.summary()
        return critic

    def _optimize_critic_target(self, state):
        ca_x = ca.SX(state)
        ca_u = ca.SX.sym('u', self.u_dim)
        ca_xu = ca.vertcat(ca_x, ca_u)
        nn_sym = self._neural_network_casadi(ca_xu, self.critic_target.get_weights())
        nn_fcn = ca.Function('nn_fun', [ca_u], [nn_sym], ['u'], ['nn'])

        # Start with an empty NLP
        w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []

        u = ca.MX.sym('U', self.u_dim)
        w.append(u)
        lbw = np.append(lbw, -np.ones((self.u_dim, 1)))
        ubw = np.append(ubw, np.ones((self.u_dim, 1)))
        w0 = np.append(w0, 2*np.random.rand(self.u_dim, 1) - 1)

        cost = nn_fcn(u)

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)

        # Create an NLP solver
        prob = {'f': cost, 'x': w, 'g': g}

        # "linear_solver": "ma27"
        # opts = {'print_time': True, 'ipopt': {'print_level': 5}}
        opts = {'print_time': False, "ipopt": {'print_level': 0}}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        opt_value = np.array(sol['f'])
        xu_value = np.array(sol['x'])
        opt_u = xu_value[self.x_dim:, :]
        opt_u = opt_u.T

        return opt_value[0]

    def _neural_network_casadi(self, x, nn_weight):
        out = x
        for k in range(int(len(nn_weight)/2)):
            if self.critic_act_fcn[k] == 'swish':  #######
                out = self._activation(ca.transpose(ca.mtimes(ca.transpose(out), nn_weight[2*k])) + nn_weight[2*k + 1])
            else:
                out = ca.transpose(ca.mtimes(ca.transpose(out), nn_weight[2*k])) + nn_weight[2*k + 1]
        return out

    @staticmethod
    def _activation(x):
        z = ca.SX.zeros(x.shape)
        data_num, _ = x.shape
        for k in range(data_num):
            z[k, :] = x[k, :]/(1 + ca.exp(-x[k, :]))  # Swish
            # z[k, :] = ca.log(1 + ca.exp(x[k, :]))  # Softplus
            # z[k, :] = ca.log(1 + x[k, :]*x[k, :])  # Custom
        return z

    def _stage_cost(self, x, u, for_casadi):
        return self.abstract_stage_cost(x, u, for_casadi, observe_model=self.observe_model)

    def _stage_constraint(self, x, u, for_casadi):
        return self.abstract_stage_constraint(x, u, for_casadi, observe_model=self.observe_model)

    def _terminal_constraint(self, x, u, for_casadi):
        return self.abstract_terminal_constraint(x, u, for_casadi, observe_model=self.observe_model)


