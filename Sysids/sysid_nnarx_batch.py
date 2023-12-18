import numpy as np
import casadi as ca
import tensorflow as tf
from tensorflow import keras
from Utility import utility as ut


class NNARX(object):
    def __init__(self, plant, config):
        self.plant = plant
        self.config = config

        self.tf_data_type = tf.float64
        self.np_data_type = np.float64
        self.plot_bool = config.plot_bool
        self.seed = config.seed if config.seed is not None else 123

        self.u_order = config.STACK_u_order
        self.y_order = config.STACK_y_order
        self.min_max_relax = config.SYSID_min_max_relax if config.SYSID_min_max_relax is not None else 5.
        self.x_dim = plant.u_dim*self.u_order + plant.y_dim*self.y_order
        self.nn_model_node_num = config.NNARX_nn_model_node_num
        self.nn_model_act_fcn = config.NNARX_nn_model_act_fcn
        self.nn_model_learning_rate = config.NNARX_learning_rate
        self.batch_size = config.NNARX_batch_size
        self.epochs = config.NNARX_epochs
        self.validation_split_ratio = config.NNARX_validation_split_ratio
        self.nn_model = []
        self.nn_weights = []

        self.ini_x = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_min = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_max = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.batch_num = 0.
        self.mean_absolute_error = 0.

        self.u_bias = []
        self.y_bias = []
        self.u_scale = []
        self.y_scale = []
        self.scaled_u = []
        self.scaled_y = []
        self.batch_index = []

    def add_data_and_scale(self, scaled_u, u_bias, u_scale, scaled_y, y_bias, y_scale, batch_index):
        self.u_bias = u_bias
        self.y_bias = y_bias
        self.u_scale = u_scale
        self.y_scale = y_scale
        self.scaled_u = scaled_u
        self.scaled_y = scaled_y
        self.batch_index = batch_index

    def do_identification(self, batch_number):
        batch_index = self.batch_index[:batch_number + 1]
        self.nn_model = self._set_up_nn()
        self.nn_weights = []

        u_dim, y_dim = self.plant.u_dim, self.plant.y_dim
        u_stack_dim, y_stack_dim = u_dim*self.u_order, y_dim*self.y_order
        m_temp = max(self.u_order, self.y_order)

        total_batch_x = np.zeros((1, y_dim*self.y_order + u_dim*self.u_order + u_dim))
        total_batch_y = np.zeros((1, y_dim))

        for index, batch_case in enumerate(batch_index[:-1]):
            batch_u = self.scaled_u[batch_case:batch_index[index + 1], :]
            batch_y = self.scaled_y[batch_case:batch_index[index + 1], :]
            batch_length = batch_u.shape[0] - m_temp + 1
            x_temp = np.zeros((batch_length - 1, y_dim*self.y_order + u_dim*self.u_order + u_dim))
            for k in range(batch_length - 1):
                for kk in range(self.y_order):
                    x_temp[k, y_dim*kk:y_dim*kk + y_dim] = batch_y[k + m_temp - kk - 1, :]
                for kk in range(self.u_order + 1):
                    x_temp[k, y_stack_dim + u_dim*kk:y_stack_dim + u_dim*kk + u_dim] = batch_u[k + m_temp - kk - 1, :]
            total_batch_x = np.vstack((total_batch_x, x_temp))
            total_batch_y = np.vstack((total_batch_y, batch_y[m_temp:, :]))
        total_batch_x = total_batch_x[1:, :]
        total_batch_y = total_batch_y[1:, :]
        fit_result = self.nn_model.fit(total_batch_x, total_batch_y,
                                       validation_split=self.validation_split_ratio,
                                       batch_size=self.batch_size,
                                       epochs=self.epochs)

        loss = fit_result.history['loss'][-1]
        self.mean_absolute_error = fit_result.history['mae'][-1]  # ### need to check if it is MAE
        print('MAE is:', np.round(self.mean_absolute_error, 4))
        self.nn_weights = self.nn_model.get_weights()

        if self.plot_bool:  # ### put it as another function later...
            import matplotlib.pyplot as plt
            y_predict = self.nn_model(total_batch_x)
            for k in range(y_dim):
                plt.plot(total_batch_y[0:115, k])
                plt.plot(y_predict[0:115, k])
                plt.show()
        self.nn_model = []  # To pickle this sysid, we need to delete tf, (weight is fine)

        self.x_min = np.min(total_batch_x[:, :-u_dim], axis=0)
        self.x_max = np.max(total_batch_x[:, :-u_dim], axis=0)
        self.x_min, self.x_max = ut.min_max_relaxation(self.x_min, self.x_max, self.min_max_relax)
        self.ini_x = np.zeros(self.x_dim, dtype=np.float64)
        self.batch_num = batch_number

    def dynamic_model(self, x_now, u_now, for_casadi=False):
        u_idx, y_idx = self.plant.u_dim*self.u_order, self.plant.y_dim*self.y_order
        u_tran = (u_now - self.u_bias)/self.u_scale
        if for_casadi:
            xu_now = ca.vcat([x_now[0:y_idx], u_tran, x_now[y_idx:]])
            y_predict = self._nn_casadi(xu_now)
            if u_idx == 0:
                x_next = ca.vcat([y_predict, x_now[0:y_idx - self.plant.y_dim]])
            else:
                x_next = ca.vcat([y_predict, x_now[0:y_idx - self.plant.y_dim],
                                  u_tran, x_now[y_idx:y_idx + u_idx - self.plant.u_dim]])
        else:
            xu_now = np.hstack((x_now[0:y_idx], u_tran, x_now[y_idx:]))
            y_predict = self._nn_numpy(xu_now)
            if u_idx == 0:
                x_next = np.hstack((y_predict, x_now[0:y_idx - self.plant.y_dim]))
            else:
                x_next = np.hstack((y_predict, x_now[0:y_idx - self.plant.y_dim],
                                    u_tran, x_now[y_idx:y_idx + u_idx - self.plant.u_dim]))
        return x_next

    def observe_model(self, x_now, for_casadi=False):
        y_now = x_now[:self.plant.y_dim]
        descaled_y = y_now*self.y_scale + self.y_bias
        return descaled_y

    def _nn_casadi(self, x):
        out = x
        for k in range(int(len(self.nn_weights)/2)):
            if k < 1:  # #####
                out = self._activation(ca.transpose(ca.mtimes(ca.transpose(out), self.nn_weights[2*k])) + self.nn_weights[2*k + 1])
            else:
                out = ca.transpose(ca.mtimes(ca.transpose(out), self.nn_weights[2*k])) + self.nn_weights[2*k + 1]
        return out

    def _nn_numpy(self, x):
        out = x
        for k in range(int(len(self.nn_weights)/2)):
            if k < 1:  # #####
                out = self._activation(out @ self.nn_weights[2*k] + self.nn_weights[2*k + 1])
            else:
                out = out @ self.nn_weights[2*k] + self.nn_weights[2*k + 1]
        return out

    def _set_up_nn(self):
        nn_model = keras.Sequential(name="nn_model")
        nn_model.add(keras.Input(shape=(self.x_dim + self.plant.u_dim,), dtype=self.tf_data_type))
        for layer_idx, node_num in enumerate(self.nn_model_node_num):
            nn_model.add(keras.layers.Dense(node_num, activation=self.nn_model_act_fcn[layer_idx],
                                            dtype=self.tf_data_type, name="nn_model" + str(layer_idx)))
        opt = keras.optimizers.Adam(learning_rate=self.nn_model_learning_rate)
        nn_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        # nn_model.summary()
        return nn_model

    @staticmethod
    def _activation(x):
        y = np.tanh(x)  # Tangent hyperbolic
        return y
