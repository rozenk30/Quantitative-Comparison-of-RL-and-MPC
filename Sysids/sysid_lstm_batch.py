import os
import numpy as np
import matlab.engine
import casadi as ca
from Utility import utility as ut


class LSTM(object):
    def __init__(self, plant, config):
        self.plant = plant
        self.config = config

        self.np_data_type = np.float64
        self.plot_bool = config.plot_bool
        self.node_num = config.LSTM_node_num if config.LSTM_node_num is not None else 10*plant.y_dim
        self.min_max_relax = config.SYSID_min_max_relax if config.SYSID_min_max_relax is not None else 5.
        self.learning_rate = config.LSTM_learning_rate if config.LSTM_learning_rate is not None else 0.002
        self.x_dim = 2*self.node_num

        self.l_i_u_w, self.l_i_f_w = 0., 0.
        self.l_i_c_w, self.l_i_o_w = 0., 0.
        self.l_r_u_w, self.l_r_f_w = 0., 0.
        self.l_r_c_w, self.l_r_o_w = 0., 0.
        self.l_b_u_w, self.l_b_f_w = 0., 0.
        self.l_b_c_w, self.l_b_o_w = 0., 0.
        self.l_cell_x, self.l_hidden_x = 0., 0.
        self.n_w, self.n_b = 0., 0.

        self.ini_x = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_min = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_max = np.zeros(self.x_dim, dtype=self.np_data_type)

        self.batch_num = 0
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
        data_u = self.scaled_u[:self.batch_index[batch_number], :]
        data_y = self.scaled_y[:self.batch_index[batch_number], :]

        matlab_script_name = 'matlab_lstm_batch.m'  # ### Find the path first, later change to pathlib
        current_directory = os.getcwd()
        for root, dirs, files in os.walk(current_directory):
            if matlab_script_name in files:
                matlab_script_path = root

        # Run MATLAB
        mat_u = matlab.double(data_u.tolist())
        mat_y = matlab.double(data_y.tolist())
        batch_index = matlab.double(self.batch_index[0:batch_number + 1])
        os.chdir(matlab_script_path)
        eng = matlab.engine.start_matlab()
        l_i_u_w, l_i_f_w, l_i_c_w, l_i_o_w, l_r_u_w, l_r_f_w, l_r_c_w, l_r_o_w, \
        l_b_u_w, l_b_f_w, l_b_c_w, l_b_o_w, l_cell_x, l_hidden_x, n_w, n_b, x_min, x_max, error, loss \
            = eng.matlab_lstm_batch(mat_u, mat_y, batch_index, self.node_num, self.learning_rate, self.plot_bool,
                                    nargout=20)
        eng.quit()
        os.chdir(current_directory)

        self.l_i_u_w, self.l_i_f_w = np.array(l_i_u_w, dtype=self.np_data_type), np.array(l_i_f_w, dtype=self.np_data_type)
        self.l_i_c_w, self.l_i_o_w = np.array(l_i_c_w, dtype=self.np_data_type), np.array(l_i_o_w, dtype=self.np_data_type)
        self.l_r_u_w, self.l_r_f_w = np.array(l_r_u_w, dtype=self.np_data_type), np.array(l_r_f_w, dtype=self.np_data_type)
        self.l_r_c_w, self.l_r_o_w = np.array(l_r_c_w, dtype=self.np_data_type), np.array(l_r_o_w, dtype=self.np_data_type)
        self.l_b_u_w, self.l_b_f_w = np.array(l_b_u_w, dtype=self.np_data_type), np.array(l_b_f_w, dtype=self.np_data_type)
        self.l_b_c_w, self.l_b_o_w = np.array(l_b_c_w, dtype=self.np_data_type), np.array(l_b_o_w, dtype=self.np_data_type)
        self.l_cell_x, self.l_hidden_x = np.array(l_cell_x, dtype=self.np_data_type), np.array(l_hidden_x, dtype=self.np_data_type)
        self.n_w, self.n_b = np.array(n_w, dtype=self.np_data_type), np.array(n_b, dtype=self.np_data_type)
        x_min, x_max = np.array(x_min, dtype=self.np_data_type), np.array(x_max, dtype=self.np_data_type)
        error, loss = np.array(error, dtype=self.np_data_type), np.array(loss, dtype=self.np_data_type)
        self.x_min, self.x_max = ut.min_max_relaxation(x_min.squeeze(), x_max.squeeze(), self.min_max_relax)
        self.mean_absolute_error = np.mean(error, dtype=self.np_data_type)
        self.ini_x = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.batch_num = batch_number

    def dynamic_model(self, x_now, u_now, for_casadi=False):
        u_tran = (u_now - self.u_bias)/self.u_scale
        hidden_x = x_now[0:self.node_num]
        cell_x = x_now[self.node_num:]

        if for_casadi:
            i_gate = self._gate_activation(self.l_i_u_w @ u_tran + self.l_r_u_w @ hidden_x + self.l_b_u_w)
            f_gate = self._gate_activation(self.l_i_f_w @ u_tran + self.l_r_f_w @ hidden_x + self.l_b_f_w)
            c_gate = self._activation(self.l_i_c_w @ u_tran + self.l_r_c_w @ hidden_x + self.l_b_c_w)
            o_gate = self._gate_activation(self.l_i_o_w @ u_tran + self.l_r_o_w @ hidden_x + self.l_b_o_w)
            cell_x_next = f_gate * cell_x + i_gate * c_gate
            hidden_x_next = o_gate * self._activation(cell_x_next)
            x_next = ca.vcat([hidden_x_next, cell_x_next])
        else:
            i_gate = self._gate_activation(self.l_i_u_w @ u_tran.T + self.l_r_u_w @ hidden_x.T + self.l_b_u_w.T)
            f_gate = self._gate_activation(self.l_i_f_w @ u_tran.T + self.l_r_f_w @ hidden_x.T + self.l_b_f_w.T)
            c_gate = self._activation(self.l_i_c_w @ u_tran.T + self.l_r_c_w @ hidden_x.T + self.l_b_c_w.T)
            o_gate = self._gate_activation(self.l_i_o_w @ u_tran.T + self.l_r_o_w @ hidden_x.T + self.l_b_o_w.T)
            cell_x_next = f_gate * cell_x + i_gate * c_gate
            hidden_x_next = o_gate * self._activation(cell_x_next)
            x_next = np.hstack((hidden_x_next, cell_x_next))
            x_next = x_next.squeeze()
        return x_next

    def observe_model(self, x_now, for_casadi=False):
        hidden_x = x_now[0:self.node_num]
        if for_casadi:
            y_now = self.n_w @ hidden_x + self.n_b
        else:
            y_now = self.n_w @ hidden_x.T + self.n_b.T
            y_now = y_now.squeeze()
        descaled_y = y_now*self.y_scale + self.y_bias
        return descaled_y

    @staticmethod
    def _gate_activation(x):
        y = 1 / (1 + np.exp(-x))  # Sigmoid
        return y

    @staticmethod
    def _activation(x):
        y = np.tanh(x)  # Tangent hyperbolic
        return y

