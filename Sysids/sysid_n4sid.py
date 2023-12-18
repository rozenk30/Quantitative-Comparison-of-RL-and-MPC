import os
import numpy as np
import matlab.engine
from Utility import utility as ut


class N4SID(object):
    def __init__(self, plant, config):
        self.plant = plant
        self.config = config

        self.np_data_type = np.float64
        self.plot_bool = config.plot_bool
        self.x_dim = config.N4SID_x_dim
        self.min_max_relax = config.SYSID_min_max_relax if config.SYSID_min_max_relax is not None else 5.

        self.a = np.zeros((self.x_dim, self.x_dim), dtype=self.np_data_type)
        self.b = np.zeros((self.x_dim, self.plant.u_dim), dtype=self.np_data_type)
        self.c = np.zeros((self.plant.y_dim, self.plant.x_dim), dtype=self.np_data_type)
        self.d = np.zeros((self.plant.y_dim, self.plant.u_dim), dtype=self.np_data_type)

        self.ini_x = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_min = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_max = np.zeros(self.x_dim, dtype=self.np_data_type)

        self.data_num = 0
        self.mean_absolute_error = 0.

        self.u_bias = []
        self.y_bias = []
        self.u_scale = []
        self.y_scale = []
        self.scaled_u = []
        self.scaled_y = []

    def add_data_and_scale(self, scaled_u, u_bias, u_scale, scaled_y, y_bias, y_scale):
        self.u_bias = u_bias
        self.y_bias = y_bias
        self.u_scale = u_scale
        self.y_scale = y_scale
        self.scaled_u = scaled_u
        self.scaled_y = scaled_y

    def do_identification(self, data_number):
        data_u = self.scaled_u[:data_number, :]
        data_y = self.scaled_y[:data_number, :]

        matlab_script_name = 'matlab_n4sid.m'  # ### Find the path first, later change to pathlib
        current_directory = os.getcwd()
        for root, dirs, files in os.walk(current_directory):
            if matlab_script_name in files:
                matlab_script_path = root

        # Run MATLAB
        mat_u = matlab.double(data_u.tolist())
        mat_y = matlab.double(data_y.tolist())
        os.chdir(matlab_script_path)
        eng = matlab.engine.start_matlab()
        a, b, c, d, ini_x, x_min, x_max, error = \
            eng.matlab_n4sid(mat_u, mat_y, float(self.plant.time_interval), float(self.x_dim), self.plot_bool, nargout=8)
        eng.quit()
        os.chdir(current_directory)

        self.a, self.b = np.array(a, dtype=self.np_data_type), np.array(b, dtype=self.np_data_type)
        self.c, self.d = np.array(c, dtype=self.np_data_type), np.array(d, dtype=self.np_data_type)
        x_min, x_max = np.array(x_min, dtype=self.np_data_type), np.array(x_max, dtype=self.np_data_type)
        ini_x, error = np.array(ini_x, dtype=self.np_data_type), np.array(error, dtype=self.np_data_type)
        ini_x, x_min, x_max = ini_x.squeeze(), x_min.squeeze(), x_max.squeeze()
        self.x_min, self.x_max = ut.min_max_relaxation(x_min, x_max, self.min_max_relax)
        self.mean_absolute_error = error
        self.ini_x = ini_x
        self.data_num = data_number

    def dynamic_model(self, x_now, u_now, for_casadi=False):
        scaled_u = (u_now - self.u_bias) / self.u_scale
        x_next = self.a @ x_now + self.b @ scaled_u
        return x_next

    def observe_model(self, x_now, for_casadi=False):
        y_now = self.c @ x_now
        descaled_y = y_now*self.y_scale + self.y_bias
        return descaled_y



