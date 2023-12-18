"""
Firstly written by Tae Hoon Oh 2022.07 (oh.taehoon.4i@kyoto-u.ac.jp)
Utilize MATLAB System Identification to Identify the model
REQUIRE to install the matlab.engine, please see the README file.
"""
import numpy as np
from Sysids import sysid_n4sid_batch as sysid_n4sid
from Sysids import sysid_lstm_batch as sysid_lstm
from Sysids import sysid_nnarx_batch as sysid_nnarx


class SYSID(object):
    def __init__(self, plant, config):
        self.plant = plant
        self.config = config
        self.sysid_method = config.sysid_method
        self.ss_transform = config.SYSID_ss_transform if config.SYSID_ss_transform is not None else False
        self.min_max_relax = config.SYSID_min_max_relax if config.SYSID_min_max_relax is not None else 5.
        self.sysid_error = -1

        if self.sysid_method == 'N4SID':
            self.sysid = sysid_n4sid.N4SID(plant=self.plant, config=config)
            self.x_est_dim = self.sysid.x_dim

        elif self.sysid_method == 'LSTM':
            self.sysid = sysid_lstm.LSTM(plant=self.plant, config=config)
            self.x_est_dim = self.sysid.x_dim

        elif self.sysid_method == 'NNARX':
            self.sysid = sysid_nnarx.NNARX(plant=self.plant, config=config)
            self.x_est_dim = self.sysid.x_dim

        elif self.sysid_method == 'STACKING':
            self.x_est_dim = self.plant.y_dim*config.STACK_y_order + self.plant.u_dim*config.STACK_u_order \
                             + config.STACK_o_dim
            self.x_est_min = np.hstack((np.tile(self.plant.y_min, (config.STACK_y_order, )),
                                        np.tile(self.plant.u_min, (config.STACK_u_order, )),
                                        config.STACK_o_min))
            self.x_est_max = np.hstack((np.tile(self.plant.y_max, (config.STACK_y_order, )),
                                        np.tile(self.plant.u_max, (config.STACK_u_order, )),
                                        config.STACK_o_max))
        else:
            self.x_est_dim = 0
            print('No SYSID')

        self.u = []
        self.y = []
        self.u_bias = np.zeros(self.plant.u_dim)
        self.y_bias = np.zeros(self.plant.y_dim)
        self.u_scale = np.ones(self.plant.u_dim)
        self.y_scale = np.ones(self.plant.y_dim)
        self.scaled_u = []
        self.scaled_y = []
        self.ini_x = []
        self.dynamic_model = []
        self.observe_model = []
        self.batch_index = [0]

    def add_data_and_scale(self, u, y):
        # Descaled in, Scaled save
        if len(self.u) == 0:
            self.u = u
            self.y = y
            self.batch_index.append(self.batch_index[-1] + u.shape[0])
        else:
            self.u = np.vstack((self.u, u))
            self.y = np.vstack((self.y, y))
            self.batch_index.append(self.batch_index[-1] + u.shape[0])

        if self.ss_transform:
            self.u_bias = self.plant.ss_u
            self.y_bias = self.plant.ss_y
        else:
            self.u_bias = np.mean(self.u, 0)
            self.y_bias = np.mean(self.y, 0)
        self.u_scale = 10**(-10) + np.std(u, 0)
        self.y_scale = 10**(-10) + np.std(y, 0)
        self.scaled_u = (self.u - self.u_bias)/self.u_scale
        self.scaled_y = (self.y - self.y_bias)/self.y_scale

        if self.sysid_method in ['N4SID', 'LSTM', 'NNARX']:
            self.sysid.add_data_and_scale(self.scaled_u, self.u_bias, self.u_scale,
                                          self.scaled_y, self.y_bias, self.y_scale, self.batch_index)
        else:
            pass

    def do_identification(self, data_number):
        if self.sysid_method in ['N4SID', 'LSTM', 'NNARX']:
            self.sysid.do_identification(data_number)
            self.dynamic_model = self.sysid.dynamic_model
            self.observe_model = self.sysid.observe_model
            self.ini_x = self.sysid.ini_x
            self.x_est_min = self.sysid.x_min
            self.x_est_max = self.sysid.x_max
            try:
                self.sysid_error = np.mean(self.sysid.mean_absolute_error)  # ### May not be MAE
            except AttributeError:
                self.sysid_error = -1

        else:
            pass
