import numpy as np
from Utility import utility as ut


class StackingEstimator(object):
    def __init__(self, sysid, config):
        self.sysid = sysid
        self.config = config

        self.u_bias = sysid.u_bias
        self.y_bias = sysid.y_bias
        self.u_scale = sysid.u_scale
        self.y_scale = sysid.y_scale
        self.u_order = config.STACK_u_order
        self.y_order = config.STACK_y_order
        self.o_dim = config.STACK_o_dim
        self.o_min = config.STACK_o_min
        self.o_max = config.STACK_o_max

    def estimate(self, x, u, y, *args):
        # descaled in, descaled out
        # x = [y, y, ..., y, u, u, ..., u, o, ..., o]
        u_dim = self.sysid.plant.u_dim
        y_dim = self.sysid.plant.y_dim
        u_idx = u_dim*self.u_order
        y_idx = y_dim*self.y_order
        scaled_u = (u - self.u_bias)/self.u_scale
        scaled_y = (y - self.y_bias)/self.y_scale
        if self.u_order == 0:
            x_est_now = np.hstack((scaled_y, x[0:y_idx - y_dim]))
        else:
            x_est_now = np.hstack((scaled_y, x[0:y_idx - y_dim], scaled_u, x[y_idx:y_idx + u_idx - u_dim]))
        if self.o_dim > 0:
            scaled_o = ut.zero_mean_scale(*args, self.o_min, self.o_max)
            x_est_now = np.hstack((x_est_now, scaled_o))
        return x_est_now
