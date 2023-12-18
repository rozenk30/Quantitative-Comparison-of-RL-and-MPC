####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################
####################################### REFAIR REQUIRED ##########################################

import numpy as np


class PID(object):
    def __init__(self, p_gain, i_gain, d_gain, bias, reverse):
        self.decimal_point = 8  # ###
        self.error_prior = 0.
        self.integral_prior = 0.
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.bias = bias
        self.reverse = reverse

    def control(self, error, time_interval):
        if self.reverse:
            error = -error
        integral = self.integral_prior + error*time_interval
        derivative = (error - self.error_prior)/time_interval
        self.error_prior = error
        self.integral_prior = integral
        u = self.p_gain*error + self.i_gain*integral + self.d_gain*derivative + self.bias
        u = np.round(u, self.decimal_point)
        return u

    def reset(self):
        self.error_prior = 0.
        self.integral_prior = 0.
