import numpy as np


class KalmanFilter(object):
    def __init__(self, sysid, config):
        self.sysid = sysid
        self.config = config

        self.u_bias = sysid.u_bias
        self.y_bias = sysid.y_bias
        self.u_scale = sysid.u_scale
        self.y_scale = sysid.y_scale

        self.a = sysid.sysid.a
        self.b = sysid.sysid.b
        self.c = sysid.sysid.c

        self.x_dim = self.a.shape[0]
        self.y_dim = self.c.shape[0]
        self.q = config.KF_process_noise_cov
        self.r = config.KF_measure_noise_cov
        self.p_est = config.KF_initial_cov

    def estimate(self, x_est, u, y, *args):
        # descaled in, descaled out
        scaled_u = (u - self.u_bias)/self.u_scale
        scaled_y = (y - self.y_bias)/self.y_scale
        x_pri, p_pri = self._predict(x_est, self.p_est, scaled_u)
        x_pos, p_pos = self._correct(x_pri, p_pri, scaled_y)
        self.p_est = p_pos
        return x_pos

    def _predict(self, x, p, u):
        x_pri = self.a @ x + self.b @ u
        p_pri = self.a @ p @ self.a.T + self.q
        p_pri = (p_pri + p_pri.T) / 2
        return x_pri, p_pri

    def _correct(self, x_pri, p_pri, y):
        gain = p_pri @ self.c.T @ np.linalg.inv(self.c @ p_pri @ self.c.T + self.r)
        x_pos = x_pri + gain @ (y - self.c @ x_pri)
        p_pos = (np.eye(self.x_dim) - gain@self.c)@p_pri@(np.eye(self.x_dim) - gain@self.c).T + gain@self.r@gain.T
        p_pos = (p_pos + p_pos.T) / 2
        return x_pos, p_pos
