import numpy as np
import casadi as ca

""" KF-style Linear filter, Note that this is not an optimal filter """


class LSTMEstimator(object):
    def __init__(self, sysid, config):
        self.sysid = sysid
        self.config = config

        self.u_bias = sysid.u_bias
        self.y_bias = sysid.y_bias
        self.u_scale = sysid.u_scale
        self.y_scale = sysid.y_scale

        self.dynamic_model = sysid.dynamic_model
        self.n_w = sysid.sysid.n_w
        self.n_b = sysid.sysid.n_b
        self.node_num = config.LSTM_node_num

        self.q = config.LSTM_KF_process_noise_cov
        self.r = config.LSTM_KF_measure_noise_cov
        self.p_est = config.LSTM_KF_initial_cov
        self.x_sym, self.u_sym, self.a_est = self._calculate_a()

    def estimate(self, x_est, u, y, *args):
        # descaled in, descaled out
        scaled_u = u
        scaled_y = (y - self.y_bias)/self.y_scale
        x_pri, p_pri = self._predict(x_est, self.p_est, scaled_u)
        x_pos, p_pos = self._correct(x_pri, p_pri, scaled_y)
        self.p_est = p_pos
        return x_pos

    def _predict(self, x_est, p, u):
        x_pri = self.dynamic_model(x_est, u, for_casadi=False)
        a_est = self.a_est(x_est, u).full()
        p_pri = a_est.T @ p @ a_est + self.q
        p_pri = (p_pri + p_pri.T)/2
        return x_pri, p_pri

    def _correct(self, x_pri, p_pri, y):
        x_dim = 2*self.node_num
        n_w_extend = np.hstack((self.n_w, np.zeros_like(self.n_w)))
        gain = p_pri @ n_w_extend.T @ np.linalg.inv(n_w_extend @ p_pri @ n_w_extend.T + self.r)
        x_pos = x_pri + gain @ (y - np.squeeze(self.n_b) - n_w_extend @ x_pri)
        p_pos = (np.eye(x_dim) - gain @ n_w_extend) @ p_pri @ (np.eye(x_dim) - gain @ n_w_extend).T \
                + gain @ self.r @ gain.T
        p_pos = (p_pos + p_pos.T) / 2
        return x_pos, p_pos

    def _calculate_a(self):
        x_sym = ca.SX.sym('x', 2*self.node_num)
        u_sym = ca.SX.sym('u', self.sysid.plant.u_dim)
        out = self.dynamic_model(x_sym, u_sym, for_casadi=True)
        dynamic_fcn = ca.Function('f', [x_sym, u_sym], [out])
        derivative_sym = ca.jacobian(dynamic_fcn(x_sym, u_sym), x_sym)
        derivative_fun = ca.Function('df', [x_sym, u_sym], [derivative_sym])
        return x_sym, u_sym, derivative_fun

