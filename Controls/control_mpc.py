"""
Firstly written by Tae Hoon Oh 2022.07 (oh.taehoon.4i@kyoto-u.ac.jp)
Model predictive control
model: x+ = f(x, u)
constraint: g(x, u) <= 0
cost: J(x, u) = c
Initial_control: u_ini = u0(x)
Optimization runs by CASADI
"""
import pickle
import casadi as ca
import numpy as np
from pathlib import Path
from Utility import utility as ut


class MPC(object):
    def __init__(self, sysid, config):
        self.config = config
        self.sysid = sysid

        self.dynamic_model = self.sysid.dynamic_model
        self.observe_model = self.sysid.observe_model
        self.abstract_stage_cost = self.config.mpc_abstract_stage_cost
        self.abstract_terminal_cost = self.config.mpc_abstract_terminal_cost
        self.abstract_stage_constraint = self.config.mpc_abstract_stage_constraint
        self.abstract_terminal_constraint = self.config.mpc_abstract_terminal_constraint
        self.initial_control = self.config.mpc_initial_control

        self.x_dim = sysid.x_est_dim
        self.x_min = sysid.x_est_min
        self.x_max = sysid.x_est_max

        self.u_dim = sysid.plant.u_dim
        self.u_min = sysid.plant.u_min
        self.u_max = sysid.plant.u_max

        self.g_dim = config.MPC_g_dim
        self.h_dim = config.MPC_h_dim

        self.prediction_horizon = config.MPC_prediction_horizon
        self.scale_grad = 2./(self.x_max - self.x_min)
        self.u_guess = np.zeros((self.u_dim, self.prediction_horizon + 1))

    def control(self, current_state):
        # Descaled in, descaled out
        ini_x, ini_u = self._guess_generation(current_state)

        ca_x = ca.SX.sym('x', self.x_dim)
        ca_u = ca.SX.sym('u', self.u_dim)
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
        terminal_cost = self._terminal_cost(ca_x_d, ca_u_d, for_casadi=True)
        stage_cost_func = ca.Function('L', [ca_x, ca_u], [stage_cost], ['x', 'u'], ['Lc'])
        terminal_cost_func = ca.Function('V', [ca_x, ca_u], [terminal_cost], ['x', 'u'], ['Vc'])

        # Start with an empty NLP
        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        cost = 0

        # x and u trajectories
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
            cost += stage_cost_func(xk, uk)

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
        cost += terminal_cost_func(xk, uk)

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
        opts = {'print_time': False, "ipopt": {'print_level': 0}}
        # opts = {'print_time': True, 'ipopt': {'print_level': 5}, 'ipopt': {"max_iter": 10000}}
        # opts = {'ipopt': {'tol': 1e-6}, 'ipopt': {'acceptable_tol': 1e-4}, 'ipopt': {"max_iter": 10000}}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        obj_cost = sol['f']
        constraint_value = sol['g']
        x_opt, u_opt = trajectories(sol['x'])

        # to numpy array
        x_opt, u_opt = x_opt.full(), u_opt.full()

        descale_x_opt = ut.zero_mean_descale(np.transpose(x_opt), self.x_min, self.x_max)  # matrix
        descale_u_opt = ut.zero_mean_descale(np.transpose(u_opt), self.u_min, self.u_max)  # matrix
        control_value = descale_u_opt[0, :]
        return control_value

    def save_controller(self, directory, name):
        control_parameters = [self.sysid, self.config]
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'wb') as handle:
            pickle.dump(control_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_controller(self, directory, name):
        with open(Path.joinpath(directory, name + '-controller_parameters.pickle'), 'rb') as handle:
            control_parameters = pickle.load(handle)
        self.__init__(control_parameters[0], control_parameters[1])

    def _stage_cost(self, x, u, for_casadi):
        return self.abstract_stage_cost(x, u, for_casadi, observe_model=self.observe_model)

    def _terminal_cost(self, x, u, for_casadi):
        return self.abstract_terminal_cost(x, u, for_casadi, observe_model=self.observe_model)

    def _stage_constraint(self, x, u, for_casadi):
        return self.abstract_stage_constraint(x, u, for_casadi, observe_model=self.observe_model)

    def _terminal_constraint(self, x, u, for_casadi):
        return self.abstract_terminal_constraint(x, u, for_casadi, observe_model=self.observe_model)

    def _guess_generation(self, current_state):
        guess_x = np.zeros((self.x_dim, self.prediction_horizon + 1))  # Trajectory
        guess_u = np.zeros((self.u_dim, self.prediction_horizon + 1))  # Trajectory
        guess_x[:, 0] = current_state
        for k in range(self.prediction_horizon):
            guess_u[:, k] = self.initial_control(guess_x[:, k])
            guess_u[:, k] = np.clip(guess_u[:, k], self.u_min, self.u_max)
            guess_x[:, k+1] = self.dynamic_model(guess_x[:, k], guess_u[:, k], for_casadi=False)  # Calculate by numpy
            guess_x[:, k+1] = np.clip(guess_x[:, k+1], self.x_min, self.x_max)
        guess_u[:, -1] = self.initial_control(guess_x[:, -1])
        return guess_x, guess_u



