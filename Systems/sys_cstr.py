"""
Firstly written by Tae Hoon Oh 2022.05 (oh.taehoon.4i@kyoto-u.ac.jp)
CSTR dynamics [van der Vusse reaction] refers to
"Nonlinear predictive control of a benchmark CSTR (1995)"

4 states (x): Concentrations of A (x1) and B (x2) [mol / L],
              Temperatures of reactor (x3) and cooling jacket (x4) [degree]
2 actions (u): Normalized flow rate (u1) [1 / h] and Heat removal (u2) [kJ / h]
2 outputs (y): Concentration of B (y1) [mol / L] and Temperature of reactor (y2) [degree]
1 parameters (p): Temperature of feed (p1) [degree]

Abbreviations
x = state, u = input, y = output, p = parameter, ref = reference
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
con = concentration / temp = temperature / var = variance
"""

import numpy as np
import casadi as ca
from Utility import utility as ut


class SysCSTR(object):
    def __init__(self, *args):
        self.process_type = 'continuous'
        self.np_data_type = np.float64

        if args:
            self.config = args[0]
            self.seed = self.config.seed
            self.state_disturb = self.config.system_state_disturb
            self.measure_disturb = self.config.system_measure_disturb
            self.para_disturb = self.config.system_para_disturb

        else:
            print('No config, use implemented setting')
            self.seed = 12345
            self.state_disturb = False
            self.measure_disturb = True
            self.para_disturb = True

        self.x_dim = 4
        self.u_dim = 2
        self.y_dim = 2
        self.p_dim = 1
        self.r_dim = 2

        self.state_std = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.measure_std = np.zeros(self.y_dim, dtype=self.np_data_type)
        self.para_std = np.zeros(self.p_dim, dtype=self.np_data_type)
        if self.state_disturb:
            self.state_std = np.array([0.001, 0.001, 0.001, 0.001])
        if self.measure_disturb:
            self.measure_std = np.array([0.003, 0.05])
        if self.para_disturb:
            self.para_std = np.array([0.05])

        self.time_interval = self.np_data_type(1/60)  # hour

        self.x_min = np.array([0, 0, 80., 80.], dtype=self.np_data_type)
        self.x_max = np.array([3., 3., 130., 130.], dtype=self.np_data_type)
        self.u_min = np.array([2, -9000.], dtype=self.np_data_type)
        self.u_max = np.array([35, 0.], dtype=self.np_data_type)
        self.y_min = np.array([0., 80.], dtype=self.np_data_type)
        self.y_max = np.array([3., 130.], dtype=self.np_data_type)
        self.p_min = np.array([100], dtype=self.np_data_type)
        self.p_max = np.array([110], dtype=self.np_data_type)

        #  y = ax + b, a = 2./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 2./(self.x_max - self.x_min)

        # initial value
        self.ini_x = np.array([2.14, 1.09, 114.2, 112.9], dtype=self.np_data_type)
        self.ini_u = np.array([14.19, -1113.5], dtype=self.np_data_type)
        self.ini_y = np.array([1.09, 114.2], dtype=self.np_data_type)
        self.ini_p = np.array([105], dtype=self.np_data_type)

        # steady-state_value
        self.ss_x = np.array([2.14, 1.09, 114.2, 112.9], dtype=self.np_data_type)
        self.ss_u = np.array([14.19, -1113.5], dtype=self.np_data_type)
        self.ss_y = np.array([1.09, 114.2], dtype=self.np_data_type)
        self.ss_p = np.array([105], dtype=self.np_data_type)

        # reference
        self.ref1 = np.array([1.09, 114.2], dtype=self.np_data_type)
        self.ref2 = np.array([1.04, 94.2], dtype=self.np_data_type)

        # For feed temperature change
        self.p_now = self.ini_p

        # Fixed parameters
        self.ca0 = 5.10  # mol/L
        self.k10 = 1.287*10**12  # 1/h +- 0.04
        self.k20 = 1.287*10**12  # 1/h +- 0.04
        self.k30 = 9.043*10**9   # 1/(molA*h) +- 0.27
        self.E1 = -9758.3  # K
        self.E2 = -9758.3  # K
        self.E3 = -8560    # K
        self.Hab = 4.2      # kJ/molA  +- 2.36
        self.Hbc = -11.     # kJ/molA  +- 1.92
        self.Had = -41.85   # kJ/molA  +- 1.41
        self.rho_Cp = 2.8119  # kJ/(L*K) +- 0.000016
        self.kw_AR = 866.88  # kJ/(h*K)  +- 25.8
        self.VR = 10  # L
        self.mk = 5.0  # kg
        self.Cpk = 2.0  # kJ/(kg*K) +- 0.05

        # Make step function for ode integration
        self.step_fcn = self._make_step_function()

    def go_step(self, x, u):
        self.p_now, p_bdd = self._disturbance_generation(self.p_now)
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = ut.zero_mean_scale(u, self.u_min, self.u_max)
        scaled_p = ut.zero_mean_scale(p_bdd, self.p_min, self.p_max)
        scaled_up = np.hstack((scaled_u, scaled_p))

        result = self.step_fcn(x0=scaled_x, p=scaled_up)
        scaled_next_x = np.squeeze(np.array(result['xf']))

        next_x = ut.zero_mean_descale(scaled_next_x, self.x_min, self.x_max)
        return next_x

    def get_observation(self, x):
        y = x[1:3] + np.random.normal(0, self.measure_std)
        return y

    def get_cost(self, y, u, ref):
        y_weight = np.diag([1, 0.01*0.01])
        y_cost = (y - ref).T @ y_weight @ (y - ref)
        u_cost = 0.01*0.01*0.1*(u[0] + np.abs(0.001*u[1]))  # Linear cost
        cost = y_cost + u_cost
        return cost

    def do_reset(self):
        self.p_now = self.np_data_type(105.)
        return self.ini_x, self.ini_u, self.ini_y, self.ini_p, self.ref1

    def get_steady_state(self):
        return self.ss_x, self.ss_u, self.ss_y, self.ss_p, self.ref1

    def get_feed_temperature(self):
        return self.p_now

    def _system_dynamics(self, x, u, p):
        x1, x2, x3, x4 = ca.vertsplit(x)
        u1, u2 = ca.vertsplit(u)
        p1 = p

        k1 = self.k10*np.exp(self.E1/(x3 + 273.15))
        k2 = self.k20*np.exp(self.E2/(x3 + 273.15))
        k3 = self.k30*np.exp(self.E3/(x3 + 273.15))

        heat_by_rxns = k1*x1*self.Hab + k2*x2*self.Hbc + k3*(x1**2)*self.Had

        x1dot = u1*(self.ca0 - x1) - k1*x1 - k3*(x1**2)
        x2dot = -u1*x2 + k1*x1 - k2*x2
        x3dot = u1*(p1 - x3) + self.kw_AR*(x4 - x3)/(self.rho_Cp*self.VR) - heat_by_rxns/self.rho_Cp
        x4dot = (u2 + self.kw_AR*(x3 - x4))/(self.mk*self.Cpk)
        xdot = ca.vcat([x1dot, x2dot, x3dot, x4dot])
        return xdot

    def _make_step_function(self):
        # Solved by casadi interface
        # scaled in - scaled out function
        x_ca = ca.SX.sym('x', self.x_dim)
        u_ca = ca.SX.sym('u', self.u_dim)
        p_ca = ca.SX.sym('p', self.p_dim)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = ut.zero_mean_descale(x_ca, self.x_min, self.x_max)
        u_d = ut.zero_mean_descale(u_ca, self.u_min, self.u_max)
        p_d = ut.zero_mean_descale(p_ca, self.p_min, self.p_max)

        # Integrating ODE with Casadi with solver cvodes
        xdot = self._system_dynamics(x_d, u_d, p_d)
        xdot = np.multiply(xdot, self.scale_grad)  # Because of scaling
        ode = {'x': x_ca, 'p': up_ca, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        ode_integrator = ca.integrator('Integrator', 'cvodes', ode, options)
        return ode_integrator

    def _disturbance_generation(self, p):
        p_next = p - (p - self.ss_p)*self.time_interval + np.random.normal(0, self.para_std, 1)
        p_bdd = np.clip(p_next, self.p_min, self.p_max)
        return p_next, p_bdd

