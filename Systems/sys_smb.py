"""
Firstly written by Tae Hoon Oh 2021.05 (oh.taehoon.4i@kyoto-u.ac.jp)
SMB model refers to
"Transition model for simulated moving bed under nonideal conditions (2019)"
"Automatic control of simulated moving bed process with deep Q-network (2021)"
"Simulated Moving Bed Chromatography for the Separation of Enantiomers (2009)"

# states (1614): Concentrations, time step, mode, previous action, concentration of extact and raffiante tanks
# actions (8): velocity of 8 sections
# outputs (10): concentration of 4 ports + purities of extract and raffinate tanks

Abbreviations
x = state, u = input, y = output, p = parameter
dim = dimension / ini = initial / para = parameter / grad = gradient
con = concentration / temp = temperature / var = variance / coeff = coefficient / equili = equilibrium
const = constant /
"""


import numpy as np
import casadi as ca
from Utility import utility as ut


class SysSMB(object):
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
            self.measure_disturb = False
            self.para_disturb = True

        self.grid_num = 50  # Number of grid for a single column
        self.column_num = 8

        self.x_dim = 4*self.column_num*self.grid_num + 1 + 1  # Switch mode indicator
        self.u_dim = 4
        self.y_dim = 4
        self.p_dim = 2
        self.r_dim = 2

        self.state_std = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.measure_std = np.zeros(self.y_dim, dtype=self.np_data_type)
        self.para_std = np.zeros(self.p_dim, dtype=self.np_data_type)
        if self.state_disturb:
            self.state_std = np.zeros(self.x_dim, dtype=self.np_data_type)
        if self.measure_disturb:
            self.measure_std = np.zeros(self.y_dim, dtype=self.np_data_type)
        if self.para_disturb:
            self.para_std = np.array([0.01, 0.01], dtype=self.np_data_type)

        self.x_min = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.x_max = np.ones(self.x_dim, dtype=self.np_data_type)
        self.u_min = np.array([0.0100, 0.0100, 0.0135, 0.0135], dtype=self.np_data_type)
        self.u_max = np.array([0.0135, 0.0135, 0.0165, 0.0165], dtype=self.np_data_type)
        self.y_min = np.zeros(self.y_dim, dtype=self.np_data_type)
        self.y_max = np.ones(self.y_dim, dtype=self.np_data_type)
        self.p_min = np.array([0.9, 0.9], dtype=self.np_data_type)
        self.p_max = np.array([1.1, 1.1], dtype=self.np_data_type)

        #  y = ax + b, a = 1./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad_for_a_column = 2. / (self.x_max[0:4*self.grid_num] - self.x_min[0:4*self.grid_num])

        # Initial value
        try:
            self.ini_x = np.loadtxt('Utility/' + 'SMB_initial_state.txt')  # ### may not work for different directory...
        except FileNotFoundError and OSError:
            self.ini_x = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.ini_u = np.array([0.013, 0.013, 0.014, 0.014], dtype=self.np_data_type)
        self.ini_y = np.array([0., 0., 0., 0.], dtype=self.np_data_type)
        self.ini_p = np.array([1.0, 1.0], dtype=self.np_data_type)

        # steady-state_value
        try:
            self.ss_x = np.loadtxt('Utility/' + 'SMB_initial_state.txt')
            self.ss_y = np.array([0.00006034, 0.00000000, 0.00000000, 0.00708570], dtype=self.np_data_type)
        except FileNotFoundError and OSError:
            self.ss_x = np.zeros(self.x_dim, dtype=self.np_data_type)
            self.ss_y = np.array([0., 0., 0., 0.], dtype=self.np_data_type)
        self.ss_u = np.array([0.013, 0.013, 0.014, 0.014], dtype=self.np_data_type)
        self.ss_p = np.array([1.0, 1.0], dtype=self.np_data_type)

        # reference
        self.ref = np.array([0.99, 0.99], dtype=self.np_data_type)

        # For feed concentration change
        self.p_now = self.ini_p

        # Tank information
        self.extract_tank = np.array([0.0001, 0.], dtype=self.np_data_type)
        self.raffinate_tank = np.array([0., 0.01], dtype=self.np_data_type)

        # Fixed parameters of column [Abbreviation]
        self.length = 1.  # m [l]
        self.diameter = 0.1  # m [Dia]
        self.porosity = 0.66  # [e]
        self.diffusion_coeff = 1e-5  # [D]
        # H = 2.25, 0.8 corresponding v = 0.0180 0.0122, del 0.1 is 0.0088
        # Henry constant = self.equili_const * self.qm
        self.equili_const = [0.5, 0.2]  # [K]
        self.langmuir_coeff = [5, 5]  # [qm]
        self.mass_transfer_coeff = [2, 2]  # [k]
        self.switch_time = 120  # sec

        # Additional values
        self.area = np.pi*self.diameter*self.diameter/4  # [A]
        self.volume = self.area * self.length  # [V]
        self.grid_length = self.np_data_type(self.length / self.grid_num)  # delz
        self.time_grid_num = 30
        self.time_interval = self.np_data_type(self.switch_time/4/self.time_grid_num)  # sec
        self.section1_flow_rate = 0.022
        self.section4_flow_rate = 0.010

        # Make step function for ode integration
        self.step_fcn = self._make_step_function()

    def go_step(self, x, u):
        self.p_now, p_bdd = self._disturbance_generation(self.p_now)
        feed_con = p_bdd
        u = self._total_inputs(u)
        N = self.grid_num
        mode = int(x[4*self.column_num*N]*10)
        time_step_index = x[4*self.column_num*N + 1]

        role_index = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
        role_index = np.concatenate((role_index[8 - mode:8], role_index[0:8 - mode]))

        s1, s2, s3, s4 = x[0 * N:4 * N], x[4 * N:8 * N], x[8 * N:12 * N], x[12 * N:16 * N]
        s5, s6, s7, s8 = x[16 * N:20 * N], x[20 * N:24 * N], x[24 * N:28 * N], x[28 * N:32 * N]

        for z in range(self.time_grid_num):
            s1_in = np.array([s8[N - 1], s8[3*N - 1]], dtype=self.np_data_type)
            s2_in = np.array([s1[N - 1], s1[3*N - 1]], dtype=self.np_data_type)
            s3_in = np.array([s2[N - 1], s2[3*N - 1]], dtype=self.np_data_type)
            s4_in = np.array([s3[N - 1], s3[3*N - 1]], dtype=self.np_data_type)
            s5_in = np.array([s4[N - 1], s4[3*N - 1]], dtype=self.np_data_type)
            s6_in = np.array([s5[N - 1], s5[3*N - 1]], dtype=self.np_data_type)
            s7_in = np.array([s6[N - 1], s6[3*N - 1]], dtype=self.np_data_type)
            s8_in = np.array([s7[N - 1], s7[3*N - 1]], dtype=self.np_data_type)

            con_in = np.array([s1_in, s2_in, s3_in, s4_in, s5_in, s6_in, s7_in, s8_in], dtype=self.np_data_type)
            con_in[np.where(role_index == 0)] = con_in[np.where(role_index == 0)] * u[7] / u[0]
            con_in[np.where(role_index == 1)] = con_in[np.where(role_index == 1)] * u[0] / u[1]
            con_in[np.where(role_index == 2)] = con_in[np.where(role_index == 2)]
            con_in[np.where(role_index == 3)] = con_in[np.where(role_index == 3)] * u[2] / u[3]
            con_in[np.where(role_index == 4)] = (con_in[np.where(role_index == 4)] * u[3] + feed_con*(u[4] - u[3]))/u[4]
            con_in[np.where(role_index == 5)] = con_in[np.where(role_index == 5)] * u[4] / u[5]
            con_in[np.where(role_index == 6)] = con_in[np.where(role_index == 6)]
            con_in[np.where(role_index == 7)] = con_in[np.where(role_index == 7)] * u[6] / u[7]

            next_con1, s1_out = self._column_step(s1, con_in[0], u[role_index[0]], [])
            next_con2, s2_out = self._column_step(s2, con_in[1], u[role_index[1]], [])
            next_con3, s3_out = self._column_step(s3, con_in[2], u[role_index[2]], [])
            next_con4, s4_out = self._column_step(s4, con_in[3], u[role_index[3]], [])
            next_con5, s5_out = self._column_step(s5, con_in[4], u[role_index[4]], [])
            next_con6, s6_out = self._column_step(s6, con_in[5], u[role_index[5]], [])
            next_con7, s7_out = self._column_step(s7, con_in[6], u[role_index[6]], [])
            next_con8, s8_out = self._column_step(s8, con_in[7], u[role_index[7]], [])
            s_out = np.array([s1_out, s2_out, s3_out, s4_out, s5_out, s6_out, s7_out, s8_out], dtype=self.np_data_type)
            self.extract_tank += s_out[np.where(role_index == 1)][0]*u[1]
            self.raffinate_tank += s_out[np.where(role_index == 5)][0]*u[5]
            # * self.area is omitted for numerical reason
            s1, s2, s3, s4 = next_con1, next_con2, next_con3, next_con4
            s5, s6, s7, s8 = next_con5, next_con6, next_con7, next_con8

        time_step_index += 0.25
        if time_step_index > 0.999:
            time_step_index = 0.
            mode += 1

        if mode == 8:
            mode = 0
        next_x = np.concatenate((next_con1, next_con2, next_con3, next_con4, next_con5, next_con6, next_con7, next_con8,
                                 [0.1*mode], [time_step_index]))
        return next_x

    def get_observation(self, x, u):
        u = self._total_inputs(u)
        y = np.zeros(self.y_dim, dtype=self.np_data_type)
        N = self.grid_num
        mode = int(x[4*self.column_num*N]*10) - 1
        if mode < 0:
            mode = 7
        role_index = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
        role_index = np.concatenate((role_index[8 - mode:8], role_index[0:8 - mode]))
        s1 = np.array([x[0*N + N - 1], x[0*N + 3*N - 1]]).squeeze()
        s2 = np.array([x[4*N + N - 1], x[4*N + 3*N - 1]]).squeeze()
        s3 = np.array([x[8*N + N - 1], x[8*N + 3*N - 1]]).squeeze()
        s4 = np.array([x[12*N + N - 1], x[12*N + 3*N - 1]]).squeeze()
        s5 = np.array([x[16*N + N - 1], x[16*N + 3*N - 1]]).squeeze()
        s6 = np.array([x[20*N + N - 1], x[20*N + 3*N - 1]]).squeeze()
        s7 = np.array([x[24*N + N - 1], x[24*N + 3*N - 1]]).squeeze()
        s8 = np.array([x[28*N + N - 1], x[28*N + 3*N - 1]]).squeeze()
        s_out = np.array([s1, s2, s3, s4, s5, s6, s7, s8])
        y[0:2] = s_out[np.where(role_index == 1)][0]*u[1] + self.measure_std[0:2]*np.random.normal(0, 1, 2)
        y[2:4] = s_out[np.where(role_index == 5)][0]*u[5] + self.measure_std[2:4]*np.random.normal(0, 1, 2)
        # * self.area is omitted for numerical reason
        return y

    def get_mode(self, x):
        mode = int(x[4*self.column_num*self.grid_num]*10)
        return mode

    def get_cost(self, y, u, ref):
        # Cost related parameters
        product = -1.
        extract_purity_penalty = 0.03
        raffinate_purity_penalty = 0.03

        puri_e = y[0] / (y[0] + y[1] + 10**(-10))  # for numerical reason
        puri_r = y[3] / (y[2] + y[3] + 10**(-10))  # for numerical reason

        # penalty_extract, penalty_raffinate = 0, 0
        # if puri_e < ref[0]:
        #     penalty_extract = extract_purity_penalty*(ref[0] - puri_e)
        # if puri_r < ref[1]:
        #     penalty_raffinate = raffinate_purity_penalty*(ref[1] - puri_r)

        penalty_extract = extract_purity_penalty * (self.ref[0] - puri_e)
        penalty_raffinate = raffinate_purity_penalty * (self.ref[1] - puri_r)
        c = 0.002 + product*(u[2] - u[1]) + penalty_extract + penalty_raffinate
        return c

    def do_reset(self):
        self.p_now = np.array([1.0, 1.0], dtype=self.np_data_type)
        self.extract_tank = np.array([0.0001, 0.], dtype=self.np_data_type)
        self.raffinate_tank = np.array([0., 0.01], dtype=self.np_data_type)
        return self.ss_x, self.ss_u, self.ss_y, self.ss_p, self.ref

    def get_steady_state(self):
        return self.ss_x, self.ss_u, self.ss_y, self.ss_p, self.ref

    def get_feed_concentration(self):
        return self.p_now

    def get_tank_information(self):
        return self.extract_tank, self.raffinate_tank

    def _make_step_function(self):
        # scaled in - scaled out function
        # step function for a column
        x_ca = ca.SX.sym('x', 4*self.grid_num)
        u_ca = ca.SX.sym('u', 1)
        p_ca = ca.SX.sym('p', 0)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = ut.zero_mean_descale(x_ca, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        u_d = ut.zero_mean_descale(u_ca, self.u_min[0], self.u_max[0])
        p_d = ut.zero_mean_descale(p_ca, np.array([]), np.array([]))

        # Integrating ODE with Casadi with solver cvodes
        xdot = self._column_dynamics(x_d, u_d, p_d)
        xdot = np.multiply(xdot, self.scale_grad_for_a_column)  # Because of scaling, but it is 1 in this case
        ode = {'x': x_ca, 'p': up_ca, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        column_ode_integrator = ca.integrator('Integrator', 'cvodes', ode, options)
        return column_ode_integrator

    def _column_dynamics(self, x, u, p):
        # Finite Difference Method
        y, v = x, u
        N, D, delz, e = self.grid_num, self.diffusion_coeff, self.grid_length, self.porosity
        K, k, qm = self.equili_const, self.mass_transfer_coeff, self.langmuir_coeff
        ee = (1 - e)/e
        xdot = ca.SX.zeros(4*N, 1)

        # Boundary condition
        xdot[0*N] = 0  #(v**2/self.D)*(c0[0] - y[0]) + self.D*(y[1] - y[0])/(self.delz**2)
        xdot[1*N-1] = D*(y[N-1] - 2*y[N-1] + y[N-2])/(delz**2) - v*(y[N-1] - y[N-2])/(delz) \
                     - ee*k[0]*(qm[0]*K[0]*y[N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[2*N-1])
        xdot[1*N] = k[0]*(qm[0]*K[0]*y[0]/(1+K[0]*y[0]+K[1]*y[2*N]) - y[N])
        xdot[2*N-1] = k[0]*(qm[0]*K[0]*y[N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[2*N-1])
        xdot[2*N] = 0  # (v**2/self.D)*(c0[1] - y[2*self.N]) + self.D*(y[2*self.N + 1] - y[2*self.N])/(self.delz**2)
        xdot[3*N-1] = D*(y[3*N-1] - 2*y[3*N-1] + y[3*N-2])/(delz**2) - v*(y[3*N-1] - y[3*N-2])/(delz) \
                     - ee*k[1]*(qm[1]*K[1]*y[3*N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[4*N-1])
        xdot[3*N] = k[1]*(qm[1]*K[1]*y[2*N]/(1+K[0]*y[0]+K[1]*y[2*N]) - y[3*N])
        xdot[4*N-1] = k[1]*(qm[1]*K[1]*y[3*N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[4*N-1])

        # Internal dynamics
        for i in range(N-2):
            xdot[0*N+i+1] = D*(y[i+2] - 2*y[i+1] + y[i])/(delz**2) - v*(y[i+1] - y[i])/delz \
                           - ee*k[0]*(qm[0]*K[0]*y[i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[N+i+1])
            xdot[1*N+i+1] = k[0]*(qm[0]*K[0]*y[i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[N+i+1])
            xdot[2*N+i+1] = D*(y[2*N+i+2] - 2*y[2*N+i+1] + y[2*N+i])/(delz**2) - v*(y[2*N+i+1] - y[2*N+i])/delz \
                           - ee*k[1]*(qm[1]*K[1]*y[2*N+i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[3*N+i+1])
            xdot[3*N+i+1] = k[1]*(qm[1]*K[1]*y[2*N+i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[3*N+i+1])
        return xdot

    def _column_step(self, x, c_in, u, p):

        x[0] = c_in[0]
        x[2*self.grid_num] = c_in[1]

        scaled_x = ut.zero_mean_scale(x, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        scaled_u = ut.zero_mean_scale(u, self.u_min[0], self.u_max[0])
        scaled_p = ut.zero_mean_scale(p, np.array([]), np.array([]))
        scaled_up = np.hstack((scaled_u, scaled_p))

        result = self.step_fcn(x0=scaled_x, p=scaled_up)
        scaled_next_x = np.squeeze(np.array(result['xf']))

        next_x = ut.zero_mean_descale(scaled_next_x, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        terminal_con = np.array([next_x[self.grid_num - 1], next_x[3*self.grid_num - 1]])
        return next_x, terminal_con

    def _total_inputs(self, u):
        total_u = np.array([self.section1_flow_rate, self.section1_flow_rate, 0.013, 0.013,
                            0.014, 0.014, self.section4_flow_rate, self.section4_flow_rate], dtype=self.np_data_type)
        total_u[2:6] = u
        return total_u

    def _disturbance_generation(self, p):
        p_next = p - 0.1*(p - self.ss_p) + self.para_std*np.random.normal(0, 1, self.p_dim)
        p_bdd = np.clip(p_next, self.p_min, self.p_max)
        return p_next, p_bdd

