"""
Firstly written by Tae Hoon Oh 2021.04 (oh.taehoon.4i@kyoto-u.ac.jp)
Penicillin product fed-batch bioreactor dynamics refers to
"The development of an industrial-scale fed-batch fermentation simulation (2015)"

# 26 states : Time, A_0, A_1, A_3, A_4, Integral_X, S, P, V, T, H, n0 - n9 nm, DO, viscosity, phi0
# 6 action :  F_S, F_oil, F_a, F_b, F_c, F_h
# 5 outputs : Time, X (A_0 + A_1 + A_3 + A_4), S, P, V
# We fixed the RPM and F_dis.

*** 'sys_penicillin' simulation with PID controllers  ***

Abbreviations
x = state, u = input, y = output, p = parameter, ref = reference
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
con = concentration / temp = temperature / var = variance
"""
import pathlib
import numpy as np
from Controls import control_pid as pid
from Systems import sys_peni


class SysPeniWithPID(object):
    def __init__(self, *args):
        self.process_type = 'batch'
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

        # Define the plant
        self.plant = sys_peni.SysPenicillin(*args)

        self.x_dim = self.plant.x_dim
        self.u_dim = 2
        self.y_dim = self.plant.y_dim
        self.p_dim = 0

        self.state_std = self.plant.state_std
        self.measure_std = self.plant.measure_std
        self.para_std = self.plant.para_std

        self.number_of_inner_steps = 10
        self.time_interval = self.np_data_type(self.plant.time_interval * self.number_of_inner_steps)  # hour
        self.terminal_time = self.plant.terminal_time
        self.total_horizon = int(self.terminal_time/self.time_interval)

        self.x_min = self.plant.x_min
        self.x_max = self.plant.x_max
        self.u_min = self.plant.u_min[0:2]
        self.u_max = self.plant.u_max[0:2]
        self.y_min = self.plant.y_min
        self.y_max = self.plant.y_max
        self.p_min = self.plant.p_min
        self.p_max = self.plant.p_max

        #  y = ax + b, a = 2./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 2./(self.x_max - self.x_min)

        # initial value
        self.ini_x = self.plant.ini_x
        self.ini_u = self.plant.ini_u[0:2]
        self.ini_y = self.plant.ini_y
        self.ini_p = self.plant.ini_p

        # Define the target value for PID
        self.temp_ref = 298.
        self.pH_ref = 6.5

        # Define the PID controller
        self.PID_cooling = pid.PID(100, 30, 0, 100, False)  # 100 50 0 100
        self.PID_heating = pid.PID(5, 0, 0, 0, True)
        self.PID_acid = pid.PID(0.01, 0, 0, 0, False)
        self.PID_base = pid.PID(1500, 15, 0, 0, True)

        self.previous_action = np.array([8., 22., 0., 0., 0.0001, 200.], dtype=self.np_data_type)

        self.current_path = pathlib.Path.cwd()  # ### may not work for different directory...
        self.reference_input_path = self.current_path.joinpath('Utility', 'peni_reference_input.txt')  ###
        self.reference_input = np.loadtxt(self.reference_input_path, dtype=self.np_data_type)

        # Make step function for ode integration
        self.step_fcn = self.plant.step_fcn

    def go_step(self, x, u):
        for step in range(self.number_of_inner_steps):
            next_x = self._go_one_step(x, u)
            x = next_x
        return x

    def get_observation(self, x):
        y = self.plant.get_observation(x)
        return y

    def do_reset(self):
        self.PID_cooling = pid.PID(100, 50, 0, 100, False)
        self.PID_heating = pid.PID(5, 0, 0, 0, True)
        self.PID_cooling.reset()
        self.PID_heating.reset()
        self.PID_base.reset()
        self.PID_acid.reset()
        x0, u0, y0, p0 = self.plant.do_reset()
        return x0, u0[0:2], y0, p0

    def get_cost(self, y, u):
        # Cost parameters
        if np.round(y[0], 3) < self.plant.terminal_time:
            return self._compute_path_cost(y, u)
        else:  # Terminal cost
            return self._compute_terminal_cost(y, u)

    def _go_one_step(self, x, u):
        pid_u = self._compute_pid_input(x)
        u_including_pid = np.zeros(self.u_dim + 4, dtype=self.np_data_type)
        u_including_pid[0:2] = u
        u_including_pid[2:] = pid_u
        next_x = self.plant.go_step(x, u_including_pid)
        next_x = np.squeeze(next_x)
        self.previous_action = u_including_pid
        ''' To prevent the numerical issue '''
        for k in range(len(next_x)):
            if next_x[k] < 0:
                next_x[k] = 2*1e-5  # ### 2*1e-5
        if next_x[1] < 0.1:
            next_x[1] = 0.1  # ###
        return next_x

    def _compute_path_cost(self, y, u):
        # cost based on the descaled value
        scaled_fs = u[0]/200
        scaled_foil = u[1]/200
        scaled_vol = y[4]/100000
        p1, p2, p3 = 2*0.00090, 2*0.00150, 2*0.00200
        path_cost = p1*scaled_fs + p2*scaled_foil + p3*scaled_vol**2
        return path_cost

    def _compute_terminal_cost(self, y, u):
        # cost based on the descaled value
        scaled_product = y[3]/50
        scaled_vol = y[4]/100000
        p1 = 0.5
        terminal_cost = p1*(1 - scaled_product*scaled_vol)
        return terminal_cost

    def get_reference_input(self, step_index):
        # descaled output
        return self.reference_input[:, step_index]

    def _compute_pid_input(self, x):
        if x[0] > 80:  # pid gain scheduling
            self.PID_cooling = pid.PID(10000, 0, 0, 0, False)
            self.PID_heating = pid.PID(100, 1000, 0, 0, True)

        temp, hydro_ion = x[9], x[10]
        _, _, f_a_p, f_b_p, f_c_p, f_h_p = self.previous_action
        _, _, f_a_l, f_b_l, f_c_l, f_h_l = self.plant.u_min
        _, _, f_a_u, f_b_u, f_c_u, f_h_u = self.plant.u_max

        # PID
        if temp > self.temp_ref - 0.03:
            f_c = self.PID_cooling.control(temp - self.temp_ref, self.plant.time_interval)
            f_h = f_h_l
        elif temp < self.temp_ref - 0.03:
            f_c = f_c_l
            f_h = self.PID_heating.control(temp - self.temp_ref, self.plant.time_interval)
        else:
            f_c = f_c_p
            f_h = f_h_p

        if -np.log10(hydro_ion) > self.pH_ref + 0.03:
            f_a = self.PID_acid.control(-np.log10(hydro_ion) - self.pH_ref, self.plant.time_interval)
            f_b = f_b_l
        elif -np.log10(hydro_ion) < self.pH_ref:
            f_a = f_a_l
            f_b = self.PID_base.control(-np.log10(hydro_ion) - self.pH_ref, self.plant.time_interval)
        else:
            f_a = f_a_p
            f_b = f_b_p
        pid_action = np.array([f_a, f_b, f_c, f_h], dtype=self.np_data_type)
        return pid_action
