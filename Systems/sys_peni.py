"""
Firstly written by Tae Hoon Oh 2021.04 (oh.taehoon.4i@kyoto-u.ac.jp)
Penicillin product fed-batch bioreactor dynamics refers to
"The development of an industrial-scale fed-batch fermentation simulation (2015)"

25 states (x): Time, A_0, A_1, A_3, A_4, Int_X, S, P, V, T, H, n0 - n9 nm, DO, viscosity, phi0
6 actions (u): F_S, F_oil, F_a, F_b, F_c, F_h
5 outputs (y): Time, X (A_0 + A_1 + A_3 + A_4), S, P, V
We fixed the RPM and F_dis.

*** 'sys_peni_with_PID' is the one with PID controllers ***

Abbreviations
x = x, u = input, y = y, p = parameter, ref = reference
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
con = concentration / temp = temperature / var = variance
"""

import numpy as np
import casadi as ca
from Utility import utility as ut


class SysPenicillin(object):
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
            self.measure_disturb = True
            self.para_disturb = True

        self.x_dim = 25
        self.u_dim = 6
        self.y_dim = 5
        self.p_dim = 0  # 2 parameter changes for each batch operation

        self.state_std = np.zeros(self.x_dim, dtype=self.np_data_type)
        self.measure_std = np.zeros(self.y_dim, dtype=self.np_data_type)
        self.para_std = np.array([0., 0.], dtype=self.np_data_type)
        if self.state_disturb:
            self.state_std = np.zeros(self.x_dim, dtype=self.np_data_type)
        if self.measure_disturb:
            self.measure_std = np.zeros(self.y_dim, dtype=self.np_data_type)
        if self.para_disturb:
            self.para_std = np.array([0.003, 0.03], dtype=self.np_data_type)

        self.time_interval = self.np_data_type(0.2)  # hour

        self.terminal_time = self.np_data_type(230.)  # +-25 terminal time
        self.horizon_length = int(self.terminal_time / self.time_interval)

        self.x_min = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 273., 10 ** (-7), 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0.], dtype=self.np_data_type)
        self.x_max = np.array([230., 100., 100., 100., 100., 50000., 25., 100., 110000., 313., 10 ** (-6), 10 ** 15,
                               10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15, 10 ** 15,
                               10 ** 15, 10 ** 15, 20., 200., 100.], dtype=self.np_data_type)
        self.u_min = np.array([30., 15., 0., 0., 0., 0.], dtype=self.np_data_type)
        self.u_max = np.array([200., 45., 20., 1000., 5000., 5000.], dtype=self.np_data_type)
        self.y_min = np.array([0., 0., 0., 0., 0.], dtype=self.np_data_type)
        self.y_max = np.array([230., 400., 25., 100., 110000.], dtype=self.np_data_type)
        self.p_min = np.array([], dtype=self.np_data_type)
        self.p_max = np.array([], dtype=self.np_data_type)

        #  y = ax + b, a = 2./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 2. / (self.x_max - self.x_min)

        self.ini_x = np.array(
            [0., 0.5 / 3, 1 / 3, 0., 0., 0., 1., 0., 58000., 297., 10 ** (-6.5), 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 15., 4., 0.], dtype=self.np_data_type)
        self.ini_u = np.array([8., 22., 0., 0., 0.0001, 200.], dtype=self.np_data_type)
        self.ini_y = np.array([0., 1., 1., 0., 58000.], dtype=self.np_data_type)
        self.ini_p = np.array([], dtype=self.np_data_type)

        self.mu_p_perturb = self.para_std[0]*np.clip(np.random.normal(0, 1), -2, 2)
        self.mu_x_max_perturb = self.para_std[1]*np.clip(np.random.normal(0, 1), -2, 2)

        # System dynamics parameters (structured model)
        self.mu_p = 0.041 + self.mu_p_perturb
        self.mux_max = 0.41 + self.mu_x_max_perturb
        # print('para:', np.round(self.mu_p, 4), np.round(self.mux_max, 4))
        self.ratio_mu_e_mu_b = 0.4
        self.P_std_dev = 0.0015
        self.mean_P = 0.002
        self.mu_v = 0.000171
        self.mu_a = 0.0035
        self.mu_diff = 0.00536
        self.beta_1 = 0.006
        self.K_b = 0.05
        self.K_diff = 0.75
        self.K_diff_L = 0.09
        self.K_e = 0.009
        self.K_v = 0.05
        self.delta_r = 7.5e-05
        self.k_v = 3.22e-05
        self.D = 2.66e-11
        self.rho_a0 = 0.35
        self.rho_d = 0.18
        self.mu_h = 0.003
        self.r_0 = 0.00015
        self.delta_0 = 0.0001

        # Process related parameters
        self.Y_sX = 1.85
        self.Y_sP = 0.9
        self.m_s = 0.029
        self.c_oil = 1000.
        self.c_s = 600.
        self.Y_O2_X = 650.
        self.Y_O2_P = 160.
        self.m_O2_X = 17.5
        self.alpha_kla = 85.
        self.a = 0.38
        self.b = 0.34
        self.c = -0.38
        self.d = 0.25
        self.Henrys_c = 0.0251
        self.n_imp = 3.
        self.r = 2.1
        self.r_imp = 0.85
        self.Po = 5.
        self.epsilon = 0.1
        self.g = 9.81
        self.R = 8.314
        self.X_crit_DO2 = 0.1
        self.P_crit_DO2 = 0.3
        self.A_inhib = 1.
        self.Tf = 288.
        self.Tw = 288.
        self.Tcin = 285.
        self.Th = 333.
        self.Tair = 290.
        self.C_ps = 5.9
        self.C_pw = 4.18
        self.dealta_H_evap = 2430.7
        self.U_jacket = 36.
        self.A_c = 105.
        self.Eg = 14880.
        self.Ed = 173250.
        self.k_g = 450.
        self.k_d = 2.5e+29
        self.Y_QX = 25.
        self.abc = 0.033
        self.gamma1 = 3.25e-07
        self.gamma2 = 2.5e-11
        self.m_ph = 0.0025
        self.K1 = 1e-05
        self.K2 = 2.5e-08
        self.B_1 = -64.29
        self.B_2 = -1.825
        self.B_3 = 0.3649
        self.B_4 = 0.128
        self.B_5 = -0.00049496
        self.k3 = 0.005
        self.k1 = 0.001
        self.k2 = 0.0001
        self.t1 = 1.
        self.t2 = 250.
        self.q_co2 = 0.1353
        self.alpha_evp = 0.000524
        self.beta_T = 2.88
        self.pho_g = 1540.
        self.pho_oil = 900.
        self.pho_w = 1000.
        self.O_2_in = 0.21
        self.C_CO2_in = 0.033
        self.Tv = 373.
        self.T0 = 273.
        self.alpha_1 = 2451.8

        # Fixed input
        self.rpm = 100
        self.f_dis = 0

        # Temperature polyfit parameter
        self.p00_cold = 334.2
        self.p10_cold = -0.6514
        self.p01_cold = -1.481e+04
        self.p11_cold = 1.924e+06
        self.p02_cold = -2.583e+06
        self.p12_cold = 3089.
        self.p03_cold = 1.123e+05

        self.p00_hot = -899.7
        self.p10_hot = +1.754
        self.p01_hot = +3.989e+04
        self.p11_hot = -5.179e+06
        self.p02_hot = +6.954e+06
        self.p12_hot = -8317.
        self.p03_hot = -3.024e+05

        # Make step function for ode integration
        self.step_fcn = self._make_step_function()

    def go_step(self, x, u):
        scaled_x = ut.zero_mean_scale(x, self.x_min, self.x_max)
        scaled_u = ut.zero_mean_scale(u, self.u_min, self.u_max)
        scaled_p = []
        scaled_up = np.hstack((scaled_u, scaled_p))

        result = self.step_fcn(x0=scaled_x, p=scaled_up)
        scaled_next_x = np.squeeze(np.array(result['xf']))

        next_x = ut.zero_mean_descale(scaled_next_x, self.x_min, self.x_max)
        return next_x

    def get_observation(self, x):
        y = np.zeros(self.y_dim, dtype=np.float64)
        y[0] = x[0] + np.random.normal(0, self.measure_std[0], 1)
        y[1] = x[1] + x[2] + x[3] + x[4] + np.random.normal(0, self.measure_std[1], 1)
        y[2] = x[6] + np.random.normal(0, self.measure_std[2], 1)
        y[3] = x[7] + np.random.normal(0, self.measure_std[3], 1)
        y[4] = x[8] + np.random.normal(0, self.measure_std[4], 1)
        return y

    def do_reset(self):
        return self.ini_x, self.ini_u, self.ini_y, self.ini_p

    def _system_dynamics(self, x, u, p):
        time, a_0, a_1, a_3, a_4, integral_x, s, p, vol, temp, hydro_ion, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, \
        dissolved_o2, viscosity, phi0 = ca.vertsplit(x)
        Fs, Foil, Fa, Fb, Fc, Fh = ca.vertsplit(u)
        Fg = 200
        Fg = Fg / 60
        pho_b = 1100 + p + a_0 + a_1 + a_3 + a_4
        Fw = 100
        pressure = 0.8
        F_discharge = -150

        # Process parameters
        A_t1 = integral_x / (a_0 + a_1 + a_3 + a_4)
        total_X = a_0 + a_1 + a_3 + a_4

        h_b = (vol / 1000) / (np.pi * self.r ** 2) * (1 - self.epsilon)
        pressure_bottom = 1 + pressure + (pho_b * h_b) * 9.81 * 10 ** (-5)
        pressure_top = 1 + pressure
        log_mean_presure = (pressure_bottom - pressure_top) / (np.log(pressure_bottom / pressure_top))
        total_pressure = log_mean_presure

        DOstar_tp = (total_pressure * self.O_2_in) / self.Henrys_c

        pH_inhib = 1 / (1 + (hydro_ion / self.K1) + (self.K2 / hydro_ion))
        T_inhib = self.k_g * np.exp(-(self.Eg / (self.R * temp))) - self.k_d * np.exp(-(self.Ed / (self.R * temp)))
        DO_2_inhib_X = 0.5 * (1 - np.tanh(
            self.A_inhib * (self.X_crit_DO2 * total_pressure * self.O_2_in / self.Henrys_c - dissolved_o2)))
        DO_2_inhib_P = 0.5 * (1 - np.tanh(
            self.A_inhib * (self.P_crit_DO2 * total_pressure * self.O_2_in / self.Henrys_c - dissolved_o2)))
        pH = -np.log10(hydro_ion)
        k4 = np.exp(self.B_1 + self.B_2 * pH + self.B_3 * temp + self.B_4 * pH ** 2 + self.B_5 * temp ** 2)
        mu_h = k4

        # Main rate of equations for kinetics
        P_inhib = 2.5 * self.P_std_dev * (self.P_std_dev * np.sqrt(2 * np.pi)) ** (-1) * np.exp(
            -0.5 * ((s - self.mean_P) / self.P_std_dev) ** 2)
        mu_a0 = self.ratio_mu_e_mu_b * self.mux_max * pH_inhib * T_inhib * DO_2_inhib_X
        mu_e = self.mux_max * pH_inhib * T_inhib * DO_2_inhib_X

        K_diff = self.K_diff - (A_t1 * self.beta_1)
        K_diff = ca.fmax(self.K_diff_L, K_diff)

        r_b0 = mu_a0 * a_1 * s / (self.K_b + s)
        r_sb0 = self.Y_sX * r_b0

        r_e1 = (mu_e * a_0 * s) / (self.K_e + s)
        r_se1 = self.Y_sX * r_e1

        r_d1 = self.mu_diff * a_0 / (K_diff + s)
        r_m0 = self.m_s * a_0 / (K_diff + s)

        v_2 = phi0
        v_2 += (4 * np.pi * (1.5e-04 + 0 * self.delta_r) ** 3) / 3 * n1 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 1 * self.delta_r) ** 3) / 3 * n2 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 2 * self.delta_r) ** 3) / 3 * n3 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 3 * self.delta_r) ** 3) / 3 * n4 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 4 * self.delta_r) ** 3) / 3 * n5 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 5 * self.delta_r) ** 3) / 3 * n6 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 6 * self.delta_r) ** 3) / 3 * n7 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 7 * self.delta_r) ** 3) / 3 * n8 * self.delta_r
        v_2 += (4 * np.pi * (1.5e-04 + 8 * self.delta_r) ** 3) / 3 * n9 * self.delta_r

        rho_a1 = a_1 / ((a_1 / self.rho_a0) + v_2)
        v_a1 = a_1 / (2 * rho_a1) - v_2

        r_p = self.mu_p * self.rho_a0 * v_a1 * P_inhib * DO_2_inhib_P - mu_h * p
        r_m1 = (self.m_s * self.rho_a0 * v_a1 * s) / (self.K_v + s)
        r_d4 = self.mu_a * a_3

        dn0_dt = ((self.mu_v * v_a1) / (self.K_v + s)) * (6 / np.pi) * (self.r_0 + self.delta_0) ** (-3) - self.k_v * n0
        dn1_dt = -self.k_v * ((n2 - n0) / (2 * self.delta_r)) + self.D * (n2 - 2 * n1 + n0) / self.delta_r ** 2
        dn2_dt = -self.k_v * ((n3 - n1) / (2 * self.delta_r)) + self.D * (n3 - 2 * n2 + n1) / self.delta_r ** 2
        dn3_dt = -self.k_v * ((n4 - n2) / (2 * self.delta_r)) + self.D * (n4 - 2 * n3 + n2) / self.delta_r ** 2
        dn4_dt = -self.k_v * ((n5 - n3) / (2 * self.delta_r)) + self.D * (n5 - 2 * n4 + n3) / self.delta_r ** 2
        dn5_dt = -self.k_v * ((n6 - n4) / (2 * self.delta_r)) + self.D * (n6 - 2 * n5 + n4) / self.delta_r ** 2
        dn6_dt = -self.k_v * ((n7 - n5) / (2 * self.delta_r)) + self.D * (n7 - 2 * n6 + n5) / self.delta_r ** 2
        dn7_dt = -self.k_v * ((n8 - n6) / (2 * self.delta_r)) + self.D * (n8 - 2 * n7 + n6) / self.delta_r ** 2
        dn8_dt = -self.k_v * ((n9 - n7) / (2 * self.delta_r)) + self.D * (n9 - 2 * n8 + n7) / self.delta_r ** 2
        dn9_dt = -self.k_v * ((nm - n8) / (2 * self.delta_r)) + self.D * (nm - 2 * n9 + n8) / self.delta_r ** 2
        n_k = dn9_dt
        r_k = self.r_0 + 8 * self.delta_r
        r_m = self.r_0 + 10 * self.delta_r
        dn_m_dt = self.k_v * n_k / (r_m - r_k) - self.mu_a * nm
        n_k = n9
        dphi_0_dt = ((self.mu_v * v_a1) / (self.K_v + s)) - self.k_v * n0 * (np.pi * (self.r_0 + self.delta_0) ** 3) / 6

        # Volume expressions
        F_evp = vol * self.alpha_evp * (np.exp(2.5 * (temp - self.T0) / (self.Tv - self.T0)) - 1)
        pho_feed = self.c_s / 1000 * self.pho_g + (1 - self.c_s / 1000) * self.pho_w
        dilution = Fs + Fb + Fa + Fw - F_evp + Foil
        dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000) - F_evp + Foil

        # ODE's for biomass regions
        da_0_dt = r_b0 - r_d1 - a_0 * dilution / vol

        da_1_dt = r_e1 - r_b0 + r_d1 - (
                np.pi * (r_k + r_m) ** 3 / 6) * self.rho_d * self.k_v * n_k - a_1 * dilution / vol
        da_3_dt = (np.pi * ((r_k + r_m) ** 3 / 6)) * self.rho_d * self.k_v * n_k - r_d4 - a_3 * dilution / vol
        da_4_dt = r_d4 - a_4 * dilution / vol
        dP_dt = r_p - p * dilution / vol

        X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
        X_t = a_0 + a_1 + a_3 + a_4

        Qrxn_X = X_1 * self.Y_QX * vol * self.Y_O2_X / 1000
        Qrxn_P = dP_dt * self.Y_QX * vol * self.Y_O2_P / 1000
        Qrxn_t = Qrxn_X + Qrxn_P
        Qrxn_t = ca.fmax(Qrxn_t, 0)

        N = self.rpm / 60
        D_imp = 2 * self.r_imp
        unaerated_power = self.n_imp * self.Po * pho_b * (N ** 3) * (D_imp ** 5)
        P_g = 0.706 * ((unaerated_power ** 2 * N * D_imp ** 3) / (Fg ** 0.56)) ** 0.45
        P_n = P_g / unaerated_power
        variable_power = (self.n_imp * self.Po * pho_b * N ** 3 * D_imp ** 5 * P_n) / 1000

        # Substrate Utilization
        ds_dt = -r_se1 - r_sb0 - r_m0 - r_m1 - (self.Y_sP * self.mu_p * self.rho_a0 * v_a1 * DO_2_inhib_P) \
                + Fs * self.c_s / vol + Foil * self.c_oil / vol - s * dilution / vol

        # Dissolved Oxygen
        V_s = Fg / (np.pi * self.r ** 2)
        V_m = vol / 1000
        P_air = (V_s * self.R * temp * V_m / (22.4 * h_b)) * np.log(1 + pho_b * 9.81 * h_b / (pressure_top * 10 ** 5))
        P_t1 = variable_power + P_air
        vis_scaled = viscosity / 100
        oil_f = Foil / vol
        kla = self.alpha_kla * ((V_s ** self.a) * (P_t1 / V_m) ** self.b * (vis_scaled ** self.c)) * (
                    1 - oil_f ** self.d)

        OUR = (-X_1) * self.Y_O2_X - self.m_O2_X * X_t - dP_dt * self.Y_O2_P
        OTR = kla * (DOstar_tp - dissolved_o2)
        ddo2_dt = OUR + OTR - (dissolved_o2 * dilution / vol)

        # pH : only consider acidic environment
        pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw
        cb = -self.abc
        caa = self.abc
        h = 0.2
        h_ode = h / 20
        step1 = h_ode
        B = -(hydro_ion * vol + caa * Fa * step1 + cb * Fb * step1) / (vol + Fb * step1 + Fa * step1)
        dH_dt = self.gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + self.m_ph * total_X) + self.gamma1 * r_p \
                + self.gamma2 * pH_dis + ((-B + np.sqrt(B ** 2 + 4e-14)) / 2 - hydro_ion)

        # Temperature
        Ws = P_t1
        Qcon = self.U_jacket * self.A_c * (temp - self.Tair)
        Qcool = self.p00_cold + self.p10_cold * Fc + self.p01_cold * (1 / pho_b) + self.p11_cold * Fc * (1 / pho_b) + \
                self.p02_cold * (1 / pho_b) ** 2 + self.p12_cold * Fc * (1 / pho_b) ** 2 + self.p03_cold * (
                        1 / pho_b) ** 3
        Qhot = self.p00_hot + self.p10_hot * Fh + self.p01_hot * (1 / pho_b) + self.p11_hot * Fh * (1 / pho_b) + \
               self.p02_hot * (1 / pho_b) ** 2 + self.p12_hot * Fh * (1 / pho_b) ** 2 + self.p03_hot * (1 / pho_b) ** 3
        dQ_dt = Fs * pho_feed * self.C_ps * (self.Tf - temp) / 1000 + Fw * self.pho_w * self.C_pw * (
                self.Tw - temp) / 1000 \
                - F_evp * pho_b * self.C_pw / 1000 - self.dealta_H_evap * F_evp * self.pho_w / 1000 \
                + Qrxn_t + Ws - Qcon - Qcool - Qhot

        dT_dt = dQ_dt / (vol / 1000 * self.C_pw * pho_b)

        # Viscosity
        dvis_dt = 3 * (a_0 ** (1 / 3)) * (1 / (1 + np.exp(-self.k1 * (time - self.t1)))) * (
                1 / (1 + np.exp(-self.k2 * (time - self.t2)))) - self.k3 * Fw

        # Total X
        dintx_dt = a_0 + a_1 + a_3 + a_4

        # time
        dt_dt = 1

        xdot = [dt_dt, da_0_dt, da_1_dt, da_3_dt, da_4_dt, dintx_dt, ds_dt, dP_dt, dV1, dT_dt, dH_dt,
                dn0_dt, dn1_dt, dn2_dt, dn3_dt, dn4_dt, dn5_dt, dn6_dt, dn7_dt, dn8_dt, dn9_dt, dn_m_dt, ddo2_dt,
                dvis_dt, dphi_0_dt]

        return xdot

    def _make_step_function(self):
        # scaled in - scaled out function
        x_ca = ca.SX.sym('x', self.x_dim)
        u_ca = ca.SX.sym('u', self.u_dim)
        p_ca = ca.SX.sym('p', self.p_dim)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = ut.zero_mean_descale(x_ca, self.x_min, self.x_max)
        u_d = ut.zero_mean_descale(u_ca, self.u_min, self.u_max)
        p_d = ut.zero_mean_descale(p_ca, self.p_min, self.p_max)

        # Integrating ODE with Casadi with solver cvodes / idas
        xdot = self._system_dynamics(x_d, u_d, p_d)
        xdot = ca.vcat(xdot)
        xdot = np.multiply(xdot, self.scale_grad)  # Because of scaling
        ode = {'x': x_ca, 'p': up_ca, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        ode_integrator = ca.integrator('Integrator', 'idas', ode, options)
        return ode_integrator
