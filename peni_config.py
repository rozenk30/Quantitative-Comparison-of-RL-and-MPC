import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PeniConfig(object):
    def __init__(self):
        self.seed = 12345
        self.plot_bool = True

        # Problem related
        self.control_method = None
        self.sysid_method = None
        self.estimate_method = None

        # System related parameters
        self.system_state_disturb = False
        self.system_measure_disturb = False
        self.system_para_disturb = False

        # Sysid related parameters
        self.SYSID_signal_length = 300
        self.SYSID_signal_min = np.array([30, 15.])  # np.array([40, 15.])
        self.SYSID_signal_max = np.array([200., 45.])  # np.array([200., 45.])
        self.SYSID_signal_interval = 30  # 30
        self.SYSID_ss_transform = False
        self.SYSID_min_max_relax = 5.
        self.N4SID_x_dim = 30
        self.LSTM_node_num = 50  # 50 good
        self.LSTM_learning_rate = 0.002
        self.STACK_u_order = 0
        self.STACK_y_order = 2
        self.STACK_o_dim = 0
        self.STACK_o_min = np.array([])
        self.STACK_o_max = np.array([])

        ''' Estimator related parameters '''
        # KF
        self.KF_process_noise_cov = np.eye(self.N4SID_x_dim)
        self.KF_measure_noise_cov = 0.5*np.eye(5)  # plant.y_dim
        self.KF_initial_cov = np.eye(self.N4SID_x_dim)

        # LSTM KF
        self.LSTM_KF_process_noise_cov = np.eye(2*self.LSTM_node_num)
        self.LSTM_KF_measure_noise_cov = 1.5*np.eye(5)  # plant.y_dim / 1.5
        self.LSTM_KF_initial_cov = np.eye(2*self.LSTM_node_num)

        # NNARX
        self.NNARX_nn_model_node_num = [400, 300, 5]  # 400 300 ydim
        self.NNARX_nn_model_act_fcn = ['tanh', 'linear', 'linear']
        self.NNARX_learning_rate = 0.0001  # 0.0001
        self.NNARX_batch_size = 256
        self.NNARX_epochs = 500  # 500
        self.NNARX_validation_split_ratio = 0.2

        ''' Control related parameters '''
        # DDPG related
        self.DDPG_actor_node_num = [400, 300, 2]  # 400 300 u_dim
        self.DDPG_actor_activation_fcn = ['relu', 'relu', 'tanh']  # relu relu tanh
        self.DDPG_actor_learning_rate = 0.0002  # 0.0001 ??
        self.DDPG_actor_target_update_parameter = 0.0050  # 0.001
        self.DDPG_critic_node_num = [400, 300, 1]  # 400 300 1
        self.DDPG_critic_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.DDPG_critic_learning_rate = 0.0002  # 0.001
        self.DDPG_critic_target_update_parameter = 0.0050  # 0.001
        self.DDPG_batch_size = 2*256  # 64
        self.DDPG_discount_factor = 1.0  # 0.99
        self.DDPG_buffer_size = int(1e6)  # int(1e6)
        self.DDPG_ou_linear_coefficient = 0.1  # Noise parameters
        self.DDPG_ou_diffusion = 0.30  # Noise parameters
        self.DDPG_value_max = 100.  # Maximum value, should depend on discount factor

        # TD3 related
        self.TD3_actor_node_num = [400, 300, 2]  # 400 300 u_dim
        self.TD3_actor_activation_fcn = ['relu', 'relu', 'tanh']  # relu relu tanh
        self.TD3_actor_learning_rate = 0.0002  # 0.001
        self.TD3_actor_target_update_parameter = 0.0180  # 0.005
        self.TD3_critic_node_num = [400, 300, 1]  # 400 300 1
        self.TD3_critic_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.TD3_critic_learning_rate = 0.0010  # 0.001
        self.TD3_critic_target_update_parameter = 0.0180  # 0.005
        self.TD3_batch_size = 2*256  # 64
        self.TD3_discount_factor = 1.0  # 0.99
        self.TD3_buffer_size = int(1e6)  # int(1e6)
        self.TD3_exploration_noise_std = 1.00  # 0.2
        self.TD3_noise_std = 0.  # 0.1
        self.TD3_noise_bound = 0.5  # 0.5
        self.TD3_actor_update_period = 1
        self.TD3_value_max = 100.  # Maximum value, should depend on discount factor

        # SAC related
        self.SAC_actor_node_num = [400, 300, 4]  # 256 256 2*u_dim
        self.SAC_actor_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.SAC_actor_learning_rate = 0.0003  # 0.0003
        self.SAC_critic_node_num = [400, 300, 1]  # 256 256 1
        self.SAC_critic_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.SAC_critic_learning_rate = 0.0003  # 0.0003
        self.SAC_critic_target_update_parameter = 0.005  # 0.005
        self.SAC_batch_size = 2*256  # 256
        self.SAC_discount_factor = 1.0  # 0.99
        self.SAC_buffer_size = int(1e6)  # int(1e6)
        self.SAC_exploration_noise_std = 1.0  # 0.2
        self.SAC_temperature = 0.0005  # None
        self.SAC_epsilon = 1e-10
        self.SAC_value_max = 100.  # Maximum value, should depend on discount factor

        # MPC related
        self.MPC_g_dim = 0
        self.MPC_h_dim = 0
        self.MPC_prediction_horizon = 230

        # QMPC related
        self.QMPC_g_dim = 0
        self.QMPC_h_dim = 0
        self.QMPC_prediction_horizon = 10
        self.QMPC_critic_node_num = [100, 50, 1]  # 400 300 1
        self.QMPC_critic_activation_fcn = ['swish', 'swish', 'linear']  # relu relu linear
        self.QMPC_critic_learning_rate = 0.001  # 0.001
        self.QMPC_critic_target_update_parameter = 0.005  # 0.001
        self.QMPC_batch_size = 16  # 64
        self.QMPC_discount_factor = 0.9  # 0.99
        self.QMPC_buffer_size = int(1e6)  # int(1e6)
        self.QMPC_noise_std = 0.05  # Noise parameter
        self.QMPC_value_max = 100.  # Maximum value, should depend on discount factor

    def mpc_abstract_stage_cost(self, x, u, for_casadi, observe_model):
        y = observe_model(x, for_casadi=for_casadi)
        scaled_fs = u[0] / 200
        scaled_foil = u[1] / 200
        scaled_vol = y[4] / 100000
        p1, p2, p3 = 2*0.00090, 2*0.00150, 2*0.00200
        c = p1 * scaled_fs + p2 * scaled_foil + p3 * scaled_vol**2
        return c

    def mpc_abstract_terminal_cost(self, x, u, for_casadi, observe_model):
        y = observe_model(x, for_casadi=for_casadi)
        scaled_product = y[3] / 50
        scaled_vol = y[4] / 100000
        p1 = 0.5
        terminal_cost = p1*(1 - scaled_product*scaled_vol)
        c = terminal_cost
        return c

    def mpc_abstract_stage_constraint(self, x, u, for_casadi, observe_model):
        return []

    def mpc_abstract_terminal_constraint(self, x, u, for_casadi, observe_model):
        return []

    def mpc_initial_control(self, x):
        return np.array([120., 23.], dtype=np.float64)

    def save_settings(self, directory, file_name):
        self_dict_copy = copy.deepcopy(self.__dict__)
        for key in self_dict_copy:
            if isinstance(self_dict_copy[key], np.ndarray):
                self_dict_copy[key] = self_dict_copy[key].tolist()
            elif callable(self_dict_copy[key]):
                self_dict_copy[key] = None
        json.dump(self_dict_copy, open(Path.joinpath(directory, file_name + 'settings.json'), 'w'))

    @staticmethod
    def plot_data(horizon, u, y, c):
        x_time = np.arange(horizon)

        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plt.subplot(211)
        plt.step(x_time, u[:, 0])
        plt.title('Fs'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.subplot(212)
        plt.step(x_time, u[:, 1])
        plt.title('Foil'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.show()

        fig2 = plt.figure()
        fig2.set_size_inches(16, 9)
        plt.subplot(221)
        plt.plot(x_time, y[:, 1], label='X')
        plt.title('X'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.legend()

        plt.subplot(222)
        plt.plot(x_time, y[:, 2], label='S')
        plt.title('S'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.legend()

        plt.subplot(223)
        plt.plot(x_time, y[:, 3], label='P')
        plt.title('P'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.legend()

        plt.subplot(224)
        plt.plot(x_time, y[:, 4], label='V')
        plt.title('V'); plt.xlabel('Time'); plt.ylabel('Value')
        plt.show()

        fig3 = plt.figure()
        fig3.set_size_inches(16, 9)
        plt.plot(x_time, c[:, 0], linewidth=2,)
        plt.title('Cost', fontsize=15); plt.xlabel('Time', fontsize=14); plt.ylabel('Value', fontsize=14)
        plt.show()

    @staticmethod
    def save_data(directory, file_name, idx, x, x_est, u, y, p, c):
        np.savetxt(fname=Path.joinpath(directory, file_name + '-x-' + str(idx) + '.txt'), X=x, fmt='%12.8f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-x_est-' + str(idx) + '.txt'), X=x_est, fmt='%12.8f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-u-' + str(idx) + '.txt'), X=u, fmt='%12.8f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-y-' + str(idx) + '.txt'), X=y, fmt='%12.8f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-p-' + str(idx) + '.txt'), X=p, fmt='%12.8f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-c-' + str(idx) + '.txt'), X=c, fmt='%12.8f')

    @staticmethod
    def save_something(directory, file_name, idx, something):
        np.savetxt(fname=Path.joinpath(directory, file_name + '-' + str(idx) + '.txt'), X=something, fmt='%12.8f')





