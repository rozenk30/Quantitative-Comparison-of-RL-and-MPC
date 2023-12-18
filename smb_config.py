import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SmbConfig(object):
    def __init__(self):
        self.seed = 12345
        self.plot_bool = True

        # Problem related
        self.ref = np.array([0.99, 0.99], dtype=np.float64)
        self.control_method = None
        self.sysid_method = None
        self.estimate_method = None

        # System related parameters
        self.system_state_disturb = False
        self.system_measure_disturb = False
        self.system_para_disturb = True

        # Sysid related parameters
        self.SYSID_signal_length = 10**6
        self.SYSID_signal_min = np.array([0.0100, 0.0100, 0.0135, 0.0135], dtype=np.float64)
        self.SYSID_signal_max = np.array([0.0135, 0.0135, 0.0155, 0.0155], dtype=np.float64)
        self.SYSID_signal_interval = 32*5  # 32*5
        self.SYSID_ss_transform = True
        self.SYSID_min_max_relax = 5.
        self.N4SID_x_dim = 30
        self.LSTM_node_num = 30
        self.LSTM_learning_rate = 0.002
        self.STACK_u_order = 4
        self.STACK_y_order = 4
        self.STACK_o_dim = 1
        self.STACK_o_min = np.array([0], dtype=np.float64)
        self.STACK_o_max = np.array([1], dtype=np.float64)

        ''' Estimator related parameters '''
        # KF
        self.KF_process_noise_cov = np.eye(self.N4SID_x_dim)
        self.KF_measure_noise_cov = 0.5*np.eye(4)  # plant.y_dim
        self.KF_initial_cov = np.eye(self.N4SID_x_dim)

        # LSTM KF
        self.LSTM_KF_process_noise_cov = np.eye(2*self.LSTM_node_num)
        self.LSTM_KF_measure_noise_cov = 1.5*np.eye(4)  # plant.y_dim
        self.LSTM_KF_initial_cov = np.eye(2*self.LSTM_node_num)

        # NNARX
        self.NNARX_nn_model_node_num = [100, 60, 4]  # 50 30 y_dim | 100 50 4
        self.NNARX_nn_model_act_fcn = ['tanh', 'linear', 'linear']
        self.NNARX_learning_rate = 0.0001
        self.NNARX_batch_size = 256
        self.NNARX_epochs = 500
        self.NNARX_validation_split_ratio = 0.2

        ''' Control related parameters '''
        # DDPG related
        self.DDPG_actor_node_num = [400, 300, 4]  # 400 300 u_dim
        self.DDPG_actor_activation_fcn = ['relu', 'relu', 'tanh']  # relu relu tanh
        self.DDPG_actor_learning_rate = 0.0001  # 0.0001 ??
        self.DDPG_actor_target_update_parameter = 0.0100  # 0.001
        self.DDPG_critic_node_num = [400, 300, 1]  # 400 300 1
        self.DDPG_critic_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.DDPG_critic_learning_rate = 0.0010  # 0.001
        self.DDPG_critic_target_update_parameter = 0.0100  # 0.001
        self.DDPG_batch_size = 2*256  # 64
        self.DDPG_discount_factor = 0.95  # 0.99
        self.DDPG_buffer_size = int(1e6)  # int(1e6)
        self.DDPG_ou_linear_coefficient = 0.15  # Noise parameters
        self.DDPG_ou_diffusion = 0.1  # Noise parameters
        self.DDPG_value_max = 100.  # Maximum value, should depend on discount factor

        # TD3 related
        self.TD3_actor_node_num = [400, 300, 4]  # 400 300 u_dim
        self.TD3_actor_activation_fcn = ['relu', 'relu', 'tanh']  # relu relu tanh
        self.TD3_actor_learning_rate = 0.0003  # 0.001
        self.TD3_actor_target_update_parameter = 0.0200  # 0.005
        self.TD3_critic_node_num = [400, 300, 1]  # 400 300 1
        self.TD3_critic_activation_fcn = ['relu', 'relu', 'linear']  # relu relu linear
        self.TD3_critic_learning_rate = 0.0010  # 0.001
        self.TD3_critic_target_update_parameter = 0.0200  # 0.005
        self.TD3_batch_size = 2*256  # 64
        self.TD3_discount_factor = 0.95  # 0.99
        self.TD3_buffer_size = int(1e6)  # int(1e6)
        self.TD3_exploration_noise_std = 0.05  # 0.2
        self.TD3_noise_std = 0.  # 0.1
        self.TD3_noise_bound = 0.5  # 0.5
        self.TD3_actor_update_period = 1
        self.TD3_value_max = 100.  # Maximum value, should depend on discount factor

        # SAC related
        self.SAC_actor_node_num = [400, 300, 8]  # 256 256 2*u_dim
        self.SAC_actor_activation_fcn = ['swish', 'swish', 'linear']  # relu relu linear
        self.SAC_actor_learning_rate = 0.0003  # 0.0003
        self.SAC_critic_node_num = [400, 300, 1]  # 256 256 1
        self.SAC_critic_activation_fcn = ['swish', 'swish', 'linear']  # relu relu linear
        self.SAC_critic_learning_rate = 0.0003  # 0.0003
        self.SAC_critic_target_update_parameter = 0.0100  # 0.005
        self.SAC_batch_size = 2*256  # 256
        self.SAC_discount_factor = 0.95  # 0.99
        self.SAC_buffer_size = int(1e6)  # int(1e6)
        self.SAC_exploration_noise_std = 0.05  # 0.2
        self.SAC_temperature = 0.001  # None
        self.SAC_epsilon = 1e-10
        self.SAC_value_max = 100.  # Maximum value, should depend on discount factor

        # MPC related
        self.MPC_g_dim = 0
        self.MPC_h_dim = 0
        self.MPC_prediction_horizon = 32  # 32

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
        puri_e = y[0] / (y[0] + y[1] + 1e-10)
        puri_r = y[3] / (y[2] + y[3] + 1e-10)
        loss_e = self.ref[0] - puri_e
        loss_r = self.ref[1] - puri_r
        y_weight = 0.03*np.array([1., 1.])
        u_cost = -1*(u[2] - u[1])
        y_cost = y_weight[0]*loss_e + y_weight[1]*loss_r
        c = 0.002 + y_cost + u_cost
        return c

    def mpc_abstract_terminal_cost(self, x, u, for_casadi, observe_model):
        y = observe_model(x, for_casadi=for_casadi)
        puri_e = y[0] / (y[0] + y[1] + 1e-10)
        puri_r = y[3] / (y[2] + y[3] + 1e-10)
        y_weight = 0.03 * np.array([1., 1.])
        u_cost = -1*(u[2] - u[1])
        y_cost = y_weight[0] * (self.ref[0] - puri_e) + y_weight[1] * (self.ref[1] - puri_r)
        c = 0.002 + y_cost + u_cost
        return c

    def mpc_abstract_stage_constraint(self, x, u, for_casadi, observe_model):
        return []

    def mpc_abstract_terminal_constraint(self, x, u, for_casadi, observe_model):
        return []

    def mpc_initial_control(self, x):
        return np.array([0.013, 0.013, 0.014, 0.014], dtype=np.float64)

    def save_settings(self, directory, file_name):
        self_dict_copy = copy.deepcopy(self.__dict__)
        for key in self_dict_copy:
            if isinstance(self_dict_copy[key], np.ndarray):
                self_dict_copy[key] = self_dict_copy[key].tolist()
            elif callable(self_dict_copy[key]):
                self_dict_copy[key] = None
        json.dump(self_dict_copy, open(Path.joinpath(directory, file_name + 'settings.json'), 'w'))

    @staticmethod
    def plot_data(horizon, u, y, r, c, *args):
        if args:
            extract_tank, raffinate_tank = args
            extract_tank_purity = extract_tank[:, 0] / (extract_tank[:, 0] + extract_tank[:, 1] + 1e-10)
            raffinate_tank_purity = raffinate_tank[:, 1] / (raffinate_tank[:, 0] + raffinate_tank[:, 1] + 1e-10)

        x_time = np.arange(horizon)
        fig = plt.figure()
        fig.set_size_inches(16, 9)

        extract_purity = y[:, 0] / (y[:, 0] + y[:, 1] + 1e-10)
        raffinate_purity = y[:, 3] / (y[:, 2] + y[:, 3] + 1e-10)

        plt.subplot(421)
        plt.step(x_time, u[:, 1])
        plt.title('u1')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(422)
        plt.step(x_time, u[:, 2])
        plt.title('u2')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(423)
        plt.plot(x_time, extract_purity, label='Extract purity')
        plt.plot(x_time, r[:, 0], '--k', label='Reference')
        plt.title('Extract purity')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(424)
        plt.plot(x_time, raffinate_purity, label='Raffinate purity')
        plt.plot(x_time, r[:, 1], '--k', label='Reference')
        plt.title('Raffinate purity')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(425)
        plt.plot(x_time, y[:, 0], label='Extract A')
        plt.title('Extract A')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(426)
        plt.plot(x_time, y[:, 1], label='Extract B')
        plt.title('Extract B')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(427)
        plt.plot(x_time, y[:, 2], label='Raffinate A')
        plt.title('Raffinate A')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(428)
        plt.plot(x_time, y[:, 3], label='Raffinate B')
        plt.title('Raffinate B')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

        if args:
            plt.subplot(411)
            plt.plot(x_time, extract_tank[:, 0], x_time, extract_tank[:, 1])
            plt.ylabel('Extract Tank')
            plt.subplot(412)
            plt.plot(x_time, raffinate_tank[:, 0], x_time, raffinate_tank[:, 1])
            plt.ylabel('Raffinate Tank')
            plt.subplot(413)
            plt.plot(x_time, extract_tank_purity, x_time, r[:, 0], '--k')
            plt.ylabel('Extract Tank Purity')
            plt.subplot(414)
            plt.plot(x_time, raffinate_tank_purity, x_time, r[:, 1], '--k')
            plt.ylabel('Raffinate Tank Purity')
            plt.show()

        plt.plot(x_time, c[:, 0], linewidth=2, )
        plt.title('Cost', fontsize=15)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.show()

    @staticmethod
    def save_data(directory, file_name, idx, x, x_est, u, y, p, r, c):
        np.savetxt(fname=Path.joinpath(directory, file_name + '-x-' + str(idx) + '.txt'), X=x, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-x_est-' + str(idx) + '.txt'), X=x_est, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-u-' + str(idx) + '.txt'), X=u, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-y-' + str(idx) + '.txt'), X=y, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-p-' + str(idx) + '.txt'), X=p, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-r-' + str(idx) + '.txt'), X=r, fmt='%30.20f')
        np.savetxt(fname=Path.joinpath(directory, file_name + '-c-' + str(idx) + '.txt'), X=c, fmt='%30.20f')

    @staticmethod
    def save_something(directory, file_name, idx, something):
        np.savetxt(fname=Path.joinpath(directory, file_name + '-' + str(idx) + '.txt'), X=something, fmt='%12.8f')





