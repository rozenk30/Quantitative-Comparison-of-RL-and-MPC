import numpy as np
from Controls import control_mpc
from Controls import control_qmpc
from Controls import control_ddpg
from Controls import control_td3
from Controls import control_sac
from Controls import control_pid

''' Master controller '''


class CONTROLLER(object):
    def __init__(self, sysid, config):
        self.sysid = sysid
        self.config = config
        self.control_method = config.control_method

        if self.control_method == 'MPC':
            self.controller = control_mpc.MPC(sysid, config)
        elif self.control_method == 'PID':  # ###
            self.pids = []
        elif self.control_method == 'QMPC':
            self.controller = control_qmpc.QMPC(sysid, config)
        elif self.control_method == 'DDPG':
            self.controller = control_ddpg.DDPG(sysid, config)
        elif self.control_method == 'TD3':
            self.controller = control_td3.TD3(sysid, config)
        elif self.control_method == 'SAC':
            self.controller = control_sac.SAC(sysid, config)
        else:
            self.controller = None
            print('No control')

    def control(self, x):
        if self.control_method == 'PID':  # ###
            u = np.zeros(self.sysid.plant.u_dim)
            return u

        elif self.control_method in ['MPC', 'QMPC', 'DDPG', 'TD3', 'SAC']:
            u = self.controller.control(x)
            return u

        else:
            return np.zeros(self.sysid.plant.u_dim)

    def control_without_exploration(self, x):
        if self.control_method == 'PID':  # ###
            u = np.zeros(self.sysid.plant.u_dim)
            return u

        elif self.control_method == 'MPC':
            u = self.controller.control(x)
            return u

        elif self.control_method in ['QMPC', 'DDPG', 'TD3', 'SAC']:
            u = self.controller.control_without_exploration(x)
            return u

        else:
            return np.zeros(self.sysid.plant.u_dim)

    def save_data_to_buffer(self, x, u, c, xp, is_terminal):
        if self.control_method in ['QMPC', 'DDPG', 'TD3', 'SAC']:
            self.controller.save_data_to_buffer(x, u, c, xp, is_terminal)
        else:
            pass

    def assign_buffer(self, replay_buffer):
        self.controller.replay_buffer = replay_buffer

    def train(self):
        if self.control_method in ['DDPG', 'TD3']:
            critic_loss = self.controller.update_critic()
            actor_loss = self.controller.update_actor()
            self.controller.update_critic_target()
            self.controller.update_actor_target()
            return actor_loss, critic_loss

        elif self.control_method == 'SAC':
            critic_loss = self.controller.update_critic()
            actor_loss = self.controller.update_actor()
            self.controller.update_critic_target()
            return actor_loss, critic_loss

        elif self.control_method == 'QMPC':
            critic_loss = self.controller.update_critic()
            self.controller.update_critic_target()
            return np.array([0.]), critic_loss

        else:
            return 0., 0.

    def save_controller(self, directory, name):
        if self.control_method in ['MPC', 'QMPC', 'DDPG', 'TD3', 'SAC']:
            self.controller.save_controller(directory, name)
        else:
            pass

    def load_controller(self, directory, name):
        if self.control_method in ['MPC', 'QMPC', 'DDPG', 'TD3', 'SAC']:
            self.controller.load_controller(directory, name)
        else:
            pass
