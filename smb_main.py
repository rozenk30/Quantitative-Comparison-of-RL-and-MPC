import os
import time
import random
import numpy as np
import tensorflow as tf

from Utility import utility as ut
from Systems import sys_smb
from Sysids import sysid
from Estimators import estimator
from Controls import controller
from smb_config import SmbConfig
from smb_test import test
from datetime import datetime

# GPU on/off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TOTAL_RUN = 10
ONLINE_SWITCH = 1001*0  # Hour
OFFLINE_SWITCH = 1000*1  # Hour
# ONLINE_SAVE_INTERVAL = 100
# OFFLINE_SAVE_INTERVAL = 100
# ONLINE_SAVE_INDEX = np.arange(ONLINE_SAVE_INTERVAL, ONLINE_SWITCH, ONLINE_SAVE_INTERVAL).astype('int')
# OFFLINE_SAVE_INDEX = np.arange(OFFLINE_SAVE_INTERVAL, OFFLINE_SWITCH, OFFLINE_SAVE_INTERVAL).astype('int')
ONLINE_SAVE_INDEX = np.array([20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)
OFFLINE_SAVE_INDEX = np.array([20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)

SYSID_METHOD = 'N4SID'  # 'N4SID', 'LSTM', 'NNARX', 'STACKING'
ESTIMATION_METHOD = 'KF'  # 'KF', 'LSTM_KF', 'STACKING', 'RL_STACKING'
CONTROL_METHOD = 'MPC'  # 'QMPC', 'MPC', 'DDPG', 'TD3', 'SAC'
EXTRA_FILE_NAME = 'MPC-I4-p32'

ONLINE_COST_FILE_NAME = 'smb_results_online_cost_1.txt'
OFFLINE_COST_FILE_NAME = 'smb_results_offline_cost_1.txt'
SYSID_ERR_FILE_NAME = 'smb_results_sysid_errors_1.txt'

LEARNING_PLOT_BOOL = False
TEST_PLOT_BOOL = False
SAVE_BOOL = True

NP_DATA_TYPE = np.float64
TF_DATA_TYPE = tf.float64


def offline_learning(config, file_name):
    learning_offline_ini_horizon = 1  # int(0.2*OFFLINE_SWITCH)
    file_name += '-offline'
    sysid_err_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size, dtype=NP_DATA_TYPE)

    """ Set up PLANT """
    plant = sys_smb.SysSMB(config)

    """ Set up SYSID """
    smb_sysid = sysid.SYSID(plant=plant, config=config)

    """ Offline simulation """
    x = np.zeros((32*OFFLINE_SWITCH + 1, plant.x_dim), dtype=NP_DATA_TYPE)
    y = np.zeros((32*OFFLINE_SWITCH + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    u = np.zeros((32*OFFLINE_SWITCH + 1, plant.u_dim), dtype=NP_DATA_TYPE)
    p = np.zeros((32*OFFLINE_SWITCH + 1, plant.p_dim), dtype=NP_DATA_TYPE)
    c = np.zeros((32*OFFLINE_SWITCH + 1, 1), dtype=NP_DATA_TYPE)
    r = np.zeros((32*OFFLINE_SWITCH + 1, 2), dtype=NP_DATA_TYPE)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    # Generating random signal
    random_signal = ut.random_step_signal_generation(signal_dim=plant.u_dim,
                                                     signal_length=config.SYSID_signal_length,
                                                     signal_min=config.SYSID_signal_min,
                                                     signal_max=config.SYSID_signal_max,
                                                     signal_interval=config.SYSID_signal_interval)

    # Short-cut
    # import pickle
    # with open('smb-offline_data-s-' + str(config.seed) + '.pickle', 'rb') as handle:
    #     data_set = pickle.load(handle)
    # x = data_set[0]
    # y = data_set[1]
    # u = data_set[2]
    # c = data_set[3]
    # r = data_set[4]
    # p = data_set[5]

    # Pre-simulation for System Identification or Learning
    for k in range(32*OFFLINE_SWITCH):
        r[k, :] = plant.ref
        if k > 32*learning_offline_ini_horizon:
            u[k, :] = random_signal[k, :]  # Random signal
        else:
            u[k, :] = plant.ss_u  # Steady state input
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_concentration()  # Disturbance generation
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :], u[k, :])
    r[-1, :] = r[-2, :]
    u[-1, :] = u[-2, :]
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.p_now

    if LEARNING_PLOT_BOOL:
        print('Plot offline signal simulation')
        config.plot_data(32*OFFLINE_SWITCH + 1, u, y, r, c)

    if SAVE_BOOL:
        config.save_data(directory, file_name, config.seed, x, x, u, y, p, r, c)

    """ Put data into the sysid """
    smb_sysid.add_data_and_scale(u, y)

    for k, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        print('seed:', config.seed, 'current offline step:', operated_time)
        smb_sysid.do_identification(32*operated_time)
        sysid_err_trajectory[k] = smb_sysid.sysid_error
        smb_controller = controller.CONTROLLER(smb_sysid, config)
        smb_controller.save_controller(directory, file_name + '-ctrl-' + str(operated_time))
    return sysid_err_trajectory, smb_sysid


def online_learning(config, file_name):
    learning_online_ini_horizon = 1
    file_name += '-online'

    """ Set up plant """
    plant = sys_smb.SysSMB(config)

    """ Set up sysid """
    smb_sysid = sysid.SYSID(plant=plant, config=config)

    """ Set up Estimator """
    smb_estimator = estimator.ESTIMATOR(sysid=smb_sysid, config=config)

    """ Set up controller """
    smb_controller = controller.CONTROLLER(smb_sysid, config)

    """ Pre-train of Q """  # ### - pretrain to config
    # if CONTROL_METHOD == 'QMPC':
    #     smb_controller.controller.pre_update_critic_by_model(50)

    """ Losses """  # ### - clean up later
    actor_loss = np.zeros((32*ONLINE_SWITCH + 1, 1), dtype=NP_DATA_TYPE)
    critic_loss = np.zeros((32*ONLINE_SWITCH + 1, 1), dtype=NP_DATA_TYPE)

    """ Online Simulation with control """
    x = np.zeros((32*ONLINE_SWITCH + 1, plant.x_dim), dtype=NP_DATA_TYPE)
    y = np.zeros((32*ONLINE_SWITCH + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    u = np.zeros((32*ONLINE_SWITCH + 1, plant.u_dim), dtype=NP_DATA_TYPE)
    p = np.zeros((32*ONLINE_SWITCH + 1, plant.p_dim), dtype=NP_DATA_TYPE)
    c = np.zeros((32*ONLINE_SWITCH + 1, 1), dtype=NP_DATA_TYPE)
    r = np.zeros((32*ONLINE_SWITCH + 1, 2), dtype=NP_DATA_TYPE)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()
    x_est = np.zeros((32*ONLINE_SWITCH + 1, smb_sysid.x_est_dim), dtype=NP_DATA_TYPE)

    for k in range(32*ONLINE_SWITCH):
        r[k, :] = plant.ref
        if k < 32*learning_online_ini_horizon:
            u[k, :] = plant.ss_u
        else:
            u[k, :] = smb_controller.control(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_concentration()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :], u[k, :])
        mode = plant.get_mode(x[k + 1, :])
        x_est[k + 1, :] = smb_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :], mode)

        # Controller learning
        smb_controller.save_data_to_buffer(x_est[k, :], u[k, :], c[k, :], x_est[k + 1, :], False)
        actor_loss[k, :], critic_loss[k, :] = smb_controller.train()

        if int(k + 1) in 32*ONLINE_SAVE_INDEX:
            print('seed:', config.seed, 'current online step:', int((k + 1)/32))
            if SAVE_BOOL:
                config.save_data(directory, file_name, config.seed, x, x_est, u, y, p, r, c)
                config.save_something(directory, file_name, 'actor-loss', actor_loss)
                config.save_something(directory, file_name, 'critic-loss', critic_loss)
            smb_controller.save_controller(directory, file_name + '-ctrl-' + str(int((k + 1)/32)))
    r[-1, :] = r[-2, :]
    u[-1, :] = smb_controller.control_without_exploration(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_concentration()

    if LEARNING_PLOT_BOOL:
        config.plot_data(32*ONLINE_SWITCH + 1, u, y, r, c)

    if SAVE_BOOL:
        config.save_data(directory, file_name, config.seed, x, x_est, u, y, p, r, c)
        config.save_something(directory, file_name, 'actor-loss', actor_loss)
        config.save_something(directory, file_name, 'critic-loss', critic_loss)

    return smb_sysid


def offline_test(config, smb_sysid, file_name):
    file_name = file_name + '-offline'
    cost_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size)
    smb_controller = controller.CONTROLLER(smb_sysid, config)
    for k, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        smb_controller.load_controller(directory, file_name + '-ctrl-' + str(operated_time))
        smb_sysid = smb_controller.controller.sysid  # must use loaded sysid... later change?
        smb_estimator = estimator.ESTIMATOR(sysid=smb_sysid, config=config)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(operated_time), config,
                                  smb_controller, smb_sysid, smb_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


def online_test(config, smb_sysid, file_name):
    file_name = file_name + '-online'
    cost_trajectory = np.zeros(ONLINE_SAVE_INDEX.size)
    smb_controller = controller.CONTROLLER(smb_sysid, config)
    for k, operated_time in enumerate(ONLINE_SAVE_INDEX):
        smb_controller.load_controller(directory, file_name + '-ctrl-' + str(operated_time))
        smb_sysid = smb_controller.controller.sysid  # must use loaded sysid... later change?
        smb_estimator = estimator.ESTIMATOR(sysid=smb_sysid, config=config)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(operated_time), config,
                                  smb_controller, smb_sysid, smb_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


if __name__ == '__main__':
    total_computation_time = time.time()
    execute_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = ut.make_directory('Data', 'smb', execute_time + '-' + CONTROL_METHOD + '-' + EXTRA_FILE_NAME)

    online_costs = np.zeros((TOTAL_RUN, ONLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)
    offline_costs = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)
    sysid_errors = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)

    config = SmbConfig()
    config.plot_bool = LEARNING_PLOT_BOOL
    config.sysid_method = SYSID_METHOD
    config.estimate_method = ESTIMATION_METHOD
    config.control_method = CONTROL_METHOD

    for seed in range(TOTAL_RUN):
        " Seeding "
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print('Current run is ' + str(seed))

        file_name = CONTROL_METHOD + '-' + SYSID_METHOD + '-' + ESTIMATION_METHOD
        file_name += '-SEED' + str(seed) + '-' + str(ONLINE_SWITCH) + '-' + str(OFFLINE_SWITCH)
        file_name += '-' + EXTRA_FILE_NAME
        config.seed = seed

        if SAVE_BOOL:
            config.save_settings(directory, file_name)

        # Offline learning
        if OFFLINE_SWITCH > 0:
            learning_time = time.time()
            errors, smb_sysid = offline_learning(config, file_name)
            sysid_errors[seed, :] = errors
            offline_costs[seed, :] = offline_test(config, smb_sysid, file_name)
            print('Offline run time takes', str(np.round(time.time() - learning_time, 4)))

        # Online learning
        if ONLINE_SWITCH > 0:
            learning_time2 = time.time()
            smb_sysid = online_learning(config, file_name)
            online_costs[seed, :] = online_test(config, smb_sysid, file_name)
            print('Online run time takes', str(np.round(time.time() - learning_time2, 4)))

        np.savetxt(fname=ONLINE_COST_FILE_NAME, X=online_costs, fmt='%12.8f')
        np.savetxt(fname=OFFLINE_COST_FILE_NAME, X=offline_costs, fmt='%12.8f')
        np.savetxt(fname=SYSID_ERR_FILE_NAME, X=sysid_errors, fmt='%12.8f')

    print('Total computation time:', np.around(time.time() - total_computation_time, 4))
    if ONLINE_SWITCH > 0:
        print('final_online_cost', np.round(online_costs[:, -1], 4), np.round(np.mean(online_costs[:, -1]), 4))
    if OFFLINE_SWITCH > 0:
        print('final_offline_cost', np.round(offline_costs[:, -1], 4), np.round(np.mean(offline_costs[:, -1]), 4))

