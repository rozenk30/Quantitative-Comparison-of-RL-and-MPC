import os
import time
import random
import numpy as np
import tensorflow as tf

from Utility import utility as ut
from Systems import sys_peni_with_PID
from Sysids import sysid_batch as sysid
from Estimators import estimator
from Controls import controller
from peni_config import PeniConfig
from peni_test import test
from datetime import datetime

# GPU on/off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TOTAL_RUN = 10
ONLINE_BATCH = 1001*1  # Hour
OFFLINE_BATCH = 1000*0  # Hour
# ONLINE_SAVE_INTERVAL = 20
# OFFLINE_SAVE_INTERVAL = 10
# ONLINE_SAVE_INDEX = np.arange(ONLINE_SAVE_INTERVAL, ONLINE_BATCH, ONLINE_SAVE_INTERVAL).astype('int')
# OFFLINE_SAVE_INDEX = np.arange(OFFLINE_SAVE_INTERVAL, OFFLINE_BATCH, OFFLINE_SAVE_INTERVAL).astype('int')
ONLINE_SAVE_INDEX = np.array([20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)
OFFLINE_SAVE_INDEX = np.array([20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)

SYSID_METHOD = 'STACKING'  # 'N4SID', 'LSTM', 'NNARX', 'STACKING'
ESTIMATION_METHOD = 'STACKING'  # 'KF', 'LSTM_KF', 'STACKING', 'RL_STACKING'
CONTROL_METHOD = 'TD3'  # 'QMPC', 'MPC', 'DDPG', 'TD3', 'SAC'
EXTRA_FILE_NAME = 'B512D1E100AL3CL3UP180'

ONLINE_COST_FILE_NAME = 'peni_results_online_cost_1.txt'
OFFLINE_COST_FILE_NAME = 'peni_results_offline_cost_1.txt'
SYSID_ERR_FILE_NAME = 'peni_results_sysid_errors_1.txt'

LEARNING_PLOT_BOOL = False
TEST_PLOT_BOOL = False
SAVE_BOOL = True

NP_DATA_TYPE = np.float64
TF_DATA_TYPE = tf.float64


def offline_learning(config, file_name):
    offline_default_batch = 10
    file_name += '-offline'
    sysid_err_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size, dtype=NP_DATA_TYPE)

    """ Set up Plant"""
    plant = sys_peni_with_PID.SysPeniWithPID(config)
    total_horizon = plant.total_horizon

    """ Set up SYSID """
    peni_sysid = sysid.SYSID(plant=plant, config=config)

    """ Offline simulation for system identification """
    for batch in range(OFFLINE_BATCH):
        x = np.zeros((total_horizon + 1, plant.x_dim), dtype=NP_DATA_TYPE)
        y = np.zeros((total_horizon + 1, plant.y_dim), dtype=NP_DATA_TYPE)
        u = np.zeros((total_horizon + 1, plant.u_dim), dtype=NP_DATA_TYPE)
        p = np.zeros((total_horizon + 1, plant.p_dim), dtype=NP_DATA_TYPE)
        c = np.zeros((total_horizon + 1, 1), dtype=NP_DATA_TYPE)
        x[0, :], u[0, :], y[0, :], p[0, :] = plant.do_reset()

        # Generating random signal
        random_signal = ut.random_step_signal_generation(signal_dim=plant.u_dim,
                                                         signal_length=config.SYSID_signal_length,
                                                         signal_min=config.SYSID_signal_min,
                                                         signal_max=config.SYSID_signal_max,
                                                         signal_interval=config.SYSID_signal_interval)

        # Pre-simulation for System Identification or Learning
        for k in range(total_horizon):
            if batch < offline_default_batch or k < 10:
                u[k, :] = plant.get_reference_input(k)
            else:
                u[k, :] = random_signal[k, :]
            c[k, :] = plant.get_cost(y[k, :], u[k, :])
            x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
            y[k + 1, :] = plant.get_observation(x[k + 1, :])
        u[-1, :] = plant.get_reference_input(k)
        c[-1, :] = plant.get_cost(y[-1, :], u[-1, :])

        if batch + 1 in OFFLINE_SAVE_INDEX:
            if LEARNING_PLOT_BOOL:
                print('Plot offline signal simulation')
                config.plot_data(total_horizon + 1, u, y, c)

            if SAVE_BOOL:
                config.save_data(directory, file_name + '-batch' + str(batch), config.seed, x, x, u, y, p, c)

        """ Put data into the sysid """
        peni_sysid.add_data_and_scale(u, y)

    for k, batch in enumerate(OFFLINE_SAVE_INDEX):
        print('seed:', config.seed, 'current offline step:', batch)
        peni_sysid.do_identification(batch)
        sysid_err_trajectory[k] = peni_sysid.sysid_error
        peni_controller = controller.CONTROLLER(peni_sysid, config)
        peni_controller.save_controller(directory, file_name + '-ctrl-' + str(batch))
    return sysid_err_trajectory, peni_sysid


def online_learning(config, file_name):
    online_default_batch = 10
    file_name += '-online'

    """ Set up plant """
    plant = sys_peni_with_PID.SysPeniWithPID(config)
    total_horizon = plant.total_horizon

    """ Set up sysid """
    peni_sysid = sysid.SYSID(plant=plant, config=config)

    """ Set up estimator """
    peni_estimator = estimator.ESTIMATOR(sysid=peni_sysid, config=config)

    """ Set up controller """
    peni_controller = controller.CONTROLLER(peni_sysid, config)

    """ Pre-train of Q """  # ### - pretrain to config
    # if CONTROL_METHOD == 'QMPC':
    #     peni_controller.controller.pre_update_critic_by_model(50)

    """ Losses """  # ### - clean up later
    actor_loss = []
    critic_loss = []

    """ Online Simulation with control """
    for batch in range(ONLINE_BATCH):
        plant = sys_peni_with_PID.SysPeniWithPID(config)
        x = np.zeros((total_horizon + 1, plant.x_dim), dtype=NP_DATA_TYPE)
        y = np.zeros((total_horizon + 1, plant.y_dim), dtype=NP_DATA_TYPE)
        u = np.zeros((total_horizon + 1, plant.u_dim), dtype=NP_DATA_TYPE)
        p = np.zeros((total_horizon + 1, plant.p_dim), dtype=NP_DATA_TYPE)
        c = np.zeros((total_horizon + 1, 1), dtype=NP_DATA_TYPE)
        x[0, :], u[0, :], y[0, :], p[0, :] = plant.do_reset()
        x_est = np.zeros((total_horizon + 1, peni_sysid.x_est_dim), dtype=NP_DATA_TYPE)

        for k in range(total_horizon):
            if batch < online_default_batch or k < 10:
                u[k, :] = plant.get_reference_input(k)
            else:
                u[k, :] = peni_controller.control(x_est[k, :])
            c[k, :] = plant.get_cost(y[k, :], u[k, :])
            x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
            y[k + 1, :] = plant.get_observation(x[k + 1, :])
            x_est[k + 1, :] = peni_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])

            # Controller learn
            peni_controller.save_data_to_buffer(x_est[k, :], u[k, :], c[k, :], x_est[k + 1, :], False)
            al, cl = peni_controller.train()
            actor_loss.append(al)
            critic_loss.append(cl)
        u[-1, :] = plant.get_reference_input(k)
        c[-1, :] = plant.get_cost(y[-1, :], u[-1, :])
        peni_controller.save_data_to_buffer(x_est[k + 1, :], u[k + 1, :], c[k + 1, :], x_est[k + 1, :], True)
        al, cl = peni_controller.train()
        actor_loss.append(al)
        critic_loss.append(cl)

        if batch + 1 in ONLINE_SAVE_INDEX:
            print('seed:', config.seed, 'current online batch:', batch + 1)
            if LEARNING_PLOT_BOOL:
                print('Plot online signal simulation')
                config.plot_data(total_horizon + 1, u, y, c)

            if SAVE_BOOL:
                config.save_data(directory, file_name + '-batch' + str(batch + 1), config.seed, x, x_est, u, y, p, c)

            peni_controller.save_controller(directory, file_name + '-ctrl-' + str(batch + 1))

        if SAVE_BOOL:
            config.save_something(directory, file_name, 'actor-loss1', actor_loss)
            config.save_something(directory, file_name, 'critic-loss1', critic_loss)
    return peni_sysid


def offline_test(config, peni_sysid, file_name):
    file_name = file_name + '-offline'
    cost_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size)
    peni_controller = controller.CONTROLLER(peni_sysid, config)
    for k, batch in enumerate(OFFLINE_SAVE_INDEX):
        peni_controller.load_controller(directory, file_name + '-ctrl-' + str(batch))
        peni_sysid = peni_controller.controller.sysid  # must use loaded sysid... later change?
        peni_estimator = estimator.ESTIMATOR(sysid=peni_sysid, config=config)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(batch), config,
                                  peni_controller, peni_sysid, peni_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


def online_test(config, peni_sysid, file_name):
    file_name = file_name + '-online'
    cost_trajectory = np.zeros(ONLINE_SAVE_INDEX.size)
    peni_controller = controller.CONTROLLER(peni_sysid, config)
    for k, batch in enumerate(ONLINE_SAVE_INDEX):
        peni_controller.load_controller(directory, file_name + '-ctrl-' + str(batch))
        peni_sysid = peni_controller.controller.sysid  # must use loaded sysid... later change?
        peni_estimator = estimator.ESTIMATOR(sysid=peni_sysid, config=config)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(batch), config,
                                  peni_controller, peni_sysid, peni_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


if __name__ == '__main__':
    total_computation_time = time.time()
    execute_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = ut.make_directory('Data', 'peni', execute_time + '-' + CONTROL_METHOD + '-' + EXTRA_FILE_NAME)
    online_costs = np.zeros((TOTAL_RUN, ONLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)
    offline_costs = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)
    sysid_errors = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=NP_DATA_TYPE)

    config = PeniConfig()
    config.plot_bool = LEARNING_PLOT_BOOL
    config.sysid_method = SYSID_METHOD
    config.estimate_method = ESTIMATION_METHOD
    config.control_method = CONTROL_METHOD

    ''' Run the learning '''
    for seed in range(TOTAL_RUN):
        " Seeding "
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print('Current run is ' + str(seed))

        file_name = CONTROL_METHOD + '-' + SYSID_METHOD + '-' + ESTIMATION_METHOD
        file_name += '-SEED' + str(seed) + '-' + str(ONLINE_BATCH) + '-' + str(OFFLINE_BATCH)
        file_name += '-' + EXTRA_FILE_NAME
        config.seed = seed

        if SAVE_BOOL:
            config.save_settings(directory, file_name)

        # Learning
        if OFFLINE_BATCH > 0:
            learning_time = time.time()
            errors, peni_sysid = offline_learning(config, file_name)
            sysid_errors[seed, :] = errors
            offline_costs[seed, :] = offline_test(config, peni_sysid, file_name)
            print('Offline run time takes', str(np.round(time.time() - learning_time, 4)))

        # Online learning
        if ONLINE_BATCH > 0:
            learning_time2 = time.time()
            peni_sysid = online_learning(config, file_name)
            online_costs[seed, :] = online_test(config, peni_sysid, file_name)
            print('Online run time takes', str(np.round(time.time() - learning_time2, 4)))

        np.savetxt(fname=ONLINE_COST_FILE_NAME, X=online_costs, fmt='%12.8f')
        np.savetxt(fname=OFFLINE_COST_FILE_NAME, X=offline_costs, fmt='%12.8f')
        np.savetxt(fname=SYSID_ERR_FILE_NAME, X=sysid_errors, fmt='%12.8f')

    print('Total computation time:', np.around(time.time() - total_computation_time, 4))
    if ONLINE_BATCH > 0:
        print('final_online_cost', np.round(online_costs[:, -1], 4), np.round(np.mean(online_costs[:, -1]), 4))
    if OFFLINE_BATCH > 0:
        print('final_offline_cost', np.round(offline_costs[:, -1], 4), np.round(np.mean(offline_costs[:, -1]), 4))

