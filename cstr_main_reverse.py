import os
import time
import random
import numpy as np
import tensorflow as tf

from Utility import utility as ut
from Systems import sys_cstr
from Sysids import sysid
from Estimators import estimator
from Controls import controller
from cstr_config import CstrConfig
from cstr_test import test
from datetime import datetime

# GPU on/off
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TOTAL_RUN = 10
ONLINE_HOUR = 1001*1  # Hour
OFFLINE_HOUR = 1000*0  # Hour
# ONLINE_SAVE_INTERVAL = 100
# OFFLINE_SAVE_INTERVAL = 100
# ONLINE_SAVE_INDEX = np.arange(ONLINE_SAVE_INTERVAL, ONLINE_HOUR, ONLINE_SAVE_INTERVAL).astype('int')
# OFFLINE_SAVE_INDEX = np.arange(OFFLINE_SAVE_INTERVAL, OFFLINE_HOUR, OFFLINE_SAVE_INTERVAL).astype('int')
ONLINE_SAVE_INDEX = np.array([5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)
OFFLINE_SAVE_INDEX = np.array([5, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=int)

SYSID_METHOD = 'N4SID'  # 'N4SID', 'LSTM', 'NNARX', 'STACKING'
ESTIMATION_METHOD = 'KF'  # 'KF', 'LSTM_KF', 'STACKING', 'RL_STACKING'
CONTROL_METHOD = 'MPC'  # 'QMPC', 'MPC', 'DDPG', 'TD3', 'SAC'
EXTRA_FILE_NAME = 'Linear-2-on'

ONLINE_COST_FILE_NAME = 'cstr_results_online_cost_1.txt'
OFFLINE_COST_FILE_NAME = 'cstr_results_offline_cost_1.txt'
SYSID_ERR_FILE_NAME = 'cstr_results_sysid_errors_1.txt'

LEARNING_PLOT_BOOL = False
TEST_PLOT_BOOL = False
SAVE_BOOL = True

NP_DATA_TYPE = np.float64
TF_DATA_TYPE = tf.float64


def offline_learning(config1, config2, file_name):
    learning_offline_ini_horizon = 1  # int(0.2*OFFLINE_HOUR)
    file_name += '-offline'
    sysid_err_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size, dtype=NP_DATA_TYPE)

    """ Set up PLANT """
    plant = sys_cstr.SysCSTR(config1)

    """ Set up SYSID """
    cstr_sysid = sysid.SYSID(plant=plant, config=config1)

    """ Offline simulation """
    x = np.zeros((60*OFFLINE_HOUR + 1, plant.x_dim), dtype=NP_DATA_TYPE)
    y = np.zeros((60*OFFLINE_HOUR + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    u = np.zeros((60*OFFLINE_HOUR + 1, plant.u_dim), dtype=NP_DATA_TYPE)
    p = np.zeros((60*OFFLINE_HOUR + 1, plant.p_dim), dtype=NP_DATA_TYPE)
    c = np.zeros((60*OFFLINE_HOUR + 1, 1), dtype=NP_DATA_TYPE)
    r = np.zeros((60*OFFLINE_HOUR + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    # Generating random signal
    random_signal = ut.random_step_signal_generation(signal_dim=plant.u_dim,
                                                     signal_length=config1.SYSID_signal_length,
                                                     signal_min=config1.SYSID_signal_min,
                                                     signal_max=config1.SYSID_signal_max,
                                                     signal_interval=config1.SYSID_signal_interval)

    # Pre-simulation for System Identification or Learning
    for k in range(60*OFFLINE_HOUR):
        if np.mod(np.floor(k / 60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2
        if k > 60*learning_offline_ini_horizon:
            u[k, :] = random_signal[k, :]  # Random signal
        else:
            u[k, :] = plant.ss_u  # Steady state input
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()  # Disturbance generation
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
    r[-1, :] = r[-2, :]
    u[-1, :] = u[-2, :]
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.p_now

    if LEARNING_PLOT_BOOL:
        print('Plot offline signal simulation')
        config1.plot_data(60*OFFLINE_HOUR + 1, u, y, r, c)

    if SAVE_BOOL:
        config1.save_data(directory, file_name, config1.seed, x, x, u, y, p, r, c)

    """ Put data into the sysid """
    cstr_sysid.add_data_and_scale(u, y)

    for k, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        print('seed:', config1.seed, 'current offline step:', operated_time)
        cstr_sysid.do_identification(60*operated_time)
        sysid_err_trajectory[k] = cstr_sysid.sysid_error
        controller1 = controller.CONTROLLER(cstr_sysid, config1)
        controller2 = controller.CONTROLLER(cstr_sysid, config2)
        controller1.save_controller(directory, file_name + '-ctrl1-' + str(operated_time))
        controller2.save_controller(directory, file_name + '-ctrl2-' + str(operated_time))
    return sysid_err_trajectory, cstr_sysid


def online_learning(config1, config2, file_name):
    learning_online_ini_horizon = 1
    learning_online_ini_horizon2 = 5

    file_name += '-online'

    """ Set up plant """
    plant = sys_cstr.SysCSTR(config1)

    """ Set up sysid """
    cstr_sysid = sysid.SYSID(plant=plant, config=config1)

    """ Online Simulation with control """
    x = np.zeros((60*ONLINE_HOUR + 1, plant.x_dim), dtype=NP_DATA_TYPE)
    y = np.zeros((60*ONLINE_HOUR + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    u = np.zeros((60*ONLINE_HOUR + 1, plant.u_dim), dtype=NP_DATA_TYPE)
    p = np.zeros((60*ONLINE_HOUR + 1, plant.p_dim), dtype=NP_DATA_TYPE)
    c = np.zeros((60*ONLINE_HOUR + 1, 1), dtype=NP_DATA_TYPE)
    r = np.zeros((60*ONLINE_HOUR + 1, plant.y_dim), dtype=NP_DATA_TYPE)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()
    x_est = np.zeros((60*ONLINE_HOUR + 1, cstr_sysid.x_est_dim), dtype=NP_DATA_TYPE)

    # Generating random signal
    random_signal = ut.random_step_signal_generation(signal_dim=plant.u_dim,
                                                     signal_length=config1.SYSID_signal_length,
                                                     signal_min=config1.SYSID_signal_min,
                                                     signal_max=config1.SYSID_signal_max,
                                                     signal_interval=config1.SYSID_signal_interval)

    for k in range(60*ONLINE_HOUR):
        if np.mod(np.floor(k/60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2

        if k < 60*learning_online_ini_horizon:
            u[k, :] = plant.ss_u
        elif k < 60*learning_online_ini_horizon2:
            u[k, :] = random_signal[k, :]  # Random signal
        elif np.mod(np.floor(k/60), 2) == 0:
            u[k, :] = controller1.control(x_est[k, :])
        else:
            u[k, :] = controller2.control(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k + 1, :])
        if k >= 60*learning_online_ini_horizon2:
            x_est[k + 1, :] = cstr_estimator.estimate(x_est[k, :], u[k, :], y[k + 1, :])

        # Controller learning
        if int(k + 1) in 60*ONLINE_SAVE_INDEX:
            print('seed:', seed, 'current online step:', int((k + 1)/60))
            config1.save_data(directory, file_name, config1.seed, x, x, u, y, p, r, c)
            cstr_sysid = sysid.SYSID(plant=plant, config=config1)
            cstr_sysid.add_data_and_scale(u, y)
            cstr_sysid.do_identification(k)
            cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)
            controller1 = controller.CONTROLLER(cstr_sysid, config1)
            controller2 = controller.CONTROLLER(cstr_sysid, config2)
            controller1.save_controller(directory, file_name + '-ctrl1-' + str(int((k + 1)/60)))
            controller2.save_controller(directory, file_name + '-ctrl2-' + str(int((k + 1)/60)))
    r[-1, :] = r[-2, :]
    u[-1, :] = controller1.control_without_exploration(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()

    if LEARNING_PLOT_BOOL:
        config1.plot_data(60*ONLINE_HOUR + 1, u, y, r, c)

    if SAVE_BOOL:
        config1.save_data(directory, file_name, seed, x, x_est, u, y, p, r, c)

    return cstr_sysid


def offline_test(config1, config2, cstr_sysid, file_name):
    file_name += '-offline'
    cost_trajectory = np.zeros(OFFLINE_SAVE_INDEX.size)
    controller1 = controller.CONTROLLER(cstr_sysid, config1)
    controller2 = controller.CONTROLLER(cstr_sysid, config2)
    for k, operated_time in enumerate(OFFLINE_SAVE_INDEX):
        controller1.load_controller(directory, file_name + '-ctrl1-' + str(operated_time))
        controller2.load_controller(directory, file_name + '-ctrl2-' + str(operated_time))
        cstr_sysid = controller1.controller.sysid  # must use loaded sysid... later change?
        cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(operated_time), config1,
                                  controller1, controller2, cstr_sysid, cstr_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


def online_test(config1, config2, cstr_sysid, file_name):
    file_name = file_name + '-online'
    cost_trajectory = np.zeros(ONLINE_SAVE_INDEX.size)
    controller1 = controller.CONTROLLER(cstr_sysid, config1)
    controller2 = controller.CONTROLLER(cstr_sysid, config2)
    for k, operated_time in enumerate(ONLINE_SAVE_INDEX):
        controller1.load_controller(directory, file_name + '-ctrl1-' + str(operated_time))
        controller2.load_controller(directory, file_name + '-ctrl2-' + str(operated_time))
        cstr_sysid = controller1.controller.sysid  # must use loaded sysid... later change?
        cstr_estimator = estimator.ESTIMATOR(sysid=cstr_sysid, config=config1)
        cost_trajectory[k] = test(directory, file_name + '-test-' + str(operated_time), config1,
                                  controller1, controller2, cstr_sysid, cstr_estimator, TEST_PLOT_BOOL, SAVE_BOOL)
    return cost_trajectory


if __name__ == '__main__':
    total_computation_time = time.time()
    execute_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = ut.make_directory('Data', 'cstr', execute_time + '-' + CONTROL_METHOD + '-' + EXTRA_FILE_NAME)

    online_costs = np.zeros((TOTAL_RUN, ONLINE_SAVE_INDEX.size), dtype=np.float64)
    offline_costs = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=np.float64)
    sysid_errors = np.zeros((TOTAL_RUN, OFFLINE_SAVE_INDEX.size), dtype=np.float64)

    config1 = CstrConfig()
    config1.plot_bool = LEARNING_PLOT_BOOL
    config1.ref = np.array([1.09, 114.2])
    config1.sysid_method = SYSID_METHOD
    config1.estimate_method = ESTIMATION_METHOD
    config1.control_method = CONTROL_METHOD

    config2 = CstrConfig()
    config2.plot_bool = LEARNING_PLOT_BOOL
    config2.ref = np.array([1.04, 94.2])
    config2.QMPC_prediction_horizon = 2
    config2.sysid_method = SYSID_METHOD
    config2.estimate_method = ESTIMATION_METHOD
    config2.control_method = CONTROL_METHOD

    for seed in range(TOTAL_RUN):
        " Seeding "
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print('Current run is ' + str(seed))

        file_name = CONTROL_METHOD + '-' + SYSID_METHOD + '-' + ESTIMATION_METHOD
        file_name += '-SEED' + str(seed) + '-' + str(ONLINE_HOUR) + '-' + str(OFFLINE_HOUR)
        file_name += '-' + EXTRA_FILE_NAME
        config1.seed, config2.seed = seed, seed

        if SAVE_BOOL:
            config1.save_settings(directory, file_name + '-1')
            config2.save_settings(directory, file_name + '-2')

        # Offline learning
        if OFFLINE_HOUR > 0:
            learning_time = time.time()
            errors, cstr_sysid = offline_learning(config1, config2, file_name)
            sysid_errors[seed, :] = errors
            offline_costs[seed, :] = offline_test(config1, config2, cstr_sysid, file_name)
            print('Offline run time takes', str(np.round(time.time() - learning_time, 4)))

        # Online learning
        if ONLINE_HOUR > 0:
            learning_time2 = time.time()
            cstr_sysid = online_learning(config1, config2, file_name)
            online_costs[seed, :] = online_test(config1, config2, cstr_sysid, file_name)
            print('Online run time takes', str(np.round(time.time() - learning_time2, 4)))

        np.savetxt(fname=ONLINE_COST_FILE_NAME, X=online_costs, fmt='%12.8f')
        np.savetxt(fname=OFFLINE_COST_FILE_NAME, X=offline_costs, fmt='%12.8f')
        np.savetxt(fname=SYSID_ERR_FILE_NAME, X=sysid_errors, fmt='%12.8f')

    print('Total computation time:', np.around(time.time() - total_computation_time, 4))
    if ONLINE_HOUR > 0:
        print('final_online_cost', np.round(online_costs[:, -1], 4), np.round(np.mean(online_costs[:, -1]), 4))
    if OFFLINE_HOUR > 0:
        print('final_offline_cost', np.round(offline_costs[:, -1], 4), np.round(np.mean(offline_costs[:, -1]), 4))

