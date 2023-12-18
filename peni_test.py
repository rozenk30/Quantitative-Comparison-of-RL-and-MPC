import random
import numpy as np
import tensorflow as tf
from Systems import sys_peni_with_PID


def test(directory, file_name, config, controller, sysid, estimator, plot_bool, save_bool):
    config.system_state_disturb = False
    config.system_measure_disturb = False
    config.system_para_disturb = False
    np_data_type = np.float32
    test_seed = config.seed + 12345

    """ Seeding for test """
    np.random.seed(test_seed)
    tf.random.set_seed(test_seed)
    random.seed(test_seed)

    # Plant setting
    plant = sys_peni_with_PID.SysPeniWithPID(config, test_seed)
    sim_horizon = plant.total_horizon

    """ Simulation with control """
    x = np.zeros((sim_horizon + 1, plant.x_dim), dtype=np_data_type)
    y = np.zeros((sim_horizon + 1, plant.y_dim), dtype=np_data_type)
    u = np.zeros((sim_horizon + 1, plant.u_dim), dtype=np_data_type)
    p = np.zeros((sim_horizon + 1, plant.p_dim), dtype=np_data_type)
    c = np.zeros((sim_horizon + 1, 1), dtype=np_data_type)
    x[0, :], u[0, :], y[0, :], p[0, :] = plant.do_reset()

    x_est = np.zeros((sim_horizon + 1, sysid.x_est_dim), dtype=np_data_type)

    for k in range(sim_horizon):
        import time
        mpc_time_check = time.time()

        if k < 10:  # fixing initial control
            u[k, :] = plant.get_reference_input(k)
        else:
            if config.control_method == 'MPC':  # ###
                controller.controller.prediction_horizon = sim_horizon - k
            u[k, :] = controller.control_without_exploration(x_est[k, :])

        # print(k, u[k, :], x_est[k, :], time.time() - mpc_time_check)
        c[k, :] = plant.get_cost(y[k, :], u[k, :])
        x[k+1, :] = plant.go_step(x[k, :], u[k, :])
        y[k+1, :] = plant.get_observation(x[k+1, :])
        x_est[k+1, :] = estimator.estimate(x_est[k, :], u[k, :], y[k+1, :])
    u[-1, :] = plant.get_reference_input(k + 1)
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :])

    print('Simulation with control is done, total cost: ', np.round(np.sum(c), 4))

    if plot_bool:
        config.plot_data(sim_horizon + 1, u, y, c)

    if save_bool:
        config.save_data(directory, file_name, config.seed, x, x_est, u, y, p, c)

    return np.sum(c)
