import random
import numpy as np
import tensorflow as tf
from Systems import sys_cstr


def test(directory, file_name, config, controller1, controller2, sysid, estimator, plot_bool, save_bool):
    config.system_state_disturb = False
    config.system_measure_disturb = True
    config.system_para_disturb = True
    np_data_type = np.float64
    test_seed = config.seed + 12345

    """ Seeding for test """
    np.random.seed(test_seed)
    tf.random.set_seed(test_seed)
    random.seed(test_seed)

    sim_horizon = 60*3
    sim_ini_horizon = 20

    # Plant setting
    plant = sys_cstr.SysCSTR(config)

    """ Simulation with control """
    x = np.zeros((sim_horizon, plant.x_dim), dtype=np_data_type)
    y = np.zeros((sim_horizon, plant.y_dim), dtype=np_data_type)
    u = np.zeros((sim_horizon, plant.u_dim), dtype=np_data_type)
    p = np.zeros((sim_horizon, plant.p_dim), dtype=np_data_type)
    r = np.zeros((sim_horizon, plant.r_dim), dtype=np_data_type)
    c = np.zeros((sim_horizon, 1), dtype=np_data_type)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    x_est = np.zeros((sim_horizon, sysid.x_est_dim), dtype=np_data_type)

    for k in range(sim_horizon - 1):
        if np.mod(np.floor(k/60), 2) == 0:
            r[k, :] = plant.ref1
        else:
            r[k, :] = plant.ref2

        if k < sim_ini_horizon:
            u[k, :] = plant.ss_u
        elif np.mod(np.floor(k/60), 2) == 0:
            u[k, :] = controller1.control_without_exploration(x_est[k, :])
        else:
            u[k, :] = controller2.control_without_exploration(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_temperature()
        x[k+1, :] = plant.go_step(x[k, :], u[k, :])
        y[k+1, :] = plant.get_observation(x[k+1, :])
        x_est[k+1, :] = estimator.estimate(x_est[k, :], u[k, :], y[k+1, :])

    r[-1, :] = plant.ref1
    u[-1, :] = controller1.control_without_exploration(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_temperature()

    print('CSTR Test is done, total cost: ', np.round(np.sum(c), 4))

    if plot_bool:
        config.plot_data(sim_horizon, u, y, r, c)

    if save_bool:
        config.save_data(directory, file_name, config.seed, x, x_est, u, y, p, r, c)

    return np.sum(c)
