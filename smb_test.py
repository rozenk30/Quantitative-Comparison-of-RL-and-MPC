import random
import numpy as np
import tensorflow as tf
from Systems import sys_smb


def test(directory, file_name, config, controller, sysid, estimator, plot_bool, save_bool):
    config.system_state_disturb = False
    config.system_measure_disturb = False
    config.system_para_disturb = True
    np_data_type = np.float64
    test_seed = config.seed + 12345

    """ Seeding for test """
    np.random.seed(test_seed)
    tf.random.set_seed(test_seed)
    random.seed(test_seed)

    sim_horizon = 32*6
    sim_ini_horizon = 32*1

    # Plant setting
    plant = sys_smb.SysSMB(config)

    """ Simulation with control """
    x = np.zeros((sim_horizon, plant.x_dim), dtype=np_data_type)
    y = np.zeros((sim_horizon, plant.y_dim), dtype=np_data_type)
    u = np.zeros((sim_horizon, plant.u_dim), dtype=np_data_type)
    p = np.zeros((sim_horizon, plant.p_dim), dtype=np_data_type)
    r = np.zeros((sim_horizon, plant.r_dim), dtype=np_data_type)
    c = np.zeros((sim_horizon, 1), dtype=np_data_type)
    et = np.zeros((sim_horizon, 2), dtype=np_data_type)
    rt = np.zeros((sim_horizon, 2), dtype=np_data_type)
    x[0, :], u[0, :], y[0, :], p[0, :], r[0, :] = plant.do_reset()

    x_est = np.zeros((sim_horizon, sysid.x_est_dim), dtype=np_data_type)

    for k in range(sim_horizon - 1):
        r[k, :] = plant.ref
        if k < sim_ini_horizon:
            u[k, :] = plant.ss_u
        else:
            u[k, :] = controller.control_without_exploration(x_est[k, :])
        c[k, :] = plant.get_cost(y[k, :], u[k, :], r[k, :])
        p[k, :] = plant.get_feed_concentration()
        et[k, :], rt[k, :] = plant.get_tank_information()
        x[k + 1, :] = plant.go_step(x[k, :], u[k, :])
        y[k + 1, :] = plant.get_observation(x[k+1, :], u[k, :])
        mode = plant.get_mode(x[k + 1, :])

        x_est[k+1, :] = estimator.estimate(x_est[k, :], u[k, :], y[k+1, :], mode)

    r[-1, :] = plant.ref
    u[-1, :] = controller.control_without_exploration(x_est[-1, :])
    c[-1, :] = plant.get_cost(y[-1, :], u[-1, :], r[-1, :])
    p[-1, :] = plant.get_feed_concentration()
    et[-1, :], rt[-1, :] = plant.get_tank_information()

    print('SMB Test is done, total cost: ', np.round(np.sum(c), 4))

    if plot_bool:
        config.plot_data(sim_horizon, u, y, r, c, et, rt)

    if save_bool:
        config.save_data(directory, file_name, config.seed, x, x_est, u, y, p, r, c)

    return np.sum(c)
