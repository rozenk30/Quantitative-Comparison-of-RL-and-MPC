import copy
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque


# scale X -> [0,1]
def zero_one_scale(var, var_min, var_max):
    scaled_var = (var - var_min) / (var_max - var_min)
    return scaled_var


# descale [0,1] -> X
def zero_one_descale(var, var_min, var_max):
    descaled_var = (var_max - var_min) * var + var_min
    return descaled_var


# scale X -> [-1,1]
def zero_mean_scale(var, var_min, var_max):
    scaled_var = (2*var - var_max - var_min) / (var_max - var_min)
    return scaled_var


# descale [-1,1] -> X
def zero_mean_descale(var, var_min, var_max):
    descaled_var = (var_max - var_min)/2*var + (var_max + var_min)/2
    return descaled_var


def min_max_relaxation(var_min, var_max, relax_coeff):
    center = (var_min + var_max)/2
    relaxed_min = center - relax_coeff*(center - var_min) - 10**(-6)*np.ones_like(var_min)
    relaxed_max = center + relax_coeff*(var_max - center) + 10**(-6)*np.ones_like(var_max)
    return relaxed_min, relaxed_max


# Make directory
def make_directory(*args):
    directory = Path.cwd()
    for arg in args:
        directory = Path.joinpath(directory, arg)
        Path(directory).mkdir(exist_ok=True)
    print('Directory:', directory, 'is made')
    return directory


def random_step_signal_generation(signal_dim, signal_length, signal_min, signal_max, signal_interval):
    random_signal = []
    while len(random_signal) < signal_length:
        signal_value = zero_one_descale(np.random.random(signal_dim), signal_min, signal_max)
        for k in range(signal_interval):
            random_signal.append(signal_value)
    random_signal = np.array(random_signal, dtype=np.float64)
    return random_signal[0:signal_length, :]


class ReplayBuffer(object):
    def __init__(self, buffer_size, seed=12345):
        self.seed = seed
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_backup = deque(maxlen=buffer_size)

    def add(self, x, u, c, xp, is_terminal):
        self.buffer.append([x, u, c, xp, is_terminal])

    def reset(self):
        self.buffer.clear()
        self.buffer_backup.clear()

    def sample(self, batch_size):
        # Sample as tensor variables
        batch = random.sample(self.buffer, k=min(len(self.buffer), batch_size))
        x_batch = tf.constant([k[0] for k in batch], dtype=tf.float64)
        u_batch = tf.constant([k[1] for k in batch], dtype=tf.float64)
        c_batch = tf.constant([k[2] for k in batch], dtype=tf.float64)
        xp_batch = tf.constant([k[3] for k in batch], dtype=tf.float64)
        is_terminal_batch = tf.constant([k[4] for k in batch], dtype=tf.bool)
        return x_batch, u_batch, c_batch, xp_batch, is_terminal_batch

    def scale_the_buffer(self, x_min, x_max, u_min, u_max):
        self.buffer_backup = copy.deepcopy(self.buffer)
        for k in range(len(self.buffer)):
            self.buffer[k][0] = zero_mean_scale(self.buffer[k][0], x_min, x_max)
            self.buffer[k][1] = zero_mean_scale(self.buffer[k][1], u_min, u_max)
            self.buffer[k][3] = zero_mean_scale(self.buffer[k][3], x_min, x_max)

    def call_buffer_backup(self):
        self.buffer = copy.deepcopy(self.buffer_backup)
