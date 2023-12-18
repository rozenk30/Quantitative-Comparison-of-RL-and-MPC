from Estimators import estimator_kf
from Estimators import estimator_lstm
from Estimators import estimator_stacking


class ESTIMATOR(object):
    def __init__(self, sysid, config):
        self.estimate_method = config.estimate_method
        self.sysid = sysid

        if self.estimate_method == 'KF':
            self.estimator = estimator_kf.KalmanFilter(self.sysid, config)

        elif self.estimate_method == 'LSTM_KF':
            self.estimator = estimator_lstm.LSTMEstimator(self.sysid, config)

        elif self.estimate_method == 'STACKING':
            self.estimator = estimator_stacking.StackingEstimator(self.sysid, config)

        else:
            print('No estimator')

    def estimate(self, x, u, y, *args):
        return self.estimator.estimate(x, u, y, *args)



