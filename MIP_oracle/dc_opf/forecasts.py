import numpy as np


class Forecasts:
    def __init__(self, env, t=0, horizon=2):
        self.t = t
        self.horizon = horizon
        self.time_steps = np.arange(self.horizon)

        self.env = env
        self.data = self._get_chronic_data()

    @property
    def load_p(self):
        return self.data.load_p[self.t : self.t + self.horizon, :]

    @property
    def prod_p(self):
        return self.data.prod_p[self.t : self.t + self.horizon, :]

    def _get_chronic_data(self):
        return self.env.chronics_handler.real_data.data


class ForecastsPlain:
    def __init__(self, env, t=0, horizon=2):
        self.t = t
        self.horizon = horizon
        self.time_steps = np.arange(self.horizon)

        self.env = env

    def __bool__(self):
        return False
