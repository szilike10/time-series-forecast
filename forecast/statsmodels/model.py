from forecast.model import ForecastingModel
from forecast.statsmodels.statsmodels_config import StatsmodelsConfig


class StatModel(ForecastingModel):
    def __init__(self, cfg: StatsmodelsConfig):
        super().__init__()

        self.cfg = cfg

        self.p = 0
        self.d = 0
        self.q = 0
        self.P = 0
        self.D = 0
        self.Q = 0
        self.S = 0

        self.model = None

    def fit(self, *args, **kwargs):
        pass
