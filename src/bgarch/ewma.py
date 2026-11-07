"""RiskMetrics-style EWMA benchmark, imported unchanged from src.real_backtest:
    h_t = lambda h_{t-1} + (1-lambda) r_{t-1}^2,  lambda = 0.94,
initialised at the sample variance of the first 100 returns of the full series
(2013 data here, so it cannot leak into any evaluation period)."""
from src.real_backtest import ewma_variance  # noqa: F401
