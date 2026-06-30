from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SignalPipeline:
    """Ridge baseline with optional LightGBM. Position sizing via z-score + clip + target vol."""

    def __init__(self, model: str = "ridge", **kwargs):
        self.model_name = model
        if model == "ridge":
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(**kwargs)),
            ])
        elif model == "lightgbm":
            import lightgbm as lgb
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("lgb", lgb.LGBMRegressor(n_estimators=100, verbose=-1, **kwargs)),
            ])
        else:
            raise ValueError(f"Unknown model: {model}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    @staticmethod
    def position_from_signal(signal: np.ndarray | pd.Series, target_vol: float = 0.10, clip_z: float = 2.0) -> pd.Series:
        """Z-score signal, clip, scale to target volatility."""
        sig = pd.Series(signal) if not isinstance(signal, pd.Series) else signal
        if sig.std() == 0:
            return pd.Series(0.0, index=sig.index)
        z = (sig - sig.mean()) / sig.std()
        z = z.clip(-clip_z, clip_z)
        pos = z * target_vol / (2.0 * z.std())
        return pos

if __name__ == "__main__":
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5))
    y = pd.Series(np.random.randn(200))
    sp = SignalPipeline("ridge")
    sp.fit(X, y)
    pred = sp.predict(X)
    pos = SignalPipeline.position_from_signal(pred)
    assert pos.abs().max() < 0.11, "position should be bounded by target_vol"
    assert len(pos) == 200
    # LightGBM path
    sp2 = SignalPipeline("lightgbm")
    sp2.fit(X, y)
    pred2 = sp2.predict(X)
    assert len(pred2) == 200