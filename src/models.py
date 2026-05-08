"""Model implementations: Statistical, ML, and Deep Learning."""

import os
import logging
import warnings
import numpy as np
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Deep learning backend detection ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — deep learning models will be skipped")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available — will be skipped")


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Models
# ═══════════════════════════════════════════════════════════════════════════════

class ARIMAModel:
    """ARIMA with AIC order selection on a short training window."""
    name = "ARIMA"

    def __init__(self, max_p=2, max_d=1, max_q=2):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.order = (1, 1, 1)

    def _select_order(self, series):
        from statsmodels.tsa.arima.model import ARIMA
        best_aic, best_order = np.inf, (1, 1, 1)
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        r = ARIMA(series, order=(p, d, q)).fit()
                        if r.aic < best_aic:
                            best_aic, best_order = r.aic, (p, d, q)
                    except Exception:
                        continue
        return best_order

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Select order on the last look_back window of training data (fast)
        ref = X_train[-1]
        self.order = self._select_order(ref)
        logger.info(f"ARIMA selected order: {self.order}")

    def predict(self, X_test, horizon=15):
        from statsmodels.tsa.arima.model import ARIMA
        preds = []
        for window in X_test:
            # Fit on just this look_back window — fast and local
            try:
                fit = ARIMA(window, order=self.order).fit()
                fc = fit.forecast(steps=horizon)
            except Exception:
                fc = np.full(horizon, window[-1])
            preds.append(fc)
        return np.array(preds)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"order": self.order}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.order = pickle.load(f)["order"]


class SARIMAModel:
    """SARIMA with fixed compact order — fitted per window."""
    name = "SARIMA"

    def __init__(self):
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 0, 1, 5)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass  # order is fixed; no global fit needed

    def predict(self, X_test, horizon=15):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        preds = []
        for window in X_test:
            try:
                fit = SARIMAX(window, order=self.order, seasonal_order=self.seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = fit.forecast(steps=horizon)
            except Exception:
                fc = np.full(horizon, window[-1])
            preds.append(fc)
        return np.array(preds)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"order": self.order, "seasonal_order": self.seasonal_order}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.order = d["order"]
        self.seasonal_order = d["seasonal_order"]


class HoltWintersModel:
    """Holt-Winters fitted per window."""
    name = "HoltWinters"

    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass  # fitted per window at predict time

    def predict(self, X_test, horizon=15):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        preds = []
        for window in X_test:
            try:
                fit = ExponentialSmoothing(
                    window, trend="add", seasonal=None, initialization_method="estimated"
                ).fit(optimized=True)
                fc = fit.forecast(horizon)
            except Exception:
                fc = np.full(horizon, window[-1])
            preds.append(fc)
        return np.array(preds)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({}, f)

    def load(self, path: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Machine Learning Models
# ═══════════════════════════════════════════════════════════════════════════════

def _make_lag_features(X: np.ndarray) -> np.ndarray:
    """Use raw look_back window as feature vector."""
    return X  # shape (n, look_back)


class RandomForestModel:
    """Multi-output Random Forest regressor."""
    name = "RandomForest"

    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.models = []  # one per horizon step

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestRegressor
        Xf = _make_lag_features(X_train)
        horizon = y_train.shape[1]
        self.models = []
        for h in range(horizon):
            rf = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=-1, random_state=42)
            rf.fit(Xf, y_train[:, h])
            self.models.append(rf)

    def predict(self, X_test, horizon=15):
        Xf = _make_lag_features(X_test)
        preds = np.stack([m.predict(Xf) for m in self.models], axis=1)
        return preds

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.models = pickle.load(f)


class SVRModel:
    """Multi-output SVR (direct strategy, one SVR per step)."""
    name = "SVR"

    def __init__(self):
        self.models = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.svm import SVR
        Xf = _make_lag_features(X_train)
        horizon = y_train.shape[1]
        self.models = []
        for h in range(horizon):
            svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
            svr.fit(Xf, y_train[:, h])
            self.models.append(svr)

    def predict(self, X_test, horizon=15):
        Xf = _make_lag_features(X_test)
        preds = np.stack([m.predict(Xf) for m in self.models], axis=1)
        return preds

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.models = pickle.load(f)


class XGBoostModel:
    """XGBoost multi-output (direct strategy)."""
    name = "XGBoost"

    def __init__(self):
        self.models = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        Xf = _make_lag_features(X_train)
        horizon = y_train.shape[1]
        self.models = []
        for h in range(horizon):
            m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  tree_method="hist", random_state=42, verbosity=0)
            m.fit(Xf, y_train[:, h])
            self.models.append(m)

    def predict(self, X_test, horizon=15):
        Xf = _make_lag_features(X_test)
        preds = np.stack([m.predict(Xf) for m in self.models], axis=1)
        return preds

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.models = pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Deep Learning Models (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class _GRUNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    class _TransformerNet(nn.Module):
        def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_len, horizon):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
            self.fc = nn.Linear(d_model, horizon)

        def forward(self, x):
            x = x.unsqueeze(-1)  # (B, T, 1)
            x = self.input_proj(x)
            x = self.encoder(x)
            return self.fc(x[:, -1, :])

    def _to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

    def _train_torch_model(net, X_train, y_train, X_val, y_val,
                            epochs=50, batch_size=64, lr=1e-3, patience=10):
        from torch.utils.data import DataLoader, TensorDataset
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        X_t = _to_tensor(X_train)
        y_t = _to_tensor(y_train)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val = np.inf
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            if X_val is not None and len(X_val) > 0:
                net.eval()
                with torch.no_grad():
                    val_pred = net(_to_tensor(X_val))
                    val_loss = criterion(val_pred, _to_tensor(y_val)).item()
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        return net

    class LSTMModel:
        name = "LSTM"

        def __init__(self, hidden_size=128, num_layers=2, dropout=0.2, horizon=15):
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.horizon = horizon
            self.net = None

        def fit(self, X_train, y_train, X_val=None, y_val=None):
            seq_len = X_train.shape[1]
            self.net = _LSTMNet(1, self.hidden_size, self.num_layers, self.dropout, self.horizon).to(DEVICE)
            X_tr = X_train[:, :, np.newaxis] if X_train.ndim == 2 else X_train
            X_v = X_val[:, :, np.newaxis] if X_val is not None and X_val.ndim == 2 else X_val
            _train_torch_model(self.net, X_tr, y_train, X_v, y_val)

        def predict(self, X_test, horizon=15):
            self.net.eval()
            X = X_test[:, :, np.newaxis] if X_test.ndim == 2 else X_test
            with torch.no_grad():
                out = self.net(_to_tensor(X)).cpu().numpy()
            return out

        def save(self, path: str):
            torch.save(self.net.state_dict(), path)

        def load(self, path: str):
            self.net.load_state_dict(torch.load(path, map_location=DEVICE))

    class GRUModel:
        name = "GRU"

        def __init__(self, hidden_size=128, num_layers=2, dropout=0.2, horizon=15):
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.horizon = horizon
            self.net = None

        def fit(self, X_train, y_train, X_val=None, y_val=None):
            self.net = _GRUNet(1, self.hidden_size, self.num_layers, self.dropout, self.horizon).to(DEVICE)
            X_tr = X_train[:, :, np.newaxis] if X_train.ndim == 2 else X_train
            X_v = X_val[:, :, np.newaxis] if X_val is not None and X_val.ndim == 2 else X_val
            _train_torch_model(self.net, X_tr, y_train, X_v, y_val)

        def predict(self, X_test, horizon=15):
            self.net.eval()
            X = X_test[:, :, np.newaxis] if X_test.ndim == 2 else X_test
            with torch.no_grad():
                out = self.net(_to_tensor(X)).cpu().numpy()
            return out

        def save(self, path: str):
            torch.save(self.net.state_dict(), path)

        def load(self, path: str):
            self.net.load_state_dict(torch.load(path, map_location=DEVICE))

    class TransformerModel:
        name = "Transformer"

        def __init__(self, d_model=64, nhead=4, num_encoder_layers=2,
                     dim_feedforward=256, dropout=0.1, horizon=15):
            self.d_model = d_model
            self.nhead = nhead
            self.num_encoder_layers = num_encoder_layers
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.horizon = horizon
            self.net = None

        def fit(self, X_train, y_train, X_val=None, y_val=None):
            seq_len = X_train.shape[1]
            self.net = _TransformerNet(self.d_model, self.nhead, self.num_encoder_layers,
                                       self.dim_feedforward, self.dropout, seq_len, self.horizon).to(DEVICE)
            _train_torch_model(self.net, X_train, y_train, X_val, y_val)

        def predict(self, X_test, horizon=15):
            self.net.eval()
            with torch.no_grad():
                out = self.net(_to_tensor(X_test)).cpu().numpy()
            return out

        def save(self, path: str):
            torch.save(self.net.state_dict(), path)

        def load(self, path: str):
            self.net.load_state_dict(torch.load(path, map_location=DEVICE))

else:
    # Stub classes when PyTorch is not available
    class _TorchStub:
        def fit(self, *a, **kw): raise ImportError("PyTorch not installed")
        def predict(self, *a, **kw): raise ImportError("PyTorch not installed")
        def save(self, *a, **kw): pass
        def load(self, *a, **kw): pass

    class LSTMModel(_TorchStub): name = "LSTM"
    class GRUModel(_TorchStub): name = "GRU"
    class TransformerModel(_TorchStub): name = "Transformer"


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_models(horizon: int = 15):
    """Return list of model instances."""
    models = [
        ARIMAModel(),
        SARIMAModel(),
        HoltWintersModel(),
        RandomForestModel(),
        SVRModel(),
        LSTMModel(horizon=horizon),
        GRUModel(horizon=horizon),
        TransformerModel(horizon=horizon),
    ]
    if XGB_AVAILABLE:
        models.insert(5, XGBoostModel())
    return models


def model_save_path(models_dir: str, model_name: str, ticker: str,
                    indicator: str, timeframe: str, period: str = "") -> str:
    tag = f"{model_name}_{ticker}_{indicator}_{timeframe}"
    if period:
        tag += f"_{period}"
    ext = ".pt" if model_name in ("LSTM", "GRU", "Transformer") else ".pkl"
    return str(Path(models_dir) / f"{tag}{ext}")
