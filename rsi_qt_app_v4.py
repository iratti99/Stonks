import sys
import os
from datetime import datetime, time, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# -------------------- Zona horaria Argentina --------------------
try:
    from zoneinfo import ZoneInfo
    TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")
except Exception:
    TZ_AR = None

WALLET_PATH = "wallet.txt"  # ajustá si querés ruta absoluta


# ============================ Wallet ============================ #
def load_wallet(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    part = txt.split(';', 1)[0]
    raw = part.split(',')
    tickers, seen = [], set()
    for r in raw:
        t = r.strip().upper()
        if t and t not in seen:
            tickers.append(t); seen.add(t)
    return tickers


# ============================ RSI utils ============================ #
def compute_rsi(series_close: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series_close, errors="coerce").astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_trend(rsi: pd.Series, lookback: int = 10):
    s = rsi.dropna()
    if len(s) < lookback: return np.nan, np.nan
    y = s.iloc[-lookback:].to_numpy(dtype=float)
    x = np.arange(len(y))
    x_m, y_m = x.mean(), y.mean()
    b1 = np.sum((x-x_m)*(y-y_m))/np.sum((x-x_m)**2)
    y_hat = (y_m - b1*x_m) + b1*x
    ss_res = np.sum((y - y_hat)**2); ss_tot = np.sum((y - y_m)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(b1), float(r2)


# ============================ Proveedores ============================ #
def fetch_yahoo(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol, interval=interval, period=period,
        progress=False, auto_adjust=False, prepost=True, threads=True
    )
    if df is None or df.empty:
        raise ValueError("Yahoo no devolvió datos.")
    return df.sort_index()


def fetch_investing(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Experimental: requiere `investpy`. No todos los símbolos funcionarán igual."""
    try:
        import investpy
    except Exception:
        raise RuntimeError("Para usar Investing instalá 'investpy' (experimental).")
    per_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
    n_days = per_map.get(period, 90)
    q = investpy.search_quotes(text=symbol, products=['stocks'],
                               countries=['argentina', 'united states'], n_results=1)
    if not q: raise ValueError("Investing no encontró el símbolo.")
    hist = q[0].historical_data(
        from_date=(datetime.now()-pd.Timedelta(days=n_days)).strftime("%d/%m/%Y"),
        to_date=datetime.now().strftime("%d/%m/%Y")
    )
    df = pd.DataFrame(hist)
    if df.empty: raise ValueError("Investing devolvió vacío.")
    df.index = pd.to_datetime(df.index)
    df.rename(columns={"Close":"Close"}, inplace=True)
    return df.sort_index()


# ============================ USD rates ============================ #
def _fmt_ars(v: float) -> str:
    return f"${v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fetch_usd_rates() -> Dict[str, float]:
    """
    Devuelve {'CCL': x, 'MEP': y, 'OFICIAL': z} en ARS/USD.
    Intenta lista general + endpoints individuales; usa CriptoYa como respaldo.
    """
    rates: Dict[str, float] = {}

    # 1) Lista general
    try:
        r = requests.get("https://dolarapi.com/v1/dolares", timeout=5)
        if r.ok:
            data = r.json()
            idx = {item.get("casa","").lower(): item for item in data if isinstance(item, dict)}
            if "oficial" in idx: rates["OFICIAL"] = float(idx["oficial"]["venta"])
            if "mep" in idx:     rates["MEP"]     = float(idx["mep"]["venta"])
            if "contadoconliqui" in idx: rates["CCL"] = float(idx["contadoconliqui"]["venta"])
    except Exception:
        pass

    # 2) Endpoints individuales (por si alguno faltó)
    try:
        if "MEP" not in rates:
            r = requests.get("https://dolarapi.com/v1/dolares/mep", timeout=5)
            if r.ok: rates["MEP"] = float(r.json().get("venta", 0) or 0)
    except Exception:
        pass
    try:
        if "CCL" not in rates:
            r = requests.get("https://dolarapi.com/v1/dolares/contadoconliqui", timeout=5)
            if r.ok: rates["CCL"] = float(r.json().get("venta", 0) or 0)
    except Exception:
        pass
    try:
        if "OFICIAL" not in rates:
            r = requests.get("https://dolarapi.com/v1/dolares/oficial", timeout=5)
            if r.ok: rates["OFICIAL"] = float(r.json().get("venta", 0) or 0)
    except Exception:
        pass

    # 3) Respaldo CriptoYa
    if len([k for k in rates if rates.get(k, 0) > 0]) < 3:
        try:
            r = requests.get("https://criptoya.com/api/dolar", timeout=5)
            if r.ok:
                d = r.json()
                rates.setdefault("OFICIAL", float(d.get("oficial", 0) or 0))
                rates.setdefault("MEP", float(d.get("mep", 0) or 0))
                rates.setdefault("CCL", float(d.get("ccl", 0) or 0))
        except Exception:
            pass

    return {k: float(v) for k, v in rates.items() if v and v > 0}


def pick_max_usd(rates: Dict[str, float]) -> Tuple[str, float]:
    if not rates: return "N/A", 0.0
    k = max(rates, key=rates.get)
    return k, float(rates[k])


# ============================ Worker ============================ #
class DataWorker(QThread):
    data_ready = pyqtSignal(object, str, float, str)  # df, name, dl_ms, currency
    error = pyqtSignal(str)

    def __init__(self, symbol: str, interval: str, period: str, provider: str, parent=None):
        super().__init__(parent)
        import time
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.provider = provider
        self._time = time

    def run(self):
        try:
            t0 = self._time.perf_counter()
            if self.provider == "Yahoo":
                df = fetch_yahoo(self.symbol, self.interval, self.period)
            else:
                df = fetch_investing(self.symbol, self.interval, self.period)
            t1 = self._time.perf_counter()
            dl_ms = (t1 - t0) * 1000.0

            # Info (Yahoo si es posible)
            name, currency = self.symbol, "N/A"
            try:
                tk = yf.Ticker(self.symbol)
                info = {}
                try: info = tk.get_info() or {}
                except Exception: info = getattr(tk, "info", {}) or {}
                name = info.get("longName") or info.get("shortName") or info.get("displayName") or info.get("symbol") or self.symbol
                currency = (info.get("currency") or "N/A").upper()
            except Exception:
                pass

            self.data_ready.emit(df, name, float(dl_ms), currency)
        except Exception as e:
            self.error.emit(f"Error descargando datos: {e}")


# ============================ Canvases ============================ #
class BaseCanvas(FigureCanvas):
    def __init__(self, title: str, ylim=None, parent=None):
        fig = Figure(figsize=(8, 3.6), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.title = title
        self.ylim = ylim

    def _prep(self):
        self.ax.clear()
        self.ax.set_title(self.title)
        if self.ylim:
            self.ax.set_ylim(*self.ylim)
        self.ax.grid(True, which="both", linestyle="--", alpha=0.3)

    @staticmethod
    def _annotate_min_max(ax, x, y, fmt="{:.2f}", yoffset=0.0):
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.size == 0 or np.all(np.isnan(y_arr)):
            return

        if isinstance(x, (pd.DatetimeIndex, pd.Series)):
            idx = pd.DatetimeIndex(x)
            try:
                if idx.tz is not None and TZ_AR is not None:
                    idx = idx.tz_convert(TZ_AR).tz_localize(None)
            except Exception:
                pass
            x_list = list(idx.to_pydatetime())
        else:
            x_list = list(x)
        if len(x_list) != y_arr.size:
            return

        i_min = int(np.nanargmin(y_arr))
        i_max = int(np.nanargmax(y_arr))
        vmin = float(y_arr[i_min])
        vmax = float(y_arr[i_max])
        ax.scatter([x_list[i_min]], [vmin], s=40, color="green", zorder=5)
        ax.scatter([x_list[i_max]], [vmax], s=40, color="red", zorder=5)
        ax.annotate(fmt.format(vmin), xy=(x_list[i_min], vmin),
                    xytext=(0, -12 + yoffset), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="green",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", lw=0.6, alpha=0.8))
        ax.annotate(fmt.format(vmax), xy=(x_list[i_max], vmax),
                    xytext=(0, 12 + yoffset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=0.6, alpha=0.8))

    # --------- Modelos / proyección ---------
    @staticmethod
    def _forecast_lineal(y: np.ndarray, steps: int, window: int, clip=None):
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size < 10:
            return None, None
        w = min(window, y.size)
        t = np.arange(w)
        y_win = y[-w:]
        b, a = np.polyfit(t, y_win, 1)
        y_fit = a + b * t
        t_future = np.arange(w, w + steps)
        y_future = a + b * t_future
        if clip is not None:
            lo, hi = clip
            y_fit = np.clip(y_fit, lo, hi)
            y_future = np.clip(y_future, lo, hi)
        return y_fit, y_future

    @staticmethod
    def _forecast_holtwinters(y: np.ndarray, steps: int, window: int, clip=None):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except Exception:
            return None, None
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size < 15:
            return None, None
        w = min(window, y.size)
        y_win = y[-w:]
        try:
            model = ExponentialSmoothing(y_win, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            y_fit = np.asarray(fit.fittedvalues)
            y_future = np.asarray(fit.forecast(steps))
            if clip is not None:
                lo, hi = clip
                y_fit = np.clip(y_fit, lo, hi)
                y_future = np.clip(y_future, lo, hi)
            return y_fit, y_future
        except Exception:
            return None, None

    @staticmethod
    def _forecast_arima(y: np.ndarray, steps: int, window: int, clip=None):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except Exception:
            return None, None
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size < 20:
            return None, None
        w = min(window, y.size)
        y_win = y[-w:]
        for order in [(1,1,1),(1,0,0),(0,1,1)]:
            try:
                fit = ARIMA(y_win, order=order).fit(method_kwargs={"warn_convergence": False})
                y_fit = np.asarray(fit.predict(start=0, end=len(y_win)-1))
                y_future = np.asarray(fit.forecast(steps))
                if clip is not None:
                    lo, hi = clip
                    y_fit = np.clip(y_fit, lo, hi)
                    y_future = np.clip(y_future, lo, hi)
                return y_fit, y_future
            except Exception:
                continue
        return None, None

    @staticmethod
    def _future_index(idx: pd.DatetimeIndex, steps: int):
        if steps <= 0 or idx.size < 2:
            return []
        base = pd.DatetimeIndex(idx)
        try:
            if base.tz is not None and TZ_AR is not None:
                base = base.tz_convert(TZ_AR).tz_localize(None)
        except Exception:
            pass
        last = base[-1]
        delta = base[-1] - base[-2]
        if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
            delta = pd.Timedelta(minutes=1)
        return [last + (i+1)*delta for i in range(steps)]

    def _plot_with_model(self, x_obs, y_obs, idx, model_name: str, steps: int,
                         window: int, clip=None, show_model=True):
        # Observado
        self.ax.plot(x_obs, y_obs, linewidth=1.5)
        if not show_model or steps <= 0:
            return
        # Ajuste + proyección
        y_fit = y_future = None
        if model_name == "Lineal":
            y_fit, y_future = self._forecast_lineal(np.array(y_obs), steps, window, clip=clip)
        elif model_name == "Holt-Winters":
            y_fit, y_future = self._forecast_holtwinters(np.array(y_obs), steps, window, clip=clip)
        elif model_name == "ARIMA":
            y_fit, y_future = self._forecast_arima(np.array(y_obs), steps, window, clip=clip)

        if y_fit is not None:
            x_fit = x_obs[-len(y_fit):]
            self.ax.plot(x_fit, list(y_fit), linestyle="--", linewidth=1.2)

        if y_future is not None:
            fut_idx = self._future_index(pd.DatetimeIndex(x_obs), steps)
            if fut_idx:
                yf_list = list(y_future)
                if len(yf_list) == 1:
                    self.ax.plot([x_obs[-1], fut_idx[0]], [y_obs[-1], float(yf_list[0])],
                                 linestyle=":", linewidth=1.5)
                    self.ax.scatter([fut_idx[0]], [float(yf_list[0])], zorder=6)
                else:
                    self.ax.plot(fut_idx, yf_list, linestyle=":", linewidth=1.5)
                    self.ax.scatter([fut_idx[-1]], [float(yf_list[-1])], zorder=6)


class PriceCanvas(BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(title="Precio (Close)", ylim=None, parent=parent)

    def _make_formatter(self, unit: str):
        symbol = "US$" if unit == "USD" else "$"
        def _fmt(y, _):
            try:
                return f"{symbol}{y:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return f"{symbol}{y}"
        return FuncFormatter(_fmt)

    def plot_price(self, times, close_series, model_name="Lineal",
                   forecast_steps=0, window=200, show_model=True, unit="USD"):
        self._prep()
        if close_series is None:
            self.draw(); return
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        s = pd.to_numeric(close_series, errors="coerce").dropna().astype(float)
        if s.size <= 1:
            self.draw(); return

        idx = pd.DatetimeIndex(s.index)
        try:
            if idx.tz is not None and TZ_AR is not None:
                idx = idx.tz_convert(TZ_AR).tz_localize(None)
        except Exception:
            pass
        x_obs = list(idx.to_pydatetime())
        y_obs = s.to_numpy(dtype=float).reshape(-1).tolist()

        self._plot_with_model(x_obs, y_obs, idx, model_name, forecast_steps, window,
                              clip=None, show_model=show_model)
        self._annotate_min_max(self.ax, x_obs, y_obs, fmt="{:.2f}")
        # sin leyenda
        self.ax.set_ylabel(f"Precio ({unit})")
        self.ax.yaxis.set_major_formatter(self._make_formatter(unit))
        self.draw()


class RsiCanvas(BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(title="RSI(14)", ylim=(0, 100), parent=parent)

    def plot_rsi(self, times, rsi_series, model_name="Lineal",
                 forecast_steps=0, window=200, show_model=True):
        self._prep()
        if rsi_series is None:
            self.draw(); return
        if isinstance(rsi_series, pd.DataFrame):
            rsi_series = rsi_series.iloc[:, 0]

        s = pd.to_numeric(rsi_series, errors="coerce").dropna().astype(float)
        if s.size <= 1:
            self.draw(); return

        idx = pd.DatetimeIndex(s.index)
        try:
            if idx.tz is not None and TZ_AR is not None:
                idx = idx.tz_convert(TZ_AR).tz_localize(None)
        except Exception:
            pass
        x_obs = list(idx.to_pydatetime())
        y_obs = s.to_numpy(dtype=float).reshape(-1).tolist()

        self._plot_with_model(x_obs, y_obs, idx, model_name, forecast_steps, window,
                              clip=(0, 100), show_model=show_model)
        # Líneas guía RSI
        self.ax.axhline(70, color="red", linestyle="--", linewidth=0.25)
        self.ax.axhline(30, color="red", linestyle="--", linewidth=0.25)
        self.ax.axhline(55, color="blue", linestyle="--", linewidth=0.5)
        self.ax.axhline(45, color="blue", linestyle="--", linewidth=0.5)
        self._annotate_min_max(self.ax, x_obs, y_obs, fmt="{:.1f}")
        # sin leyenda
        self.ax.set_ylabel("RSI")
        self.draw()


# ============================ Ventana principal ============================ #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RSI Monitor - EconometricaGPT")
        self.resize(1280, 960)

        # Proveedor (default Yahoo)
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems(["Yahoo", "Investing (exp)"])
        self.provider_combo.setCurrentIndex(0)

        # Wallet
        self.wallet_combo = QtWidgets.QComboBox()
        self.btn_reload_wallet = QtWidgets.QPushButton("Recargar wallet")

        # Intervalo / Periodo / Refresco
        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.addItems(["1m","5m","15m","30m","60m","1d"])
        self.period_combo = QtWidgets.QComboBox()
        self.period_combo.addItems(["1d","5d","1mo","3mo","6mo","1y"])
        self.refresh_spin = QtWidgets.QSpinBox()
        self.refresh_spin.setRange(5,3600)
        self.refresh_spin.setValue(5)  # <-- 2) refresco por defecto 5 s
        self.refresh_spin.setSuffix(" s")

        # Unidades de visualización
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["ARS","USD"])
        self.unit_combo.setCurrentIndex(1)  # <-- 1) USD por defecto

        # Proyección
        self.horizon_combo = QtWidgets.QComboBox(); self.horizon_combo.addItems(["1m","1h","1d"])
        self.model_combo = QtWidgets.QComboBox(); self.model_combo.addItems(["Lineal","Holt-Winters","ARIMA"])
        self.window_spin = QtWidgets.QSpinBox(); self.window_spin.setRange(30,2000); self.window_spin.setValue(200); self.window_spin.setSuffix(" pts")
        self.chk_show_model = QtWidgets.QCheckBox("Mostrar regresión/proyección"); self.chk_show_model.setChecked(True)

        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_stop = QtWidgets.QPushButton("Detener"); self.btn_stop.setEnabled(False)

        # Título
        self.instrument_title = QtWidgets.QLabel("Instrumento: -")
        f = self.instrument_title.font(); f.setPointSize(12); f.setBold(True); self.instrument_title.setFont(f)

        # ---- Línea de info debajo del instrumento (4) ----
        self.info_line = QtWidgets.QLabel("Precio: - | Señal: - | Tendencia RSI: -")
        fi = self.info_line.font(); fi.setPointSize(11); self.info_line.setFont(fi)

        # ---------- Panel USD ----------
        self.usd_group = QtWidgets.QGroupBox("Dólar de referencia (máximo entre CCL/MEP/Oficial)")
        self.lbl_usd_all = QtWidgets.QLabel("CCL: - | MEP: - | OFICIAL: -")
        self.lbl_usd_pick = QtWidgets.QLabel("Usando: -")
        usd_layout = QtWidgets.QHBoxLayout()
        usd_layout.addWidget(self.lbl_usd_all); usd_layout.addStretch(); usd_layout.addWidget(self.lbl_usd_pick)
        self.usd_group.setLayout(usd_layout)

        # Gráficos
        self.price_canvas = PriceCanvas(self)
        self.rsi_canvas = RsiCanvas(self)

        # -------- Panel de Latencia (abajo) (6) --------
        self.lat_group = QtWidgets.QGroupBox("Latencia")
        self.lbl_lag = QtWidgets.QLabel("Lag (s): -")
        self.lbl_now_ar = QtWidgets.QLabel("Hora AR: --:--"); self.lbl_now_ar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lat_layout = QtWidgets.QGridLayout()
        lat_layout.addWidget(self.lbl_lag, 0, 0)
        lat_layout.addWidget(QtWidgets.QLabel("Hora Argentina:"), 0, 1)
        lat_layout.addWidget(self.lbl_now_ar, 0, 2)
        self.lat_group.setLayout(lat_layout)

        # ----- Layout -----
        top_controls = QtWidgets.QGridLayout()
        top_controls.addWidget(QtWidgets.QLabel("Proveedor:"), 0, 0)
        top_controls.addWidget(self.provider_combo, 0, 1)
        top_controls.addWidget(QtWidgets.QLabel("Wallet:"), 0, 2)
        top_controls.addWidget(self.wallet_combo, 0, 3)
        top_controls.addWidget(self.btn_reload_wallet, 0, 4)
        top_controls.addWidget(QtWidgets.QLabel("Unidades:"), 0, 5)
        top_controls.addWidget(self.unit_combo, 0, 6)

        top_controls.addWidget(QtWidgets.QLabel("Intervalo:"), 1, 0); top_controls.addWidget(self.interval_combo, 1, 1)
        top_controls.addWidget(QtWidgets.QLabel("Periodo:"), 1, 2); top_controls.addWidget(self.period_combo, 1, 3)
        top_controls.addWidget(QtWidgets.QLabel("Refresco:"), 1, 4); top_controls.addWidget(self.refresh_spin, 1, 5)

        top_controls.addWidget(QtWidgets.QLabel("Proyección:"), 2, 0)
        top_controls.addWidget(self.horizon_combo, 2, 1)
        top_controls.addWidget(QtWidgets.QLabel("Modelo:"), 2, 2)
        top_controls.addWidget(self.model_combo, 2, 3)
        top_controls.addWidget(QtWidgets.QLabel("Ventana:"), 2, 4)
        top_controls.addWidget(self.window_spin, 2, 5)
        top_controls.addWidget(self.chk_show_model, 2, 6)
        top_controls.addWidget(self.btn_start, 2, 7); top_controls.addWidget(self.btn_stop, 2, 8)

        central = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(central)
        v.addLayout(top_controls)
        v.addWidget(self.usd_group)
        v.addWidget(self.instrument_title)
        v.addWidget(self.info_line)           # línea de información compacta (4)
        v.addWidget(self.price_canvas, stretch=1)
        v.addWidget(self.rsi_canvas, stretch=1)
        v.addWidget(self.lat_group)           # solo lag, al pie (6)
        self.setCentralWidget(central)

        # Timers
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh_data)
        self.latency_timer = QTimer(self); self.latency_timer.setInterval(1000); self.latency_timer.timeout.connect(self.refresh_latency_only)
        self.usd_timer = QTimer(self); self.usd_timer.setInterval(120000); self.usd_timer.timeout.connect(self.refresh_usd_rates)

        # Señales
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reload_wallet.clicked.connect(self.reload_wallet)

        widgets_to_connect = [
            self.wallet_combo, self.horizon_combo, self.model_combo,
            self.interval_combo, self.period_combo, self.provider_combo,
            self.unit_combo,  # QComboBox
            self.window_spin, # QSpinBox
            self.chk_show_model  # QCheckBox
        ]
        for w in widgets_to_connect:
            if isinstance(w, QtWidgets.QSpinBox):
                w.valueChanged.connect(self.refresh_data)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.connect(self.refresh_data)
            else:
                w.currentIndexChanged.connect(self.refresh_data)

        # Estado
        self.worker = None; self.last_df = None; self.last_name = "-"; self.currency_native = "N/A"
        self._last_bar_utc = None; self._last_interval_txt = "1m"
        self.usd_rates: Dict[str,float] = {}; self.usd_name = "N/A"; self.usd_value = 0.0

        # Cargar wallet y primeras cotizaciones USD
        self.reload_wallet(initial=True)
        self.refresh_usd_rates()

    # ---- helpers ----
    def reload_wallet(self, initial=False):
        tickers = load_wallet(WALLET_PATH)
        self.wallet_combo.blockSignals(True); self.wallet_combo.clear()
        self.wallet_combo.addItems(tickers if tickers else ["-SIN TICKERS-"])
        self.wallet_combo.blockSignals(False)
        if not initial: self.refresh_data()

    def current_symbol(self) -> str:
        t = self.wallet_combo.currentText().strip().upper()
        return "" if t == "-SIN TICKERS-" else t

    def start(self):
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.refresh_data()
        self.timer.start(self.refresh_spin.value()*1000)
        self.latency_timer.start()
        self.usd_timer.start()

    def stop(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.timer.stop(); self.latency_timer.stop(); self.usd_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption(); self.worker.wait(1000)

    def _interval_minutes(self, txt: str) -> int:
        return {"1m":1,"5m":5,"15m":15,"30m":30,"60m":60,"1d":1440}.get(txt,1)

    def _horizon_to_steps(self, interval_txt: str, horizon_txt: str) -> int:
        int_min = self._interval_minutes(interval_txt); hor_min = {"1m":1,"1h":60,"1d":1440}[horizon_txt]
        return max(1, int(np.ceil(hor_min/int_min)))
    
    def _trading_color_ar(self, now_ar: datetime) -> str:
        start = time(10, 30)
        end = time(17, 0)
        t = now_ar.time()
        return "#2e7d32" if (start <= t <= end) else "#c62828"

    def refresh_usd_rates(self):
        try:
            self.usd_rates = fetch_usd_rates()
            self.usd_name, self.usd_value = pick_max_usd(self.usd_rates)
            ccl = self.usd_rates.get("CCL","-")
            mep = self.usd_rates.get("MEP","-")
            off = self.usd_rates.get("OFICIAL","-")
            self.lbl_usd_all.setText(f"CCL: {_fmt_ars(ccl) if ccl!='-' else '-'} | "
                                     f"MEP: {_fmt_ars(mep) if mep!='-' else '-'} | "
                                     f"OFICIAL: {_fmt_ars(off) if off!='-' else '-'}")
            self.lbl_usd_pick.setText(f"Usando: {self.usd_name} ({_fmt_ars(self.usd_value)})" if self.usd_value>0 else "Usando: -")
            # Replot si ya hay datos
            if self.last_df is not None:
                self.on_data_ready(self.last_df, self.last_name, 0.0, self.currency_native)
        except Exception:
            pass

    # ---- ciclo ----
    def refresh_data(self):
        symbol = self.current_symbol()
        if not symbol: return
        interval = self.interval_combo.currentText(); period = self.period_combo.currentText()

        ok = True
        if interval=="1m" and period not in ("1d","5d"): ok=False
        if interval in ("5m","15m","30m") and period not in ("5d","1mo","3mo"): ok=False
        if interval=="60m" and period not in ("1mo","3mo","6mo","1y"): ok=False
        if interval=="1d": ok=True
        if not ok: return

        if self.worker and self.worker.isRunning(): return
        provider = "Yahoo" if self.provider_combo.currentIndex()==0 else "Investing"
        self.worker = DataWorker(symbol, interval, period, provider)
        self.worker.data_ready.connect(self.on_data_ready, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

    def convert_units(self, series: pd.Series, native_currency: str, target_unit: str) -> pd.Series:
        """
        ARS -> USD: divide por TC; USD -> ARS: multiplica por TC.
        Si no hay TC válido o moneda no es ARS/USD, devuelve la serie original.
        """
        if native_currency == target_unit:
            return series
        if self.usd_value <= 0:
            return series
        s = pd.to_numeric(series, errors="coerce").astype(float)
        if native_currency == "ARS" and target_unit == "USD":
            return s / self.usd_value
        if native_currency == "USD" and target_unit == "ARS":
            return s * self.usd_value
        return s

    def _fmt_unit(self, unit: str, value: float) -> str:
        return ("US$" if unit == "USD" else "$") + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def on_data_ready(self, df: pd.DataFrame, instrument_name: str, _dl_ms_unused: float, currency: str):
        self.last_df = df.copy()
        self.last_name = instrument_name or self.current_symbol()
        self.currency_native = (currency or "N/A").upper()
        sym = self.current_symbol()
        self.instrument_title.setText(f"Instrumento: {self.last_name} ({sym})")

        try:
            close_raw = df["Close"]
            if isinstance(close_raw, pd.DataFrame): close_raw = close_raw.iloc[:,0]
            close_native = pd.to_numeric(close_raw, errors="coerce").astype(float).dropna()
            if close_native.size <= 1: return

            # RSI
            rsi = compute_rsi(close_native, period=14)

            # Unidad elegida
            unit = self.unit_combo.currentText()  # "ARS" o "USD"
            native = self.currency_native if self.currency_native in ("ARS","USD") else ("ARS" if sym.endswith(".BA") else "USD")
            close_view = self.convert_units(close_native, native, unit)

            # Parámetros de proyección
            horizon = self.horizon_combo.currentText()
            interval_txt = self.interval_combo.currentText()
            steps = self._horizon_to_steps(interval_txt, horizon)
            model_name = self.model_combo.currentText()
            window = int(self.window_spin.value())
            show_model = self.chk_show_model.isChecked()

            # Graficar
            self.price_canvas.plot_price(
                close_view.index, close_view,
                model_name=model_name, forecast_steps=steps, window=window,
                show_model=show_model, unit=unit
            )
            self.rsi_canvas.plot_rsi(
                close_native.index, rsi,
                model_name=model_name, forecast_steps=steps, window=window,
                show_model=show_model
            )

            # --- Señales e info (4) ---
            s = rsi.dropna()
            last_rsi = float(s.iloc[-1]) if len(s) else np.nan
            beta, r2 = rsi_trend(rsi, lookback=10)
            estado = "-" if np.isnan(last_rsi) else ("Sobrecompra" if last_rsi>=70 else "Sobreventa" if last_rsi<=30 else "Neutral")
            tend_txt = "-"
            if not np.isnan(beta):
                direc = "alcista" if beta>0 else "bajista" if beta<0 else "plana"
                tend_txt = f"{direc} (pend={beta:.3f}, R²={r2:.2f})"
            # Precio último en unidad elegida
            last_price = float(close_view.iloc[-1])
            self.info_line.setText(f"Precio ({unit}): {self._fmt_unit(unit, last_price)} | "
                                   f"Señal: {estado if estado!='-' else ' - '} "
                                   f"{f'({last_rsi:.1f})' if not np.isnan(last_rsi) else ''} | "
                                   f"Tendencia RSI: {tend_txt}")

            # --- Latencia: sólo lag (6) ---
            last_idx = close_native.index[-1]
            if getattr(close_native.index, "tz", None) is not None:
                last_utc = last_idx.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
            else:
                last_utc = pd.Timestamp(last_idx).tz_localize("UTC").to_pydatetime().replace(tzinfo=None)
            self._last_bar_utc = last_utc
            self.refresh_latency_only()

        except Exception as e:
            self.on_error(f"Error procesando datos: {e}")

    def refresh_latency_only(self):
        try:
            # Reloj AR
            now_ar = datetime.now(TZ_AR) if TZ_AR else datetime.now()
            self.lbl_now_ar.setText(now_ar.strftime("%H:%M"))
            # <<< pintar la hora según franja horaria >>>
            self.lbl_now_ar.setStyleSheet(
                f"QLabel {{ color: {self._trading_color_ar(now_ar)}; font-weight: 600; }}"
            )

            # Lag efectivo: ahora - timestamp última barra (UTC)
            if self._last_bar_utc is not None:
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                lag_s = max(0.0, (now_utc - self._last_bar_utc).total_seconds())
                self.lbl_lag.setText(f"Lag (s): {lag_s:.1f}")
            else:
                self.lbl_lag.setText("Lag (s): -")
        except Exception:
            pass

    def on_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Datos", msg)


# ============================ Main ============================ #
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
