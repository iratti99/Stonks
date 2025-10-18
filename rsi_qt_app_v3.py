import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------- Utilidades RSI ----------------------------- #
def compute_rsi(series_close: pd.Series, period: int = 14) -> pd.Series:
    """RSI clásico de Wilder (suavizado exponencial equivalente)."""
    series_close = pd.to_numeric(series_close, errors="coerce").astype(float)
    delta = series_close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_trend(rsi: pd.Series, lookback: int = 10):
    """Pendiente y R^2 de una recta ajustada al RSI en los últimos 'lookback' puntos."""
    s = rsi.dropna()
    if len(s) < lookback:
        return np.nan, np.nan
    y = s.iloc[-lookback:].to_numpy(dtype=float)
    x = np.arange(len(y))
    x_mean, y_mean = x.mean(), y.mean()
    beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    y_hat = (y_mean - beta1 * x_mean) + beta1 * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(beta1), float(r2)


# ----------------------------- Worker en Thread ----------------------------- #
class DataWorker(QThread):
    data_ready = pyqtSignal(pd.DataFrame, str)  # df, instrument_name
    error = pyqtSignal(str)

    def __init__(self, symbol: str, interval: str, period: str, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.interval = interval
        self.period = period

    def run(self):
        try:
            df = yf.download(
                tickers=self.symbol,
                interval=self.interval,
                period=self.period,
                progress=False,
                auto_adjust=False,
                prepost=True,
                threads=True
            )
            if df is None or df.empty:
                raise ValueError("Sin datos devueltos. Revisa símbolo/intervalo/periodo.")
            df = df.sort_index()

            # Intentar obtener nombre “largo”
            name = self.symbol
            try:
                tk = yf.Ticker(self.symbol)
                info = {}
                try:
                    info = tk.get_info() or {}
                except Exception:
                    info = getattr(tk, "info", {}) or {}
                name = (
                    info.get("longName")
                    or info.get("shortName")
                    or info.get("displayName")
                    or info.get("symbol")
                    or self.symbol
                )
            except Exception:
                pass

            self.data_ready.emit(df, name)
        except Exception as e:
            self.error.emit(f"Error descargando datos: {e}")


# ----------------------------- Lienzos Matplotlib ----------------------------- #
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
        """Marca mínimos/máximos con puntos y etiquetas."""
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.size == 0 or np.all(np.isnan(y_arr)):
            return

        if isinstance(x, (pd.DatetimeIndex, pd.Series)):
            idx = pd.DatetimeIndex(x)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            x_list = list(idx.to_pydatetime())
        else:
            x_list = list(x)

        if len(x_list) != y_arr.size:
            return

        i_min = int(np.nanargmin(y_arr))
        i_max = int(np.nanargmax(y_arr))
        vmin = float(y_arr[i_min]); vmax = float(y_arr[i_max])

        ax.scatter([x_list[i_min]], [vmin], s=40, color="red", zorder=5)
        ax.scatter([x_list[i_max]], [vmax], s=40, color="green", zorder=5)

        ax.annotate(fmt.format(vmin), xy=(x_list[i_min], vmin),
                    xytext=(0, -12 + yoffset), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=0.6, alpha=0.8))
        ax.annotate(fmt.format(vmax), xy=(x_list[i_max], vmax),
                    xytext=(0, 12 + yoffset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="green",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", lw=0.6, alpha=0.8))

    # ----------------- Modelos de proyección ----------------- #
    @staticmethod
    def _forecast_lineal(y: np.ndarray, steps: int, window: int, clip=None):
        y = np.asarray(y, dtype=float).reshape(-1)
        n = y.size
        if n < 10:
            return None, None
        w = min(window, n)
        t = np.arange(w)
        y_win = y[-w:]
        b, a = np.polyfit(t, y_win, 1)  # y ≈ a + b*t
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
        n = y.size
        if n < 15:
            return None, None
        w = min(window, n)
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
        n = y.size
        if n < 20:
            return None, None
        w = min(window, n)
        y_win = y[-w:]
        try:
            # Modelo sencillo por defecto; si falla, probamos alternativas
            for order in [(1,1,1), (1,0,0), (0,1,1)]:
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
        except Exception:
            return None, None

    @staticmethod
    def _future_index(idx: pd.DatetimeIndex, steps: int):
        """Construye índices futuros equiespaciados a partir del último delta observado."""
        if steps <= 0 or idx.size < 2:
            return []
        last = idx[-1]
        delta = idx[-1] - idx[-2]
        if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
            delta = pd.Timedelta(minutes=1)
        fut = [last + (i + 1) * delta for i in range(steps)]
        return fut

    def _plot_with_model(self, x_obs, y_obs, idx, model_name: str, steps: int, window: int, clip=None):
        """Dibuja línea observada + ajuste + proyección según el modelo elegido."""
        label_map = {"Lineal": "Regresión", "Holt-Winters": "HW (tendencia)", "ARIMA": "ARIMA"}
        self.ax.plot(x_obs, y_obs, linewidth=1.5, label="Observado")

        y_fit = y_future = None
        if steps > 0:
            if model_name == "Lineal":
                y_fit, y_future = self._forecast_lineal(np.array(y_obs), steps, window, clip=clip)
            elif model_name == "Holt-Winters":
                y_fit, y_future = self._forecast_holtwinters(np.array(y_obs), steps, window, clip=clip)
            elif model_name == "ARIMA":
                y_fit, y_future = self._forecast_arima(np.array(y_obs), steps, window, clip=clip)

        if y_fit is not None:
            x_fit = x_obs[-len(y_fit):]
            self.ax.plot(x_fit, list(y_fit), linestyle="--", linewidth=1.2, label=label_map.get(model_name, "Ajuste"))
        if y_future is not None:
            fut_idx = self._future_index(idx, steps)
            if fut_idx:
                self.ax.plot(fut_idx, list(y_future), linestyle=":", linewidth=1.5, label="Proyección")
                self.ax.scatter([fut_idx[-1]], [float(y_future[-1])], zorder=6)


class PriceCanvas(BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(title="Precio (Close)", ylim=None, parent=parent)

    def plot_price(self, times, close_series, model_name="Lineal", forecast_steps=0, window=200):
        self._prep()
        if close_series is None:
            self.draw(); return
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        s = pd.to_numeric(close_series, errors="coerce").dropna().astype(float)
        if s.size <= 1:
            self.draw(); return

        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        x_obs = list(idx.to_pydatetime())
        y_obs = s.to_numpy(dtype=float).reshape(-1).tolist()

        # Observado + modelo
        self._plot_with_model(x_obs, y_obs, idx, model_name, forecast_steps, window, clip=None)
        self._annotate_min_max(self.ax, x_obs, y_obs, fmt="{:.2f}")
        self.ax.legend(loc="best")
        self.ax.set_ylabel("Precio")
        self.draw()


class RsiCanvas(BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(title="RSI(14)", ylim=(0, 100), parent=parent)

    def plot_rsi(self, times, rsi_series, model_name="Lineal", forecast_steps=0, window=200):
        self._prep()
        if rsi_series is None:
            self.draw(); return
        if isinstance(rsi_series, pd.DataFrame):
            rsi_series = rsi_series.iloc[:, 0]

        s = pd.to_numeric(rsi_series, errors="coerce").dropna().astype(float)
        if s.size <= 1:
            self.draw(); return

        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        x_obs = list(idx.to_pydatetime())
        y_obs = s.to_numpy(dtype=float).reshape(-1).tolist()

        # Observado + modelo (con recorte 0-100)
        self._plot_with_model(x_obs, y_obs, idx, model_name, forecast_steps, window, clip=(0, 100))
        # Líneas 30/70 y min/max
        self.ax.axhline(70, linestyle="--")
        self.ax.axhline(30, linestyle="--")
        self._annotate_min_max(self.ax, x_obs, y_obs, fmt="{:.1f}")

        self.ax.legend(loc="best")
        self.ax.set_ylabel("RSI")
        self.draw()


# ----------------------------- Ventana Principal ----------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RSI Monitor - EconometricaGPT")
        self.resize(1200, 880)

        # Widgets de control
        self.symbol_edit = QtWidgets.QLineEdit("AAPL")
        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.addItems(["1m", "5m", "15m", "30m", "60m", "1d"])
        self.period_combo = QtWidgets.QComboBox()
        self.period_combo.addItems(["1d", "5d", "1mo", "3mo", "6mo", "1y"])
        self.refresh_spin = QtWidgets.QSpinBox()
        self.refresh_spin.setRange(5, 3600)
        self.refresh_spin.setValue(60)
        self.refresh_spin.setSuffix(" s")

        # Selectores de proyección
        self.horizon_combo = QtWidgets.QComboBox()
        self.horizon_combo.addItems(["1m", "1h", "1d"])
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["Lineal", "Holt-Winters", "ARIMA"])
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(30, 2000)
        self.window_spin.setValue(200)
        self.window_spin.setSuffix(" pts")

        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_stop = QtWidgets.QPushButton("Detener")
        self.btn_stop.setEnabled(False)

        # Título con nombre completo del instrumento
        self.instrument_title = QtWidgets.QLabel("Instrumento: -")
        f = self.instrument_title.font()
        f.setPointSize(12); f.setBold(True)
        self.instrument_title.setFont(f)

        # Estado y señales
        self.status_label = QtWidgets.QLabel("Listo.")
        self.status_label.setWordWrap(True)
        self.signal_label = QtWidgets.QLabel("Señal: -")
        self.trend_label = QtWidgets.QLabel("Tendencia RSI: -")
        fontb = self.signal_label.font(); fontb.setPointSize(11)
        self.signal_label.setFont(fontb); self.trend_label.setFont(fontb)

        # Canvases
        self.price_canvas = PriceCanvas(self)
        self.rsi_canvas = RsiCanvas(self)

        # Layout superior (controles)
        controls_layout = QtWidgets.QGridLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Símbolo:"), 0, 0)
        controls_layout.addWidget(self.symbol_edit, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Intervalo:"), 0, 2)
        controls_layout.addWidget(self.interval_combo, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Periodo:"), 0, 4)
        controls_layout.addWidget(self.period_combo, 0, 5)
        controls_layout.addWidget(QtWidgets.QLabel("Refresco:"), 0, 6)
        controls_layout.addWidget(self.refresh_spin, 0, 7)

        controls_layout.addWidget(QtWidgets.QLabel("Proyección:"), 1, 0)
        controls_layout.addWidget(self.horizon_combo, 1, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Modelo:"), 1, 2)
        controls_layout.addWidget(self.model_combo, 1, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Ventana:"), 1, 4)
        controls_layout.addWidget(self.window_spin, 1, 5)

        controls_layout.addWidget(self.btn_start, 1, 10)
        controls_layout.addWidget(self.btn_stop, 1, 11)

        # Layout señales
        signals_layout = QtWidgets.QHBoxLayout()
        signals_layout.addWidget(self.signal_label)
        signals_layout.addSpacing(30)
        signals_layout.addWidget(self.trend_label)
        signals_layout.addStretch()

        # Layout central
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(controls_layout)
        v.addWidget(self.instrument_title)
        v.addLayout(signals_layout)
        v.addWidget(self.price_canvas, stretch=1)
        v.addWidget(self.rsi_canvas, stretch=1)
        v.addWidget(self.status_label)
        self.setCentralWidget(central)

        # Timer de refresco
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)

        # Señales botones
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.horizon_combo.currentIndexChanged.connect(self.refresh_data)
        self.model_combo.currentIndexChanged.connect(self.refresh_data)
        self.window_spin.valueChanged.connect(self.refresh_data)

        # Worker & cache
        self.worker = None
        self.last_df = None
        self.last_name = "-"

    # ------------------ Lógica de la app ------------------ #
    def start(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Descargando datos iniciales...")
        self.refresh_data()
        self.timer.start(self.refresh_spin.value() * 1000)

    def stop(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(1000)
        self.status_label.setText("Detenido.")

    def _interval_minutes(self, txt: str) -> int:
        return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 1440}.get(txt, 1)

    def _horizon_to_steps(self, interval_txt: str, horizon_txt: str) -> int:
        """Convierte el horizonte (1m/1h/1d) a 'steps' según el intervalo actual."""
        int_min = self._interval_minutes(interval_txt)
        hor_min = {"1m": 1, "1h": 60, "1d": 1440}[horizon_txt]
        steps = int(np.ceil(hor_min / int_min))
        return max(1, steps)

    def refresh_data(self):
        symbol = self.symbol_edit.text().strip().upper()
        interval = self.interval_combo.currentText()
        period = self.period_combo.currentText()

        ok = True
        if interval == "1m" and period not in ("1d", "5d"):
            ok = False
        if interval in ("5m", "15m", "30m") and period not in ("5d", "1mo", "3mo"):
            ok = False
        if interval == "60m" and period not in ("1mo", "3mo", "6mo", "1y"):
            ok = False
        if interval == "1d":
            ok = True
        if not ok:
            self.status_label.setText("⚠️ Para 1m usa 1d/5d; para 5-30m usa 5d/1mo/3mo; para 60m usa ≥1mo.")
            return

        if self.worker and self.worker.isRunning():
            return  # evita solapamientos
        self.worker = DataWorker(symbol, interval, period)
        self.worker.data_ready.connect(self.on_data_ready, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

    def on_data_ready(self, df: pd.DataFrame, instrument_name: str):
        self.last_df = df.copy()
        self.last_name = instrument_name or self.symbol_edit.text().strip().upper()
        try:
            # 1) Título
            sym = self.symbol_edit.text().strip().upper()
            self.instrument_title.setText(f"Instrumento: {self.last_name} ({sym})")

            # 2) Close como Series 1D float (defensa si viene DataFrame)
            close_raw = df["Close"]
            if isinstance(close_raw, pd.DataFrame):
                close_raw = close_raw.iloc[:, 0]
            close = pd.to_numeric(close_raw, errors="coerce").astype(float).dropna()
            if close.size <= 1:
                self.status_label.setText("Sin suficientes datos para graficar.")
                return

            # 3) RSI
            rsi = compute_rsi(close, period=14)

            # 4) Proyección: parámetros desde la UI
            horizon = self.horizon_combo.currentText()
            interval_txt = self.interval_combo.currentText()
            steps = self._horizon_to_steps(interval_txt, horizon)
            model_name = self.model_combo.currentText()
            window = int(self.window_spin.value())

            # 5) Dibujar con modelo elegido
            self.price_canvas.plot_price(close.index, close,
                                         model_name=model_name, forecast_steps=steps, window=window)
            self.rsi_canvas.plot_rsi(close.index, rsi,
                                     model_name=model_name, forecast_steps=steps, window=window)

            # 6) Señales RSI
            s = rsi.dropna()
            last_rsi = float(s.iloc[-1]) if len(s) else np.nan
            beta, r2 = rsi_trend(rsi, lookback=10)
            if not np.isnan(last_rsi):
                if last_rsi >= 70: estado = f"Sobrecompra ({last_rsi:.1f})"
                elif last_rsi <= 30: estado = f"Sobreventa ({last_rsi:.1f})"
                else: estado = f"Neutral ({last_rsi:.1f})"
            else:
                estado = "-"
            self.signal_label.setText(f"Señal: {estado}")

            if not np.isnan(beta):
                direc = "alcista" if beta > 0 else "bajista" if beta < 0 else "plana"
                trend_txt = f"{direc} (pendiente={beta:.3f}, R²={r2:.2f})"
            else:
                trend_txt = "-"
            self.trend_label.setText(f"Tendencia RSI: {trend_txt}")

            # 7) Lag efectivo
            last_idx = close.index[-1]
            if getattr(close.index, "tz", None) is not None:
                last_utc = last_idx.tz_convert("UTC").to_pydatetime()
            else:
                last_utc = pd.Timestamp(last_idx).tz_localize("UTC").to_pydatetime()
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            lag_sec = max(0.0, (now_utc - last_utc).total_seconds())
            try:
                last_local = last_idx.tz_convert("America/Argentina/Buenos_Aires")
            except Exception:
                last_local = last_idx

            self.status_label.setText(
                f"Última barra: {last_local} | Filas: {len(df)} | Lag aprox: {lag_sec:.1f}s | "
                f"Proyección: {horizon} | Modelo: {model_name} | Ventana: {window}"
            )

        except Exception as e:
            self.status_label.setText(f"Error procesando datos: {e}")

    def on_error(self, msg: str):
        self.status_label.setText(msg)


# ----------------------------- Main ----------------------------- #
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
