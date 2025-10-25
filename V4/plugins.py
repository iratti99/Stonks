"""
plugins.py — Arquitectura de *plugins* de gráficos para tu app PyQt6

Objetivo
--------
Centraliza TODA la lógica que **genera gráficos** en un módulo único y extensible.
Con esto, en `rsi_qt_app_v4.py` solo creas el manejador de pestañas y le pasas
los datos + contexto. Para agregar un gráfico nuevo, implementás un plugin y lo
registrás: sin tocar el resto.

Estructura
---------
- BasePlugin: interfaz mínima de un plugin de gráfico.
- PluginTabsManager (QTabWidget): contenedor de pestañas que registra plugins y
  los actualiza a todos con `update_all(...)`.
- PricePlugin: gráfico de precio con proyección (Lineal/Holt-Winters/ARIMA-like).
- RSIPlugin: gráfico de RSI con líneas 30/70 y proyección.
- IndicatorsPlugin: envoltura para `IndicatorTabs` (MACD/SMA/Volumen/CaroBarato/OBV/MFI).

Uso en tu app (ejemplo mínimo)
------------------------------
from plugins import PluginTabsManager, PricePlugin, RSIPlugin, IndicatorsPlugin

# En __init__ de tu MainWindow
self.plugins = PluginTabsManager(parent=self)
self.plugins.add_plugin(PricePlugin())
self.plugins.add_plugin(RSIPlugin())
self.plugins.add_plugin(IndicatorsPlugin())
layout.addWidget(self.plugins)

# En on_data_ready(df, ticker, ...)
ctx = {
    "unit": self.unit_combo.currentText(),
    "model_name": self.model_combo.currentText(),
    "forecast_steps": self._horizon_to_steps(self._last_interval_txt, self.horizon_combo.currentText()),
    "window": self.window_spin.value(),
    "show_model": self.chk_show_model.isChecked(),
}
self.plugins.update_all(df, ticker, ctx)

Extender
--------
Para crear otro gráfico:
class MiPlugin(BasePlugin):
    name = "Mi Gráfico"
    def __init__(self):
        super().__init__()
        # crear self.widget (por ejemplo un MatplotlibCanvas)
    def update_data(self, df, ticker, ctx):
        # dibujar con df y parámetros de ctx

self.plugins.add_plugin(MiPlugin())

Requisitos de DF: columnas ['Open','High','Low','Close','Volume'] y DatetimeIndex.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Reusamos tu módulo existente de indicadores
try:
    from indicadores_tabs import IndicatorTabs
except Exception:
    IndicatorTabs = None  # permite importar aunque aún no exista el archivo


# ============================ Utilidades comunes ============================ #

class MatplotlibCanvas(FigureCanvas):
    """Canvas simple con helpers para líneas, rejilla y título."""
    def __init__(self, title: str = "", ylim: Optional[tuple] = None, parent: Optional[QtWidgets.QWidget] = None):
        fig = Figure(figsize=(8, 3.6), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.title = title
        self.ylim = ylim

    def prep(self):
        self.ax.clear()
        if self.title:
            self.ax.set_title(self.title)
        if self.ylim:
            self.ax.set_ylim(*self.ylim)
        self.ax.grid(True, which="both", linestyle="--", alpha=0.3)

    @staticmethod
    def annotate_minmax(ax, x_list, y_series, yoffset=0, fmt="{:.2f}"):
        if y_series is None or len(y_series) == 0:
            return
        y_arr = np.asarray(y_series, dtype=float)
        if np.all(np.isnan(y_arr)):
            return
        i_min = int(np.nanargmin(y_arr)); i_max = int(np.nanargmax(y_arr))
        vmin = float(y_arr[i_min]); vmax = float(y_arr[i_max])
        ax.scatter([x_list[i_min]], [vmin], s=40, color="green", zorder=5)
        ax.scatter([x_list[i_max]], [vmax], s=40, color="red", zorder=5)
        ax.annotate(fmt.format(vmin), xy=(x_list[i_min], vmin), xytext=(0, -12 + yoffset),
                    textcoords="offset points", ha="center", va="top", fontsize=9, color="green",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", lw=0.6, alpha=0.8))
        ax.annotate(fmt.format(vmax), xy=(x_list[i_max], vmax), xytext=(0, 12 + yoffset),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=0.6, alpha=0.8))


# ============================ Modelos sencillos (internos al plugin) ============================ #

def linear_fit_forecast(y: np.ndarray, steps: int, window: int = 200):
    if len(y) < 3:
        return y, np.array([])
    w = min(window, len(y))
    x = np.arange(w); ys = y[-w:]
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_hat = m * x + b
    x_fut = np.arange(w, w + steps)
    y_fut = m * x_fut + b
    return y_hat, y_fut


def holt_winters_like(y: np.ndarray, steps: int, alpha: float = 0.4):
    if len(y) == 0:
        return y, np.array([])
    s = np.zeros_like(y); s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i - 1]
    return s, np.full(steps, s[-1] if len(s) else np.nan)


def arima_like(y: np.ndarray, steps: int):
    return y, np.full(steps, y[-1] if len(y) else np.nan)


# ============================ Interfaz de Plugin ============================ #

class BasePlugin(Protocol):
    """Interfaz mínima: nombre visible, widget (QtWidget) y actualización."""
    name: str
    widget: QtWidgets.QWidget
    def update_data(self, df: pd.DataFrame, ticker: str, ctx: Dict):
        ...


class PluginTabsManager(QtWidgets.QTabWidget):
    """Contenedor de pestañas para plugins de gráficos."""
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._plugins: List[BasePlugin] = []

    def add_plugin(self, plugin: BasePlugin):
        self._plugins.append(plugin)
        self.addTab(plugin.widget, plugin.name)

    def update_all(self, df: pd.DataFrame, ticker: str, ctx: Optional[Dict] = None):
        ctx = ctx or {}
        for p in self._plugins:
            try:
                p.update_data(df, ticker, ctx)
            except Exception as e:
                print(f"Plugin '{p.name}' falló en update_data: {e}")


# ============================ Plugins concretos ============================ #

class PricePlugin:
    name = "Precio"
    def __init__(self):
        self.canvas = MatplotlibCanvas(title="Precio (Close)")
        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.addWidget(self.canvas)

    def _make_formatter(self, unit: str):
        symbol = "US$" if unit == "USD" else "$"
        def _fmt(y, _):
            try:
                return f"{symbol}{y:,.2f}".replace(",","X").replace(".",",").replace("X",".")
            except Exception:
                return f"{symbol}{y}"
        return FuncFormatter(_fmt)

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: Dict):
        unit = ctx.get("unit", "USD")
        model_name = ctx.get("model_name", "Lineal")
        steps = int(ctx.get("forecast_steps", 0))
        window = int(ctx.get("window", 200))
        show_model = bool(ctx.get("show_model", True))

        close = df["Close"].astype(float)
        times = df.index.to_pydatetime()
        y = np.asarray(close, dtype=float)

        ax = self.canvas.ax
        self.canvas.prep()
        ax.plot(times, y, linewidth=1.5, label=f"{ticker} Close")

        if show_model and len(y) > 2:
            if model_name == "Lineal":
                y_hat, y_future = linear_fit_forecast(y, steps, window=window)
            elif model_name == "Holt-Winters":
                y_hat, y_future = holt_winters_like(y, steps)
            else:
                y_hat, y_future = arima_like(y, steps)
            idx_hat = times[-len(y_hat):] if len(y_hat) <= len(times) else times
            ax.plot(idx_hat, y_hat, linestyle="--", linewidth=1.2, label=f"Ajuste {model_name}")
            if len(y_future) > 0:
                fut_idx = pd.date_range(times[-1], periods=len(y_future)+1, freq=pd.infer_freq(pd.Index(times)) or 'D')[1:]
                ax.plot(fut_idx, y_future, linestyle=":", linewidth=1.5, label="Proyección")

        MatplotlibCanvas.annotate_minmax(ax, times, close, yoffset=0)
        ax.set_ylabel(f"Close ({unit})")
        ax.yaxis.set_major_formatter(self._make_formatter(unit))
        ax.legend(loc="best")
        self.canvas.draw()


class RSIPlugin:
    name = "RSI"
    def __init__(self, period: int = 14):
        self.period = period
        self.canvas = MatplotlibCanvas(title=f"RSI({period})", ylim=(0, 100))
        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.addWidget(self.canvas)

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: Dict):
        model_name = ctx.get("model_name", "Lineal")
        steps = int(ctx.get("forecast_steps", 0))
        window = int(ctx.get("window", 200))
        show_model = bool(ctx.get("show_model", True))

        close = df["Close"].astype(float)
        rsi = self.compute_rsi(close, period=self.period)
        y = np.asarray(rsi, dtype=float)
        times = df.index.to_pydatetime()

        ax = self.canvas.ax
        self.canvas.prep()
        ax.plot(times, y, linewidth=1.5, label="RSI")
        ax.axhline(70, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(30, linestyle="--", linewidth=1.0, alpha=0.6)

        if show_model and len(y) > 2:
            if model_name == "Lineal":
                y_hat, y_future = linear_fit_forecast(y, steps, window=window)
            elif model_name == "Holt-Winters":
                y_hat, y_future = holt_winters_like(y, steps)
            else:
                y_hat, y_future = arima_like(y, steps)
            idx_hat = times[-len(y_hat):] if len(y_hat) <= len(times) else times
            ax.plot(idx_hat, y_hat, linestyle="--", linewidth=1.2, label=f"Ajuste {model_name}")
            if len(y_future) > 0:
                fut_idx = pd.date_range(times[-1], periods=len(y_future)+1, freq=pd.infer_freq(pd.Index(times)) or 'D')[1:]
                ax.plot(fut_idx, y_future, linestyle=":", linewidth=1.5, label="Proyección")

        MatplotlibCanvas.annotate_minmax(ax, times, rsi, yoffset=0)
        ax.set_ylabel("RSI")
        ax.legend(loc="best")
        self.canvas.draw()


class IndicatorsPlugin:
    """Envoltura para tu `IndicatorTabs` existente como un plugin más."""
    name = "Indicadores"
    def __init__(self):
        if IndicatorTabs is None:
            raise ImportError("No se encontró 'indicadores_tabs.py'. Coloca ese archivo junto a plugins.py")
        self._tabs = IndicatorTabs()
        self.widget = self._tabs  # se usa directamente como pestaña

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: Dict):
        self._tabs.update_data(df, ticker)


# ============================ Demo rápida ============================ #
if __name__ == "__main__":
    # Demostración independiente con datos de yfinance
    import yfinance as yf

    app = QtWidgets.QApplication([])

    w = QtWidgets.QMainWindow(); w.setWindowTitle("Demo Plugins")
    central = QtWidgets.QWidget(); w.setCentralWidget(central)
    lay = QtWidgets.QVBoxLayout(central)

    mgr = PluginTabsManager()
    mgr.add_plugin(PricePlugin())
    mgr.add_plugin(RSIPlugin())
    if IndicatorTabs is not None:
        mgr.add_plugin(IndicatorsPlugin())
    lay.addWidget(mgr)

    ticker = "AAPL"
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)
    df.index = pd.to_datetime(df.index, utc=True)

    ctx = {"unit": "USD", "model_name": "Lineal", "forecast_steps": 10, "window": 200, "show_model": True}
    mgr.update_all(df, ticker, ctx)

    w.resize(1100, 800); w.show()
    app.exec()
