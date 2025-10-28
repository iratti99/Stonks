# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ============================================================= #
# NORMALIZACIÓN OHLCV (robusta para yfinance)
# ============================================================= #
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas estándar: Open, High, Low, Close, Volume.
    - Acepta minúsculas / MultiIndex.
    - Usa 'Adj Close' si no hay 'Close'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)

    mapping = {c.lower(): c for c in out.columns}

    def pick(*names):
        for n in names:
            if n in mapping:
                return mapping[n]
        return None

    col_open  = pick("open")
    col_high  = pick("high")
    col_low   = pick("low")
    col_close = pick("close") or pick("adj close", "adj_close", "adjusted close")
    col_vol   = pick("volume", "vol")

    cols = {}
    if col_open:  cols["Open"]   = out[col_open]
    if col_high:  cols["High"]   = out[col_high]
    if col_low:   cols["Low"]    = out[col_low]
    if col_close: cols["Close"]  = out[col_close]
    if col_vol:   cols["Volume"] = out[col_vol]

    if set(cols.keys()) != {"Open","High","Low","Close","Volume"}:
        return pd.DataFrame()
    return pd.DataFrame(cols, index=out.index)


# ============================================================= #
# Lienzo base Matplotlib
# ============================================================= #
class PlotCanvas(FigureCanvas):
    def __init__(self, title: str):
        fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.title = title

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)

    def draw_plot(self):
        self.figure.tight_layout()
        self.draw_idle()


# ============================================================= #
# Plugins concretos
# ============================================================= #
class PricePlugin(QtWidgets.QWidget):
    name = "Precio"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        self.canvas = PlotCanvas("Precio (Close)")
        lay.addWidget(self.canvas)

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: dict):
        df = normalize_ohlcv(df)
        self.canvas.clear_plot()
        if df.empty:
            self.canvas.ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center",
                                transform=self.canvas.ax.transAxes)
            self.canvas.draw_plot()
            return

        unit = ctx.get("unit", "USD")
        currency_native = ctx.get("currency_native", "USD").upper()
        usd_value = float(ctx.get("usd_value", 0.0) or 0.0)

        close = df["Close"].astype(float)
        if unit == "ARS" and currency_native == "USD" and usd_value > 0:
            close = close * usd_value
        elif unit == "USD" and currency_native == "ARS" and usd_value > 0:
            close = close / usd_value

        times = df.index.to_pydatetime()
        # Cortar gaps >4h para evitar “rectas”
        mask = np.diff(pd.Series(times).astype("int64")/1e9, prepend=times[0].timestamp()) < 60*60*4

        self.canvas.ax.plot(np.array(times)[mask], close[mask], linewidth=1.5, label=f"{ticker} Close")
        self.canvas.ax.set_ylabel(f"Close ({unit})")
        self.canvas.ax.legend(loc="best")
        self.canvas.draw_plot()


class RSIPlugin(QtWidgets.QWidget):
    name = "RSI"

    def __init__(self, parent=None, period: int = 14):
        super().__init__(parent)
        self.period = period
        lay = QtWidgets.QVBoxLayout(self)
        self.canvas = PlotCanvas(f"RSI({period})")
        lay.addWidget(self.canvas)

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        return 100 - (100 / (1 + rs))

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: dict):
        df = normalize_ohlcv(df)
        self.canvas.clear_plot()
        if df.empty:
            self.canvas.ax.text(0.5, 0.5, "Sin datos válidos", ha="center", va="center",
                                transform=self.canvas.ax.transAxes)
            self.canvas.draw_plot()
            return

        rsi = self.compute_rsi(df["Close"].astype(float), period=self.period)
        times = df.index.to_pydatetime()
        mask = np.diff(pd.Series(times).astype("int64")/1e9, prepend=times[0].timestamp()) < 60*60*4

        self.canvas.ax.plot(np.array(times)[mask], rsi[mask], label="RSI", linewidth=1.5)
        self.canvas.ax.axhline(70, linestyle=":", color="red",  label="Sobrecompra 70")
        self.canvas.ax.axhline(30, linestyle=":", color="green", label="Sobreventa 30")
        self.canvas.ax.set_ylim(0, 100)
        self.canvas.ax.legend(loc="best")
        self.canvas.draw_plot()


# =============== Indicadores integrados (antes en indicadores_tabs.py) =============== #
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def calc_sma(df: pd.DataFrame, windows=(20,50,200)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out

def calc_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = ema(out["Close"], fast)
    out["EMA_slow"] = ema(out["Close"], slow)
    out["MACD"] = out["EMA_fast"] - out["EMA_slow"]
    out["Signal"] = ema(out["MACD"], signal)
    out["Hist"] = out["MACD"] - out["Signal"]
    return out

def calc_obv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sign = np.sign(out["Close"].diff().fillna(0.0))
    out["OBV"] = (sign * out["Volume"]).cumsum()
    return out

def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    rmf = tp * out["Volume"].astype(float)
    tp_delta = tp.diff()
    pos_mf = rmf.where(tp_delta > 0, 0.0)
    neg_mf = rmf.where(tp_delta < 0, 0.0)
    pmf = pos_mf.rolling(period, min_periods=period).sum()
    nmf = neg_mf.rolling(period, min_periods=period).sum().abs()
    ratio = pmf / nmf
    out["MFI"] = 100 - (100 / (1 + ratio))
    return out


class _PlotWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.canvas = PlotCanvas(title)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.canvas)


class IndicatorsPlugin(QtWidgets.QTabWidget):
    name = "Indicadores"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.tab_sma = _PlotWidget("SMA 20/50/200")
        self.tab_macd = _PlotWidget("MACD (12,26,9)")
        self.tab_vol = _PlotWidget("Volumen")
        self.tab_val = _PlotWidget("Caro/Barato (SMA200)")
        self.tab_obv = _PlotWidget("On-Balance Volume (OBV)")
        self.tab_mfi = _PlotWidget("Money Flow Index (14)")
        self.addTab(self.tab_sma, "SMA")
        self.addTab(self.tab_macd, "MACD")
        self.addTab(self.tab_vol, "Volumen")
        self.addTab(self.tab_val, "Caro/Barato (SMA)")
        self.addTab(self.tab_obv, "OBV")
        self.addTab(self.tab_mfi, "MFI")

    def update_data(self, df: pd.DataFrame, ticker: str, ctx: dict):
        base = normalize_ohlcv(df)
        if base.empty:
            for w in (self.tab_sma, self.tab_macd, self.tab_vol, self.tab_val, self.tab_obv, self.tab_mfi):
                w.canvas.clear_plot(); w.canvas.draw_plot()
            return

        # SMA
        sma = calc_sma(base)
        ax = self.tab_sma.canvas.ax; self.tab_sma.canvas.clear_plot()
        ax.plot(sma.index, sma["Close"], label=f"{ticker} Close")
        for w in (20, 50, 200):
            ax.plot(sma.index, sma[f"SMA{w}"], label=f"SMA{w}")
        ax.legend(loc="best"); self.tab_sma.canvas.draw_plot()

        # MACD
        macd = calc_macd(base)
        ax = self.tab_macd.canvas.ax; self.tab_macd.canvas.clear_plot()
        ax.plot(macd.index, macd["MACD"], label="MACD")
        ax.plot(macd.index, macd["Signal"], linestyle=":", label="Signal")
        ax.bar(macd.index, macd["Hist"], alpha=0.35, label="Histograma")
        ax.legend(loc="best"); self.tab_macd.canvas.draw_plot()

        # Volumen
        ax = self.tab_vol.canvas.ax; self.tab_vol.canvas.clear_plot()
        ax.bar(base.index, base["Volume"], label="Volumen")
        ax.legend(loc="best"); self.tab_vol.canvas.draw_plot()

        # Caro/Barato por SMA200
        val = sma.copy()
        val["dist_sma"] = (val["Close"] / val["SMA200"] - 1.0)
        ax = self.tab_val.canvas.ax; self.tab_val.canvas.clear_plot()
        ax.plot(val.index, val["dist_sma"], label="Distancia a SMA200")
        ax.axhline(-0.05, linestyle=":", label="Barata ≤ -5%")
        ax.axhline(+0.05, linestyle=":", label="Cara ≥ +5%")
        ax.legend(loc="best"); self.tab_val.canvas.draw_plot()

        # OBV
        obv = calc_obv(base)
        ax = self.tab_obv.canvas.ax; self.tab_obv.canvas.clear_plot()
        ax.plot(obv.index, obv["OBV"], label="OBV")
        ax.legend(loc="best"); self.tab_obv.canvas.draw_plot()

        # MFI
        mfi = calc_mfi(base)
        ax = self.tab_mfi.canvas.ax; self.tab_mfi.canvas.clear_plot()
        ax.plot(mfi.index, mfi["MFI"], label="MFI (14)")
        ax.axhline(80, linestyle=":", label="Sobrecompra 80")
        ax.axhline(20, linestyle=":", label="Sobreventa 20")
        ax.legend(loc="best"); self.tab_mfi.canvas.draw_plot()


# ============================================================= #
# Contenedor de plugins (pestañas)
# ============================================================= #
class PluginTabsManager(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plugins: list[QtWidgets.QWidget] = []

    def add_plugin(self, plugin_widget: QtWidgets.QWidget):
        self._plugins.append(plugin_widget)
        name = getattr(plugin_widget, "name", plugin_widget.__class__.__name__)
        self.addTab(plugin_widget, name)

    def update_all(self, df: pd.DataFrame, ticker: str, ctx: dict):
        for p in self._plugins:
            try:
                p.update_data(df, ticker, ctx)
            except Exception as e:
                print(f"Plugin '{p}': {e}")
