"""
Parche modular para integrar en tu app PyQt6 (rsi_qt_app_v4.py) nuevas pestañas con:
- MACD
- SMA20/50/200
- Volumen diario
- Señal Caro/Barato según distancia a SMA (configurable)
- OBV
- MFI

Uso sugerido:
1) Copia este archivo como `indicadores_tabs.py` en la misma carpeta de tu app.
2) En tu `rsi_qt_app_v4.py`:
   from indicadores_tabs import IndicatorTabs
3) Donde ya obtengas el DataFrame `df` (con columnas ['Open','High','Low','Close','Volume'])
   y el string `ticker`, crea/actualiza las pestañas:
   self.ind_tabs = self.ind_tabs or IndicatorTabs(parent=self)
   self.ind_tabs.update_data(df, ticker)
   self.right_layout.addWidget(self.ind_tabs)  # o donde corresponda

Este módulo NO bloquea tu diseño actual: es un widget QTabWidget que puedes añadir donde te convenga.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================= Indicadores ============================= #

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = {"Open", "High", "Low", "Close", "Volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df: {missing}")
    # Asegurar orden de columnas (no imprescindible)
    return df.copy()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def calc_sma(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
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
    # OBV acumula +Volume si Close sube vs cierre previo, -Volume si baja
    sign = np.sign(out["Close"].diff().fillna(0.0))
    out["OBV"] = (sign * out["Volume"]).cumsum()
    return out


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0  # Typical Price
    rmf = tp * out["Volume"].astype(float)  # Raw Money Flow
    # Flujos positivos/negativos según variación del TP
    tp_delta = tp.diff()
    pos_mf = rmf.where(tp_delta > 0, 0.0)
    neg_mf = rmf.where(tp_delta < 0, 0.0)
    pmf = pos_mf.rolling(period, min_periods=period).sum()
    nmf = neg_mf.rolling(period, min_periods=period).sum().abs()
    money_ratio = pmf / nmf
    out["MFI"] = 100 - (100 / (1 + money_ratio))
    return out


@dataclass
class ValuationConfig:
    sma_window: int = 200  # SMA de referencia
    cheap_thr: float = -0.05  # -5% o menos => Barata
    expensive_thr: float = 0.05  # +5% o más => Cara


def calc_valuation(df: pd.DataFrame, cfg: ValuationConfig = ValuationConfig()) -> pd.DataFrame:
    out = df.copy()
    out[f"SMA{cfg.sma_window}"] = out["Close"].rolling(cfg.sma_window, min_periods=cfg.sma_window).mean()
    out["dist_sma"] = (out["Close"] / out[f"SMA{cfg.sma_window}"] - 1.0)
    def tag(x):
        if np.isnan(x):
            return "N/A"
        if x <= cfg.cheap_thr:
            return "Barata"
        if x >= cfg.expensive_thr:
            return "Cara"
        return "Neutral"
    out["ValuTag"] = out["dist_sma"].apply(tag)
    return out


# ============================= Widgets de gráfico ============================= #

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.title = title
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()

    def draw(self):
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


# ============================= Tabs contenedor ============================= #

class IndicatorTabs(QtWidgets.QTabWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        # Widgets por pestaña
        self.tab_sma = PlotWidget("SMA 20/50/200")
        self.tab_macd = PlotWidget("MACD (12,26,9)")
        self.tab_vol = PlotWidget("Volumen diario")
        self.tab_val = QtWidgets.QWidget()
        self.tab_obv = PlotWidget("On-Balance Volume (OBV)")
        self.tab_mfi = PlotWidget("Money Flow Index (14)")

        # Layout para la pestaña de valoración (con texto + mini gráfico)
        val_layout = QtWidgets.QVBoxLayout(self.tab_val)
        self.val_label = QtWidgets.QLabel("—")
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.val_plot = PlotWidget("Distancia vs SMA de referencia")
        val_layout.addWidget(self.val_label)
        val_layout.addWidget(self.val_plot)

        # Añadir pestañas
        self.addTab(self.tab_sma, "SMA")
        self.addTab(self.tab_macd, "MACD")
        self.addTab(self.tab_vol, "Volumen")
        self.addTab(self.tab_val, "Caro/Barato (SMA)")
        self.addTab(self.tab_obv, "OBV")
        self.addTab(self.tab_mfi, "MFI")

        self._cfg_val = ValuationConfig()
        self._ticker = ""

    # ------------------------- API pública ------------------------- #
    def update_data(self, df: pd.DataFrame, ticker: str):
        """Actualiza todas las pestañas con un DataFrame OHLCV y nombre de ticker."""
        self._ticker = ticker
        base = ensure_cols(df)

        # SMA
        sma = calc_sma(base)
        ax = self.tab_sma.ax; self.tab_sma.clear()
        ax.plot(sma.index, sma["Close"], label=f"{ticker} Close")
        for w in (20, 50, 200):
            ax.plot(sma.index, sma[f"SMA{w}"], label=f"SMA{w}")
        ax.legend(loc="best")
        self.tab_sma.draw()

        # MACD
        macd = calc_macd(base)
        ax = self.tab_macd.ax; self.tab_macd.clear()
        ax.plot(macd.index, macd["MACD"], label="MACD")
        ax.plot(macd.index, macd["Signal"], label="Signal", linestyle=":")
        ax.bar(macd.index, macd["Hist"], alpha=0.4, label="Histograma")
        ax.legend(loc="best")
        self.tab_macd.draw()

        # Volumen diario (barras)
        ax = self.tab_vol.ax; self.tab_vol.clear()
        ax.bar(base.index, base["Volume"], label="Volumen")
        ax.legend(loc="best")
        self.tab_vol.draw()

        # Caro/Barato según SMA de referencia
        val = calc_valuation(base, self._cfg_val)
        last = val.dropna().iloc[-1] if not val.dropna().empty else None
        tag = last.get("ValuTag", "N/A") if last is not None else "N/A"
        dist = last.get("dist_sma", np.nan) if last is not None else np.nan
        thr_c = self._cfg_val.cheap_thr; thr_e = self._cfg_val.expensive_thr
        sma_w = self._cfg_val.sma_window
        self.val_label.setText(
            f"<b>{ticker}</b> — Estado: <b>{tag}</b> (distancia a SMA{sma_w}: {dist:.2%} | umbrales: ≤{thr_c:.0%}=Barata, ≥{thr_e:.0%}=Cara)"
        )
        ax = self.val_plot.ax; self.val_plot.clear()
        ax.plot(val.index, val["dist_sma"], label="Distancia a SMA")
        ax.axhline(thr_c, linestyle=":")
        ax.axhline(thr_e, linestyle=":")
        ax.legend(loc="best")
        self.val_plot.draw()

        # OBV
        obv = calc_obv(base)
        ax = self.tab_obv.ax; self.tab_obv.clear()
        ax.plot(obv.index, obv["OBV"], label="OBV")
        ax.legend(loc="best")
        self.tab_obv.draw()

        # MFI
        mfi = calc_mfi(base)
        ax = self.tab_mfi.ax; self.tab_mfi.clear()
        ax.plot(mfi.index, mfi["MFI"], label="MFI (14)")
        ax.axhline(80, linestyle=":", label="Sobrecompra 80")
        ax.axhline(20, linestyle=":", label="Sobreventa 20")
        ax.legend(loc="best")
        self.tab_mfi.draw()

    # ------------------------- Opciones ------------------------- #
    def set_valuation_cfg(self, sma_window: Optional[int] = None,
                          cheap_thr: Optional[float] = None,
                          expensive_thr: Optional[float] = None):
        if sma_window is not None:
            self._cfg_val.sma_window = int(sma_window)
        if cheap_thr is not None:
            self._cfg_val.cheap_thr = float(cheap_thr)
        if expensive_thr is not None:
            self._cfg_val.expensive_thr = float(expensive_thr)


# ============================= Demo mínima independiente ============================= #
if __name__ == "__main__":
    import sys
    import yfinance as yf

    app = QtWidgets.QApplication(sys.argv)

    # Ventana de prueba
    w = QtWidgets.QMainWindow()
    w.setWindowTitle("Indicadores Técnicos — Demo")
    central = QtWidgets.QWidget(); w.setCentralWidget(central)
    lay = QtWidgets.QVBoxLayout(central)

    tabs = IndicatorTabs()
    lay.addWidget(tabs)

    # Datos de ejemplo
    ticker = "AAPL"
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})

    tabs.update_data(df, ticker)

    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec())
