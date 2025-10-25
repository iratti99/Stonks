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
from plugins import PluginTabsManager, PricePlugin, RSIPlugin, IndicatorsPlugin

# ============================ Plugins ============================ #
# Requiere plugins.py e indicadores_tabs.py en la misma carpeta
from plugins import PluginTabsManager, PricePlugin, RSIPlugin, IndicatorsPlugin

# ============================ Config ============================ #
WALLET_PATH = os.path.join(os.path.dirname(__file__), "wallet.txt")

# ============================ Utils ============================ #
def is_market_hours_ar(now_ar: datetime) -> tuple[bool, bool]:
    """
    Devuelve (en_horario, es_fin_de_semana).
    Mercado BYMA aprox. 11:00–17:30 hora local AR.
    """
    is_weekend = now_ar.weekday() >= 5  # 5=sábado, 6=domingo
    if is_weekend:
        return False, True
    in_hours = time(11, 0) <= now_ar.time() <= time(17, 30)
    return in_hours, False



def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _fmt_ars(v: float) -> str:
    return f"${v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ============================ Wallet ============================ #
def load_wallet(path: str) -> List[str]:
    """Lee wallet.txt: CSV por líneas, ignora comentarios tras ';'."""
    if not os.path.exists(path):
        return ["GGAL.BA", "YPFD.BA", "AL30.BA", "SPY"]
    tickers = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            part = line.split(";", 1)[0]
            raw = part.split(',')
            tickers.extend([t.strip() for t in raw])
    seen = set(); out = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out or ["SPY"]


# ============================ Datos: Yahoo / Placeholder ============================ #
def fetch_yahoo(ticker: str, interval: str, period: str) -> Tuple[pd.DataFrame, str]:
    """Devuelve df OHLCV con DatetimeIndex UTC y la moneda nativa (asume USD)."""
    try:
        data = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return pd.DataFrame(), ""
        if not isinstance(data.index, pd.DatetimeIndex):
            data.reset_index(inplace=True)
            data.rename(columns={"Datetime": "Date"}, inplace=True)
            data.set_index("Date", inplace=True)
        data.index = pd.to_datetime(data.index, utc=True)
        return data, "USD"
    except Exception as e:
        print("Yahoo error:", e)
        return pd.DataFrame(), ""


def fetch_investing_like(ticker: str, interval: str, period: str) -> Tuple[pd.DataFrame, str]:
    """Placeholder compatible (puedes sustituir por otra fuente)."""
    return fetch_yahoo(ticker, interval, period)


# ============================ USD rates ============================ #
def fetch_usd_rates() -> Dict[str, float]:
    """Devuelve {'CCL', 'MEP', 'OFICIAL'} en ARS/USD. Usa dos fuentes con fallback."""
    try:
        r = requests.get("https://dolarapi.com/v1/dolares", timeout=8)
        if r.ok:
            data = r.json(); out = {}
            for item in data:
                n = item.get("casa", "").upper()
                v = float(item.get("venta") or item.get("value") or 0.0)
                if "CCL" in n or "CONTADO CON LIQ" in n:
                    out["CCL"] = v
                elif "MEP" in n or "BOLSA" in n:
                    out["MEP"] = v
                elif "OFICIAL" in n:
                    out["OFICIAL"] = v
            if out:
                return out
    except Exception:
        pass
    try:
        r = requests.get("https://criptoya.com/api/dolar", timeout=8)
        if r.ok:
            data = r.json()
            return {
                "CCL": float(data.get("ccb","{}") and data["ccb"].get("ask", 0.0)) or float(data.get("ccl","{}") and data["ccl"].get("ask", 0.0)),
                "MEP": float(data.get("mep","{}") and data["mep"].get("ask", 0.0)),
                "OFICIAL": float(data.get("oficial","{}") and data["oficial"].get("venta", 0.0))
            }
    except Exception:
        pass
    return {}


# ============================ Worker ============================ #
class DataWorker(QThread):
    data_ready = pyqtSignal(object, str, str)  # (df, ticker_name, currency_native)
    error = pyqtSignal(str)

    def __init__(self, provider: str, ticker: str, interval: str, period: str):
        super().__init__()
        self.provider = provider
        self.ticker = ticker
        self.interval = interval
        self.period = period

    def run(self):
        try:
            if self.provider == "Yahoo":
                df, curr = fetch_yahoo(self.ticker, self.interval, self.period)
            else:
                df, curr = fetch_investing_like(self.ticker, self.interval, self.period)
            if df.empty:
                self.error.emit("Sin datos (ticker/intervalo/periodo)")
                return
            df = df.copy(); df.index = pd.to_datetime(df.index)
            df.rename(columns={"Close":"Close"}, inplace=True)
            self.data_ready.emit(df, self.ticker, curr or "USD")
        except Exception as e:
            self.error.emit(f"Error al recuperar datos: {e}")


# ============================ Main Window ============================ #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RSI Monitor - EconometricaGPT (modular)")
        self.resize(1280, 960)

        # Proveedor / Wallet
        self.provider_combo = QtWidgets.QComboBox(); self.provider_combo.addItems(["Yahoo", "Investing (exp)"])
        self.wallet_combo = QtWidgets.QComboBox(); self.btn_reload_wallet = QtWidgets.QPushButton("Recargar wallet")

        # Intervalo / Periodo / Refresco
        self.interval_combo = QtWidgets.QComboBox(); self.interval_combo.addItems(["1m","5m","15m","30m","60m","1d"]); self.interval_combo.setCurrentIndex(3)
        self.period_combo = QtWidgets.QComboBox(); self.period_combo.addItems(["1d","5d","1mo","3mo","6mo","1y","2y","5y","max"]); self.period_combo.setCurrentIndex(6)
        self.refresh_spin = QtWidgets.QSpinBox(); self.refresh_spin.setRange(5, 3600); self.refresh_spin.setValue(20); self.refresh_spin.setSuffix(" s")

        # Unidades
        self.unit_combo = QtWidgets.QComboBox(); self.unit_combo.addItems(["ARS","USD"]); self.unit_combo.setCurrentIndex(1)

        # Proyección
        self.horizon_combo = QtWidgets.QComboBox(); self.horizon_combo.addItems(["1m","1h","1d"])
        self.model_combo = QtWidgets.QComboBox(); self.model_combo.addItems(["Lineal","Holt-Winters","ARIMA"])
        self.window_spin = QtWidgets.QSpinBox(); self.window_spin.setRange(10, 10000); self.window_spin.setValue(200); self.window_spin.setSuffix(" pts")
        self.chk_show_model = QtWidgets.QCheckBox("Mostrar regresión/proyección"); self.chk_show_model.setChecked(True)

        self.btn_start = QtWidgets.QPushButton("Iniciar"); self.btn_stop = QtWidgets.QPushButton("Detener"); self.btn_stop.setEnabled(False)

        # Título e info
        self.instrument_title = QtWidgets.QLabel("Instrumento: -"); f = self.instrument_title.font(); f.setPointSize(14); f.setBold(True); self.instrument_title.setFont(f)
        self.instrument_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.info_line = QtWidgets.QLabel("RSI: - | Tendencia: - | Último: - | Δ: - | Barra: -"); self.info_line.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Panel USD
        self.usd_group = QtWidgets.QGroupBox("USD (ARS por USD)")
        self.lbl_usd_all = QtWidgets.QLabel("CCL: - | MEP: - | OFICIAL: -")
        self.lbl_usd_pick = QtWidgets.QLabel("Usando: -")
        usd_layout = QtWidgets.QHBoxLayout(); usd_layout.addWidget(self.lbl_usd_all); usd_layout.addStretch(); usd_layout.addWidget(self.lbl_usd_pick)
        self.usd_group.setLayout(usd_layout)

        # Latencia
        self.lat_group = QtWidgets.QGroupBox("Latencia")
        self.lbl_lag = QtWidgets.QLabel("Lag (s): -")
        self.lbl_now_ar = QtWidgets.QLabel("Hora AR: --:--"); self.lbl_now_ar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lat_layout = QtWidgets.QGridLayout(); lat_layout.addWidget(self.lbl_lag, 0, 0); lat_layout.addWidget(QtWidgets.QLabel("Hora Argentina:"), 0, 1); lat_layout.addWidget(self.lbl_now_ar, 0, 2)
        self.lat_group.setLayout(lat_layout)

        # ----- Layout superior -----
        top_controls = QtWidgets.QGridLayout()
        top_controls.addWidget(QtWidgets.QLabel("Proveedor:"), 0, 0); top_controls.addWidget(self.provider_combo, 0, 1)
        top_controls.addWidget(QtWidgets.QLabel("Wallet:"), 0, 2); top_controls.addWidget(self.wallet_combo, 0, 3)
        top_controls.addWidget(self.btn_reload_wallet, 0, 4)
        top_controls.addWidget(QtWidgets.QLabel("Unidades:"), 0, 5); top_controls.addWidget(self.unit_combo, 0, 6)

        top_controls.addWidget(QtWidgets.QLabel("Intervalo:"), 1, 0); top_controls.addWidget(self.interval_combo, 1, 1)
        top_controls.addWidget(QtWidgets.QLabel("Periodo:"), 1, 2); top_controls.addWidget(self.period_combo, 1, 3)
        top_controls.addWidget(QtWidgets.QLabel("Refresco:"), 1, 4); top_controls.addWidget(self.refresh_spin, 1, 5)

        top_controls.addWidget(QtWidgets.QLabel("Proyección:"), 2, 0); top_controls.addWidget(self.horizon_combo, 2, 1)
        top_controls.addWidget(QtWidgets.QLabel("Modelo:"), 2, 2); top_controls.addWidget(self.model_combo, 2, 3)
        top_controls.addWidget(QtWidgets.QLabel("Ventana:"), 2, 4); top_controls.addWidget(self.window_spin, 2, 5)
        top_controls.addWidget(self.chk_show_model, 2, 6)
        top_controls.addWidget(self.btn_start, 2, 7); top_controls.addWidget(self.btn_stop, 2, 8)

        # ----- Centro con Plugins (pestañas) -----
        central = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(central)
        v.addLayout(top_controls)
        v.addWidget(self.usd_group)
        v.addWidget(self.instrument_title)
        v.addWidget(self.info_line)

        self.plugins = PluginTabsManager(parent=self)
        self.plugins.add_plugin(PricePlugin())
        self.plugins.add_plugin(RSIPlugin())
        try:
            self.plugins.add_plugin(IndicatorsPlugin())  # requiere indicadores_tabs.py
        except Exception as e:
            # Si no está disponible, el resto sigue funcionando
            print("Indicadores deshabilitados:", e)
            
        v.addWidget(self.plugins, stretch=1)
        v.addWidget(self.lat_group)
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
            self.unit_combo, self.window_spin, self.chk_show_model
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

        # Inicializaciones
        self.reload_wallet(initial=True)
        self.refresh_usd_rates()

    # ---- helpers ----
    def reload_wallet(self, initial=False):
        tickers = load_wallet(WALLET_PATH)
        self.wallet_combo.clear(); self.wallet_combo.addItems(tickers)
        if initial and tickers:
            self.wallet_combo.setCurrentIndex(0)

    def _interval_minutes(self, txt: str) -> int:
        return {"1m":1,"5m":5,"15m":15,"30m":30,"60m":60,"1d":1440}.get(txt,1)

    def _horizon_to_steps(self, interval_txt: str, horizon_txt: str) -> int:
        int_min = self._interval_minutes(interval_txt); hor_min = {"1m":1,"1h":60,"1d":1440}[horizon_txt]
        return max(1, int(np.ceil(hor_min/int_min)))

    # ---- timers ----
    def start(self):
        if self.worker and self.worker.isRunning():
            return
        ticker = self.wallet_combo.currentText().strip() or "SPY"
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        provider = self.provider_combo.currentText()
        interval = self.interval_combo.currentText(); period = self.period_combo.currentText()
        self._last_interval_txt = interval

        self.worker = DataWorker(provider, ticker, interval, period)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.error.connect(self.on_error)
        self.worker.start()

        self.timer.setInterval(max(5, self.refresh_spin.value())*1000); self.timer.start()
        self.latency_timer.start(); self.usd_timer.start()

    def refresh_data(self):
        if not self.btn_stop.isEnabled():
            return
        ticker = self.wallet_combo.currentText().strip() or "SPY"
        provider = self.provider_combo.currentText()
        interval = self.interval_combo.currentText(); period = self.period_combo.currentText()
        self._last_interval_txt = interval
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption(); self.worker.wait(100)
        self.worker = DataWorker(provider, ticker, interval, period)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def refresh_latency_only(self):
        # Latencia = segundos desde la última barra recibida hasta ahora (UTC)
        if self._last_bar_utc is None:
            self.lbl_lag.setText("Lag (s): -")
        else:
            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            lag = max(0.0, (now_utc - self._last_bar_utc).total_seconds())
            self.lbl_lag.setText(f"Lag (s): {lag:.0f}")
        now_ar = datetime.now().astimezone()
        self.lbl_now_ar.setText(now_ar.strftime("%H:%M:%S"))
        self._update_market_clock_color()

    def refresh_usd_rates(self):
        def _fmt_or_na(v: float) -> str:
            return "N/A" if v is None or v <= 0 else _fmt_ars(v)

        try:
            self.usd_rates = fetch_usd_rates()
        except Exception:
            self.usd_rates = {}

        ccl = float(self.usd_rates.get("CCL", 0.0) or 0.0)
        mep = float(self.usd_rates.get("MEP", 0.0) or 0.0)
        ofi = float(self.usd_rates.get("OFICIAL", 0.0) or 0.0)

        self.lbl_usd_all.setText(f"CCL: {_fmt_or_na(ccl)} | MEP: {_fmt_or_na(mep)} | OFICIAL: {_fmt_or_na(ofi)}")

        self.usd_name, self.usd_value = self._pick_usd()
        if self.usd_value > 0:
            self.lbl_usd_pick.setText(f"Usando: {self.usd_name} ({_fmt_ars(self.usd_value)})")
        else:
            self.lbl_usd_pick.setText("Usando: N/A")
        
    def _update_market_clock_color(self):
        now_ar = datetime.now().astimezone()
        in_hours, is_weekend = is_market_hours_ar(now_ar)
        if is_weekend:
            color = "gray"   # fin de semana
        else:
            color = "green" if in_hours else "red"  # entre semana: verde/rojo
        self.lbl_now_ar.setStyleSheet(f"color: {color};")

        
    def stop(self):
        # Rehabilito botones
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # Detengo timers
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.latency_timer.stop()
        except Exception:
            pass
        try:
            self.usd_timer.stop()
        except Exception:
            pass

        # Detengo el worker si sigue corriendo
        try:    
            if self.worker and self.worker.isRunning():
                self.worker.requestInterruption()
                self.worker.wait(1000)
        except Exception:
            pass


    def _pick_usd(self):
        if not self.usd_rates:
            return "N/A", 0.0
        for k in ("CCL", "MEP", "OFICIAL"):
            v = float(self.usd_rates.get(k, 0.0) or 0.0)
            if v > 0:
                return k, v
        return "N/A", 0.0


    # ---- data callbacks ----
    def on_data_ready(self, df: pd.DataFrame, ticker_name: str, currency_native: str):
        self.last_df = df; self.last_name = ticker_name; self.currency_native = currency_native
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                self._last_bar_utc = df.index[-1].to_pydatetime().astimezone(timezone.utc)
            else:
                self._last_bar_utc = None
        except Exception:
            self._last_bar_utc = None

        # Conversión ARS/USD si corresponde (solo para la línea informativa)
        unit = self.unit_combo.currentText()
        close = df["Close"].astype(float)
        if unit == "ARS" and self.usd_value > 0 and (self.currency_native or "USD").upper() == "USD":
            close_vis = close * float(self.usd_value)
        elif unit == "USD" and self.usd_value > 0 and (self.currency_native or "USD").upper() == "ARS":
            close_vis = close / float(self.usd_value)
        else:
            close_vis = close

        rsi = compute_rsi(close)

        # Actualizar plugins
        ctx = {
            "unit": unit,
            "model_name": self.model_combo.currentText(),
            "forecast_steps": self._horizon_to_steps(self._last_interval_txt, self.horizon_combo.currentText()),
            "window": self.window_spin.value(),
            "show_model": self.chk_show_model.isChecked(),
            # Puedes pasar más claves y serán visibles para todos los plugins
        }
        self.plugins.update_all(df, ticker_name, ctx)

        # Línea de estado compacta
        try:
            last = float(close_vis.iloc[-1]); prev = float(close_vis.iloc[-2]) if len(close_vis) > 1 else last
            delta = last - prev
            rsi_last = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan
            state = "Sobreventa" if rsi_last <= 30 else ("Sobrecompra" if rsi_last >= 70 else "Neutral")
            txt_last = f"{('US$' if unit=='USD' else '$')}{last:,.2f}".replace(",","X").replace(".",",").replace("X",".")
            txt_delta = f"{delta:+,.2f}".replace(",","X").replace(".",",").replace("X",".")
            bar = self._last_bar_utc.astimezone().strftime("%Y-%m-%d %H:%M") if self._last_bar_utc else "-"
            self.info_line.setText(f"RSI: {rsi_last:5.1f} ({state}) | Tendencia: {ctx['model_name']} | Último: {txt_last} | Δ: {txt_delta} | Barra: {bar}")
        except Exception:
            self.info_line.setText("RSI: - | Tendencia: - | Último: - | Δ: - | Barra: -")

    def on_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Datos", msg)


# ============================ Main ============================ #
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
