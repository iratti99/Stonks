# 🧠 Dummy Monitor

**Dummy Monitor** es una aplicación de monitoreo financiero y técnico en tiempo real, desarrollada en **Python 3.11** utilizando **PyQt6** y **Matplotlib**. Permite visualizar la evolución de precios, calcular el índice **RSI (Relative Strength Index)**, realizar proyecciones mediante modelos de regresión, y ajustar las unidades entre **USD** y **ARS**, considerando el tipo de cambio más caro entre **CCL**, **MEP** y **Oficial**.

---

## ⚙️ Requisitos e instalación

### 📦 Dependencias

Asegurate de tener **Python 3.11+** instalado y ejecutá:

```bash
pip install PyQt6 matplotlib yfinance numpy pandas statsmodels pytz requests
```

### 🗂️ Estructura del proyecto

```text
DummyMonitor/
├── rsi_qt_app_v4.py           # Script principal
├── wallet.txt                 # Lista de instrumentos a seguir
├── images/                    # Capturas para el README
│   ├── 1_params.png
│   ├── 2_dollar.png
│   ├── 3_price_rsi.png
│   ├── 4_price_rsi_2.png
│   └── 5_latency.png
└── README.md                  # Este archivo
```

---

## 💼 Configuración del archivo `wallet.txt`

El archivo `wallet.txt` define los instrumentos disponibles para seguimiento.

### 🧾 Formato

```txt
AAPL, MSFT, SLV, GGAL.BA, YPFD.BA;
```

### 🧩 Reglas

- Los instrumentos se separan por **coma (,)**
- El punto y coma **(;)** indica el final de la cartera
- Los tickers con **`.BA`** corresponden al **Merval (ARS)**
- Los tickers sin `.BA` se asumen en **USD (CEDEARs o internacionales)**

📘 **Ejemplo válido:**
```txt
SLV, AAPL, YPFD.BA, GGAL.BA;
```

---

## 🖥️ Interfaz principal

### Panel de configuración

![Parámetros](images/1_params.png)

**Elementos:**
- **Proveedor:** Fuente de datos (Yahoo o Investing)
- **Wallet:** Lista de instrumentos cargada desde `wallet.txt`
- **Recargar wallet:** Actualiza la lista de instrumentos sin reiniciar la app
- **Intervalo:** Frecuencia de las velas (1m, 5m, 1d...)
- **Periodo:** Rango de historia descargado
- **Refresco:** Tiempo de actualización automática (por defecto 5 segundos)
- **Proyección / Modelo / Ventana:** Controlan la longitud y tipo de regresión aplicada
- **Unidades:** USD o ARS (según tipo de cambio de referencia)
- **Mostrar regresión/proyección:** Activa o desactiva el modelo predictivo

---

### Panel de cotización del dólar

![Dólar](images/2_dollar.png)

La app obtiene los valores **públicos** de los tres principales tipos de cambio:

- 💸 **CCL** (Contado con Liquidación)
- 💵 **MEP** (Bolsa)
- 💰 **Oficial**

Luego selecciona el **más alto** y lo muestra como referencia:
> Ejemplo: `Usando: CCL ($1.569,00)`

Este tipo de cambio se usa automáticamente para convertir los precios a **ARS** si las unidades seleccionadas no son USD.

---

## 📈 Visualización de precios y RSI

![Precio y RSI](images/3_price_rsi.png)

Cada instrumento se representa con dos gráficos:

- **Superior:** Precio (Close)
- **Inferior:** RSI (14)

📊 **Líneas de referencia RSI:**
- 70 / 30 (rojo): zonas de sobrecompra y sobreventa
- 55 / 45 (azul): zona neutral

🔹 Los valores máximos se marcan en **verde**
🔹 Los mínimos se marcan en **rojo**

El título muestra tres indicadores:
```
Precio (USD): US$47,72 | Señal: Sobrecompra (70.2) | Tendencia RSI: alcista (pend=1.238, R²=0.28)
```

---

## ⏱️ Latencia y hora local

![Latencia](images/5_latency.png)

El bloque inferior muestra la **latencia** (tiempo que tarda el dato en reflejarse en la interfaz) y la **hora argentina actual**.

### 🔹 LAG
Tiempo transcurrido entre la última barra descargada y el reloj local.

### 🔹 Color horario
- 🟢 Verde: dentro del horario bursátil argentino (10:30–17:00)
- 🔴 Rojo: fuera del horario

El reloj se actualiza en tiempo real (cada 1s) según la zona horaria **America/Argentina/Buenos_Aires**.

---

## 🧮 Indicadores debajo del instrumento

Justo debajo del título principal del instrumento aparecen tres valores esenciales:

- **Precio (USD o ARS)** — Valor actual de cierre
- **Señal RSI** — Estado de sobrecompra, sobreventa o neutral
- **Tendencia RSI** — Pendiente y R² del ajuste lineal sobre los últimos puntos del RSI

Ejemplo:
```
Precio (USD): US$47,72 | Señal: Sobrecompra (70.2) | Tendencia RSI: alcista (pend=1.238, R²=0.28)
```

---

## 🚀 Ejemplo de uso

1. Editá el archivo `wallet.txt` con los instrumentos deseados.
2. Ejecutá la aplicación:
   ```bash
   python rsi_qt_app_v4.py
   ```
3. Seleccioná el proveedor, instrumento y unidad (USD/ARS).
4. Presioná **Iniciar** para comenzar el monitoreo.
5. Observá en tiempo real el RSI, las señales técnicas y la latencia.

---

## 🧠 Notas técnicas

- **Latencia real:** se calcula como la diferencia entre la hora UTC actual y el timestamp de la última barra de datos.
- **Datos:** actualmente la fuente predeterminada es **Yahoo Finance**, con opción futura a Investing.com.
- **Zonas horarias:** todos los datos se convierten a **UTC-3 (Argentina)** antes de graficarse.
- **Por defecto:**
  - Unidad: **USD**
  - Refresco: **5 s**
  - Modelo de proyección: **Lineal**

---

📅 **Autor:** Proyecto EconometricaGPT
🏦 **Entorno:** Python 3.11 (Windows)
📊 **Licencia:** Uso educativo y de investigación

