# TPT: Scalping Trading Model (BTCUSDT 1m) 🚀

📚 Overview
The TPT Scalping Trading Model is a cutting-edge, high-frequency trading system tailored for the BTCUSDT pair on a 1-minute timeframe. It harnesses a hybrid architecture that blends a Transformer neural network for precise market trend forecasting with Proximal Policy Optimization (PPO) reinforcement learning for optimized trade execution. Designed for the volatile cryptocurrency market, this model thrives on capturing short-term price movements through rapid, frequent trades. A robust suite of technical indicators and a realistic trading environment make it a powerful tool for scalping enthusiasts.

💡 Note: This open-source model is shared for educational and networking purposes. Premium models, boasting win rates up to 40–60%, drawdowns below 20%, and adaptive strategies for diverse market conditions, are available for purchase. Contact **TPTBusiness@proton.me** for inquiries about premium models, customizations, or licensing.
Alternative contact option via session (https://getsession.org/) 058777f00166f75c89f315c687340bfffc90e6a605bea2a8c0f538779885b67179

## ✨ Features

🤖 Hybrid Architecture: Combines a Transformer neural network for sequence-based trend prediction with PPO reinforcement learning for dynamic trade decisions (buy/sell, position sizing, stop-loss/take-profit, leverage).
📊 Technical Indicators:
RSI: Detects overbought/oversold conditions (window=14).
MACD: Captures momentum shifts (fast=12, slow=26, signal=9).
SMA/EMA: Smooths price trends for trend-following signals (window=20).
ATR: Measures volatility to guide risk management (window=14).
Bollinger Bands: Identifies volatility ranges for breakouts (window=20, deviation=2).
VWAP: Provides a volume-weighted price reference (window=14).
Volume Profile: Tracks cumulative volume (window=20).
Fibonacci Levels: Pinpoints support/resistance levels (window=20).


⚡ Scalping Strategy: Targets rapid trades with a high risk-reward ratio, capitalizing on small price movements in minutes.
🏦 Realistic Trading Environment: Simulates real-world trading with fees, leverage, margin constraints, and position limits.
🔧 Hyperparameter Optimization: Fine-tuned parameters with checkpointing for robust performance.


## 📈 Performance Metrics
The model was backtested from 2025-05-27 14:35:00 to 2025-06-10 11:54:00 (13.89 days), yielding:


- Winrate: 13.14% (typical for scalping)
- Risk-Reward Ratio: 13.06 (strong gains per trade)
- Net Profit: 2208.99 (from 3000 capital)
- Profit Factor: 1.975 (profitable but risky)
- Max Drawdown: 30.21% (1069.02 absolute)
- Average Win: 26.01
- Average Loss: -1.99
- Trade Frequency: 94.25 trades
- Average Trade Duration: 17.07 minutes
- Average Leverage: 4.32
- Average Stop Loss: 3.0%
- Average Take Profit: 4.0%

## 🔍 Backtest Process

The backtest evaluates the TPT Scalping Trading Model's performance using historical BTCUSDT 1-minute Kline data, simulating real-world trading conditions without risking capital. It tests three model variants—PPO-only, Transformer-only, and Hybrid (Transformer + PPO)—to provide a comprehensive assessment. Below is a step-by-step overview of the backtest process for transparency:

1. **Data Preparation**:
   - Loads historical Kline data (OHLCV: open, high, low, close, volume) from `data/klines/klines_BTCUSDT_default.parquet` or Binance API (via CCXT/python-binance).
   - Limits data to ~1,000,000 rows (~2 years of 1m data) for efficiency.
   - Converts timestamps to datetime, handling invalid values with fallbacks (e.g., generated ranges).
   - Cleans data by removing NaNs, infinities, and zeros; ensures numeric types.
   - Computes technical indicators using TA-Lib: RSI (14), MACD (12/26/9), SMA/EMA (20), ATR (14), Bollinger Bands (20/2), VWAP (14), Volume Profile (20), Fibonacci Levels (20-period range).
   - Normalizes features using a `StandardScaler` (from `scaler.pkl` or created).

2. **Model Loading**:
   - Loads the latest PPO checkpoint (`ppo_checkpoint_*.zip`) and Transformer model (`.pth`) from `backtesting/results/models/`.
   - Hybrid mode combines both, with Transformer predictions augmenting PPO inputs.

3. **Trading Environment**:
   - Uses a Gymnasium `TradingEnv` (vectorized for PPO/Hybrid, single for Transformer).
   - Configures with starting capital (3000), max leverage (100), stop-loss/take-profit (3%/4%), max positions (10), and sequence length (32).
   - Simulates trading with fees (maker 0.01%, taker 0.02%, funding 0.01%), margin, liquidation risks, and volatility-adjusted SL/TP via ATR.
   - Observations include 15 features plus Transformer predictions (sigmoid, up/down probability).
   - Actions are 5D: [trade_type (-1 short, +1 long), position_size (0-1 ratio), SL_pct, TP_pct, leverage (1-100)].
   - Rewards balance profit, risk/drawdown penalties, holding time, SL/TP efficiency, position management, leverage bonuses, and win rate incentives.

4. **Simulation**:
   - Resets environment to initial state.
   - Iterates through data:
     - Fetches 32-step sequence, scaled, with Transformer prediction.
     - Generates action: PPO (policy), Transformer (threshold-based), or Hybrid (augmented PPO).
     - Executes trades if margin/position limits allow; applies fees.
     - Updates positions, checking SL/TP/liquidation; logs trade details (entry/exit, profit, duration).
     - Hybrid mode periodically fine-tunes Transformer and PPO on recent data.
   - Closes all positions at data end.

5. **Metrics and Reporting**:
   - Calculates: winrate, risk-reward ratio, net profit, profit factor, max drawdown (%/abs), avg win/loss, trade frequency, avg duration/leverage/SL/TP, Sharpe ratio, Transformer precision/recall/F1.
   - Saves trades (JSON/Excel), metrics (JSON), parameters (JSON), and models in `backtesting/results/backtest_<timestamp>/<variant>`.
   - Generates PDF reports (per variant and combined) with equity curves, trade details, and metrics.

**Transparency Notes**:
- **Reproducibility**: Deterministic with fixed data; model stochasticity may vary results.
- **Limitations**: Simulates slippage/latency via factors; assumes optimistic fills; no weekend gaps; risks overfitting.
- **Live Trading**: Results may differ due to execution delays and liquidity. Premium models enhance adaptability.

For custom backtests or live trading setups, refer to `example_simulation.py` or contact **TPTBusiness@proton.me**.

**⚠️ Disclaimer: These metrics are from a backtest and may vary in live trading due to market dynamics, slippage, and latency. Premium models offer win rates of 40–60%, drawdowns below 20%, and enhanced adaptability. Contact TPTBusiness@proton.me for details.**


**⚙️ Configuration**

- Symbol: BTCUSDT
- Timeframe: 1-minute
- Starting Capital: 3000
- Max Leverage: 100
- Max Stop Loss: 3%
- Max Take Profit: 4%
- Max Open Positions: 10
- Sequence Length: 32 (Transformer input)
- Features: open, high, low, close, volume, rsi, macd, sma, ema, atr, bb_upper, bb_lower, vwap, volume_profile, fibonacci_levels
- Backtest Period: 2025-05-27 14:35:00 to 2025-06-10 11:54:00 (13.89 days)


**🚀 How It Works**
The TPT Scalping Trading Model is a state-of-the-art system designed to excel in the fast-paced cryptocurrency market. Here’s a high-level overview of its core components:

🔍 Market Analysis with Transformer Neural Network
The model employs a **Transformer neural network** to analyze 32-step sequences of market data, incorporating 15 technical indicators. It predicts short-term price trends with high accuracy, providing a probability of upward price movements. This predictive power enables the model to identify high-probability trading opportunities in volatile conditions.

# 🛠️ Setup and Directory Structure
To run the TPT Scalping Trading Model, you need to prepare specific data and model files and place them in the correct directory structure. The provided example_simulation.py script is a demonstration for backtesting the model with historical BTCUSDT 1-minute kline data.
Required Files
- Kline Data: Historical BTCUSDT 1-minute kline data in Parquet format (e.g., klines_BTCUSDT_default.parquet). Obtain this via CCXT or other data providers.
- PPO Model: Trained PPO model (ppo_checkpoint.zip) for reinforcement learning-based trade decisions.
- Transformer Model: Trained Transformer model (transformer_model.pth) for market trend prediction.
- Parameters File: Configuration file (parameters.json) defining trading parameters like feature_cols, start_capital, etc.
- Scaler (Optional): StandardScaler file (scaler.pkl) for data normalization. If missing, the script will create one automatically using the kline data.

**Directory Structure**
Place the files in the following structure:
```
TPT/
├── backtesting/
│   ├── results/
│   │   ├── models/
│   │   │   ├── ppo_checkpoint.zip
│   │   │   ├── transformer_model.pth
│   │   │   ├── scaler.pkl  # Optional, created if missing
│   │   ├── parameters.json
├── data/
│   ├── klines/
│   │   ├── klines_BTCUSDT_default.parquet
├── example_simulation.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
```
parameters.json:
```
{
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "start_capital": 3000,
    "max_leverage": 100,
    "max_stop_loss_pct": 0.03,
    "max_take_profit_pct": 0.04,
    "max_open_positions": 10,
    "min_profit_threshold": 0.0,
    "seq_length": 32,
    "feature_cols": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "macd",
        "sma",
        "ema",
        "atr",
        "bb_upper",
        "bb_lower",
        "vwap",
        "volume_profile",
        "fibonacci_levels"
    ],
    "ppo_model_path": "/home/nico/Dev/TPT-1/models/runs/2025-09-06_13-10-04/checkpoints/ppo_checkpoint_400000_steps.zip",
    "transformer_model_path": "/home/nico/Dev/TPT-1/models/runs/2025-09-06_11-42-18/checkpoints/transformer_checkpoint_epoch_52.pth",
    "backtest_start_time": "2025-05-27 14:35:00",
    "backtest_end_time": "2025-06-10 11:54:00",
    "total_time_days": 13.888194444444444
}
```
**Installation Steps**

Clone the Repository:```git clone https://github.com/TPTBusiness/TPT.git
cd TPT```


Create a Virtual Environment: ```python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate```


Install Dependencies:
```pip install -r requirements.txt```

See requirements.txt for the list of required libraries (pandas, numpy, torch, stable-baselines3, scikit-learn, tqdm, gymnasium, ccxt, ta).
Install TA-Lib:
Follow the TA-Lib installation guide for your system to enable technical indicator calculations.


**📊 Collecting Historical Kline Data**

To run the TPT Scalping Trading Model, you need historical 1-minute kline data for BTCUSDT in Parquet format, stored at `data/klines/klines_BTCUSDT_default.parquet`. The `utils/collect_data.py` script uses the `python-binance` library to fetch this data from Binance's Spot API and save it with the required columns: `timestamp`, `open`, `high`, `low`, `close`, and `volume`.

**Steps to Collect Data**

1. **Ensure Dependencies**:
   Ensure `python-binance`, `pandas`, and `pyarrow` are installed. These are not included in `requirements.txt` by default, so add them if necessary:
   ```bash
   echo -e "python-binance\npyarrow" >> requirements.txt
   pip install -r requirements.txt

2. **Prepare the Script:**
The utils/collect_data.py script is included in the repository. It fetches 30 days of historical 1-minute kline data for BTCUSDT by default (e.g., from 2025-08-08 to 2025-09-07, as future data like the backtest period 2025-05-27 to 2025-06-10 is unavailable). The data is saved to data/klines/klines_BTCUSDT_default.parquet.


3. **Run the Script:**
Activate the virtual environment and execute the script:
`source .venv/bin/activate`  # On Windows: `.venv\Scripts\activate`
`python utils/collect_data.py`
Expected Output:
- ✅ Initialized Binance Spot client for BTCUSDT 1m
- Fetching klines...
- ✅ Fetched 1000 klines, total: 1000
- ✅ Processed 43200 kline records from 2025-08-08 00:00:00 to 2025-09-07 23:59:00
- ✅ Saved kline data to: data/klines/klines_BTCUSDT_default.parquet
- ✅ Verified saved data: 43200 rows with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
- ✅ Binance client closed


4. **Verify the Data:**
Check that the Parquet file exists:
`ls data/klines/`
Expected output: klines_BTCUSDT_default.parquet
Verify the columns:
`import pandas as pd
df = pd.read_parquet('data/klines/klines_BTCUSDT_default.parquet')
print(df.columns)`  # Expected: ['timestamp', 'open', 'high', 'low', 'close', 'volume']


6. **Use with Backtest:**
The generated klines_BTCUSDT_default.parquet is compatible with example_simulation.py. Ensure it is in data/klines/ before running the backtest:
bashpython example_simulation.py

Prepare Data and Models:
Place kline data in data/klines/ (e.g., klines_BTCUSDT_default.parquet).
Place trained models in backtesting/results/models/ (ppo_checkpoint.zip, transformer_model.pth).
Create or place parameters.json in backtesting/results/.
Optionally, place scaler.pkl in backtesting/results/models/ (created automatically if missing).


Testing the Model
To test the model, run the example_simulation.py script, which executes a 1000-step backtest with a forced buy action at step 5 for demonstration purposes. The script logs trade details, rewards, and model predictions.

Ensure Files Are in Place:Verify that the following files are correctly placed:

- data/klines/klines_BTCUSDT_default.parquet
- backtesting/results/models/ppo_checkpoint.zip
- backtesting/results/models/transformer_model.pth
- backtesting/results/parameters.json
- backtesting/results/models/scaler.pkl (optional)


**Run the Backtest:**
```python example_simulation.py```

The script will:
- Load the models, parameters, and kline data.
- Create scaler.pkl if it does not exist.
- Run a 1000-step backtest, logging balance, profit, positions, and trade details.
- Output a summary of the simulation results (e.g., final balance, total profit, total trades).


Verify Output:Check the console output for logs like:
✅ New scaler created and saved to: backtesting/results/models/scaler.pkl
[Step 5] Forcing buy action: [0.0, 0.5, 0.01, 0.02, 50.0]
[Step 1000] Test run completed: Steps=1000, Final Balance=3208.99, Total Profit=2208.99, Total Trades=94



💡 Note: If you lack trained models or kline data, you must generate them. Refer to the CCXT documentation for fetching kline data or contact **TPTBusiness@proton.me** for guidance on training models.
📖 Usage

**Run the Simulation:**
```python example_simulation.py```

This executes a 1000-step backtest, forcing a buy action at step 5 for testing, and logs trade details, rewards, and model predictions. If scaler.pkl is missing, it will be created automatically using the kline data.

**Customize:**

Modify parameters.json in backtesting/results/ to adjust trading parameters (e.g., start_capital, max_leverage).
Extend the script for live trading by integrating a binance_client (see CCXT documentation).


**Premium Models: The provided example_simulation.py is a demonstration script. For access to premium models with 40–60% win rates, drawdowns below 20%, and live trading capabilities, contact TPTBusiness@proton.me.**



**🎯 Trade Optimization with PPO Reinforcement Learning**
Using **PPO reinforcement learning**, the model optimizes trading decisions by learning a policy that balances profit and risk. It dynamically determines:
- **Trade Type**: Long, short, or no trade based on market signals.
- **Position Sizing**: Allocates 1-3% of capital per trade.
- **Stop-Loss/Take-Profit**: Sets volatility-adjusted levels (0.5-3% for stop-loss, 0.5-4% for take-profit).
- **Leverage**: Applies up to 100x, constrained by volatility and margin.  
The model maximizes returns while minimizing drawdowns, prioritizing short holding times and effective risk management.



**🏦 Robust Trading Environment**
The model operates in a custom environment that mirrors real-world trading, incorporating:
- **Fees**: Maker/taker fees (0.01-0.02%) and funding rates (0.01%).
- **Position Limits**: Up to 10 simultaneous positions.
- **Margin and Leverage**: Enforces realistic margin and liquidation risks.
- **Volatility Adjustments**: Dynamically adjusts parameters based on market volatility.  
It supports backtesting and optional live trading with real-time data integration.



**⚡ Scalping Strategy**
Optimized for **scalping**, the model executes rapid trades (average duration: 17.07 minutes) to exploit small price movements. Its high risk-reward ratio (13.06) ensures significant profits despite a low win rate (13.14%), making it ideal for the volatile BTCUSDT market.



### 🌟 Premium Models Available
### This open-source model showcases the power of the Transformer-PPO hybrid approach, but premium models offer significant enhancements:

Higher Win Rates: Achieve 40–60% through advanced feature engineering and hyperparameter tuning.
Lower Drawdowns: Reduced to below 20% with superior risk management.
Adaptive Strategies: Dynamically adjusts to trending, ranging, or volatile markets.
Live Trading Optimization: Seamless real-time data integration and low-latency execution.
Customizations: Tailored for specific assets, timeframes, or risk profiles.

For pricing, licensing, or inquiries, contact **TPTBusiness@proton.me**.

**🛠️ Installation**
To set up the open-source model:

Clone the repository:```git clone https://github.com/TPTBusiness/TPT.git```


Install dependencies:```pip install torch stable-baselines3 pandas ta numpy```


Install TA-Lib for technical indicators (see TA-Lib documentation for setup).


**📖 Usage**
This repository is designed for educational and networking purposes. The core model demonstrates a powerful scalping framework, but its full implementation is proprietary. For access to the complete system, collaboration opportunities, or premium models with enhanced performance, contact **TPTBusiness@proton.me**.

**📜 License**
Licensed under GNU AGPL v3.0. Modifications must be open-sourced, and commercial use requires permission. Contact **TPTBusiness@proton.me** for commercial licensing inquiries.

**🙌 Acknowledgments**

Built with PyTorch, Stable-Baselines3, and TA-Lib.
Inspired by the trading community’s insights and contributions.

For further details, collaboration, or to purchase advanced models with superior performance, reach out to TPTBusiness@proton.me.
