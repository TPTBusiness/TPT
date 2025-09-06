# TPT: Scalping Trading Model (BTCUSDT 1m) 🚀

📚 Overview
The TPT Scalping Trading Model is a cutting-edge, high-frequency trading system tailored for the BTCUSDT pair on a 1-minute timeframe. It harnesses a hybrid architecture that blends a Transformer neural network for precise market trend forecasting with Proximal Policy Optimization (PPO) reinforcement learning for optimized trade execution. Designed for the volatile cryptocurrency market, this model thrives on capturing short-term price movements through rapid, frequent trades. A robust suite of technical indicators and a realistic trading environment make it a powerful tool for scalping enthusiasts.

💡 Note: This open-source model is shared for educational and networking purposes. Premium models, boasting win rates up to 25-30%, drawdowns below 20%, and adaptive strategies for diverse market conditions, are available for purchase. Contact TPTBusiness@proton.me for inquiries about premium models, customizations, or licensing.


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



### ⚠️ Disclaimer: These metrics are from a backtest and may vary in live trading due to market dynamics, slippage, and latency. Premium models offer win rates of 25-30%, drawdowns below 20%, and enhanced adaptability. Contact TPTBusiness@proton.me for details.


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

Higher Win Rates: Achieve 25-30% through advanced feature engineering and hyperparameter tuning.
Lower Drawdowns: Reduced to below 20% with superior risk management.
Adaptive Strategies: Dynamically adjusts to trending, ranging, or volatile markets.
Live Trading Optimization: Seamless real-time data integration and low-latency execution.
Customizations: Tailored for specific assets, timeframes, or risk profiles.

For pricing, licensing, or inquiries, contact TPTBusiness@proton.me.

**🛠️ Installation**
To set up the open-source model:

Clone the repository:git clone https://github.com/TPTBusiness/TPT.git


Install dependencies:pip install torch stable-baselines3 pandas ta numpy


Install TA-Lib for technical indicators (see TA-Lib documentation for setup).


**📖 Usage**
This repository is designed for educational and networking purposes. The core model demonstrates a powerful scalping framework, but its full implementation is proprietary. For access to the complete system, collaboration opportunities, or premium models with enhanced performance, contact TPTBusiness@proton.me.

**📜 License**
Licensed under GNU AGPL v3.0. Modifications must be open-sourced, and commercial use requires permission. Contact TPTBusiness@proton.me for commercial licensing inquiries.

**🙌 Acknowledgments**

Built with PyTorch, Stable-Baselines3, and TA-Lib.
Inspired by the trading community’s insights and contributions.

For further details, collaboration, or to purchase advanced models with superior performance, reach out to TPTBusiness@proton.me.
