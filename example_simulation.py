import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
import os
import json
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import ccxt.async_support as ccxt
import math
import glob

class PositionalEncoding(nn.Module):
    """Positional encoding layer for transformer models to incorporate sequence position information."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].expand(x.size(0), -1, -1)
        return self.dropout(x)

class TradingTransformer(nn.Module):
    """Transformer model for predicting trading signals based on market data sequences."""
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout, output_size, max_seq_length=100):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Processes input sequence through the transformer to produce trading predictions.

        Args:
            x: Tensor, shape [batch_size, seq_len, input_size]
        Returns:
            Tensor, shape [batch_size, output_size]
        """
        x = self.input_projection(x)  # Shape: [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # Shape: [batch_size, seq_len, d_model]
        out = self.transformer_encoder(x)  # Shape: [batch_size, seq_len, d_model]
        out = out[:, -1, :]  # Shape: [batch_size, d_model]
        out = self.fc(out)  # Shape: [batch_size, output_size]
        out = self.sigmoid(out)  # Shape: [batch_size, output_size]
        return out

class RewardWrapper(gym.Wrapper):
    """Custom wrapper for TradingEnv to compute enhanced rewards based on trading performance."""
    def __init__(self, env, reward_params):
        super().__init__(env)
        self.profit_weight = reward_params.get('profit_weight', 1.0)
        self.risk_weight = reward_params.get('risk_weight', 0.1)
        self.holding_time_weight = reward_params.get('holding_time_weight', 0.01)
        self.drawdown_penalty = reward_params.get('drawdown_penalty', 1.0)
        self.sl_tp_optimization_weight = reward_params.get('sl_tp_optimization_weight', 0.5)
        self.position_management_weight = reward_params.get('position_management_weight', 0.2)
        self.leverage_bonus_weight = reward_params.get('leverage_bonus_weight', 0.1)
        self.high_leverage_threshold = reward_params.get('high_leverage_threshold', 50.0)
        self.high_leverage_bonus_multiplier = reward_params.get('high_leverage_bonus_multiplier', 0.5)
        self.max_stop_loss_pct = reward_params.get('max_stop_loss_pct', 0.02)
        self.max_take_profit_pct = reward_params.get('max_take_profit_pct', 0.03)
        self.max_open_positions = reward_params.get('max_open_positions', 5)
        self.min_profit_threshold = reward_params.get('min_profit_threshold', 0.0)

    def step(self, action):
        """
        Modifies the step method to calculate a custom reward based on trade outcomes.

        Args:
            action: Action taken by the agent
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # Synchronous call
        trade = info.get('trade', {})
        reward = 0.0

        if trade.get('type') != 'none':
            profit = trade.get('profit', 0.0)
            leverage = trade.get('leverage', 1.0)
            holding_time = trade.get('holding_time', 0)
            drawdown = info.get('drawdown', 0.0)
            stop_loss_pct = trade.get('stop_loss_pct', 0.0)
            take_profit_pct = trade.get('take_profit_pct', 0.0)
            open_positions = len(self.env.positions)

            profit_reward = self.profit_weight * profit if profit > self.min_profit_threshold else 0.0
            risk_reward = -self.risk_weight * drawdown if drawdown > 0 else 0.0
            holding_penalty = -self.holding_time_weight * holding_time if holding_time > 0 else 0.0
            drawdown_penalty = -self.drawdown_penalty * drawdown if drawdown > 0 else 0.0
            sl_tp_reward = self.sl_tp_optimization_weight * (stop_loss_pct + take_profit_pct)
            position_management_reward = -self.position_management_weight * open_positions if open_positions > self.max_open_positions else 0.0
            leverage_bonus = self.leverage_bonus_weight * leverage if leverage > self.high_leverage_threshold else 0.0
            high_leverage_bonus = self.high_leverage_bonus_multiplier * leverage if leverage > self.high_leverage_threshold else 0.0

            reward = (
                profit_reward +
                risk_reward +
                holding_penalty +
                drawdown_penalty +
                sl_tp_reward +
                position_management_reward +
                leverage_bonus +
                high_leverage_bonus
            )

        info['custom_reward'] = reward
        return obs, reward, terminated, truncated, info

class TradingEnv(gym.Env):
    """Custom trading environment for reinforcement learning using market data."""
    def __init__(
        self,
        df,
        initial_balance=1000,
        max_leverage=100,
        max_stop_loss_pct=0.03,
        max_take_profit_pct=0.04,
        max_open_positions=10,
        min_profit_threshold=0.0,
        dl_model_state_dict=None,
        scaler=None,
        timeframe='1m',
        seq_length=32,
        feature_cols=None,
        dl_model_params=None,
        binance_client=None,
        symbol='BTCUSDT',
        is_live=False
    ):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_leverage = max_leverage
        self.max_stop_loss_pct = max_stop_loss_pct
        self.max_take_profit_pct = max_take_profit_pct
        self.max_open_positions = max_open_positions
        self.min_profit_threshold = min_profit_threshold
        self.timeframe = timeframe
        self.seq_length = seq_length
        self.feature_cols = feature_cols or ['open', 'high', 'low', 'close', 'volume']
        self.scaler = scaler
        self.binance_client = binance_client
        self.symbol = symbol
        self.is_live = is_live
        self.positions = []
        self.current_step = self.seq_length
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.trade_history = []
        self.sequence = deque(maxlen=seq_length)

        if dl_model_state_dict is not None and dl_model_params is not None:
            self.dl_model = TradingTransformer(**dl_model_params)
            self.dl_model.load_state_dict(dl_model_state_dict)
            self.dl_model.eval()
        else:
            self.dl_model = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.005, 0.005, 1.0], dtype=np.float32),  # action_type, amount_pct, stop_loss_pct, take_profit_pct, leverage
            high=np.array([1.0, 1.0, 0.03, 0.04, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

    def _get_sequence(self):
        """Returns the current sequence of market data, padded if necessary."""
        sequence = np.array(self.sequence)
        if len(sequence) < self.seq_length:
            pad = np.zeros((self.seq_length - len(sequence), len(self.feature_cols)))
            sequence = np.vstack((pad, sequence))
        return sequence

    def _get_augmented_observation(self):
        """Generates an observation augmented with transformer predictions."""
        sequence = self._get_sequence()
        if self.dl_model is not None:
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Shape: [1, seq_length, input_size]
            pred = self.dl_model(sequence_tensor).detach().numpy().flatten()[0]
        else:
            pred = 0.5
        current_features = sequence[-1] if len(sequence) > 0 else np.zeros(len(self.feature_cols))
        aug_obs = np.append(current_features, pred).astype(np.float32)
        return aug_obs

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = []
        self.current_step = self.seq_length
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.trade_history = []
        self.sequence.clear()
        for i in range(self.seq_length):
            data = self.df.iloc[i][self.feature_cols].values
            if self.scaler is not None:
                data = self.scaler.transform([data])[0]
            self.sequence.append(data)
        obs = self._get_augmented_observation()
        info = {
            'balance': self.balance,
            'positions': len(self.positions),
            'profit': 0.0,
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'drawdown': self.max_drawdown,
            'trade': {'type': 'none', 'entry_price': 0.0, 'amount': 0.0, 'leverage': 0.0, 'stop_loss_pct': 0.0, 'take_profit_pct': 0.0},
            'total_trades': len(self.trade_history)
        }
        return obs, info

    def step(self, action):
        """
        Executes one step in the environment, processing the agent's action.

        Args:
            action: Array containing action_type, amount_pct, stop_loss_pct, take_profit_pct, leverage
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_type, amount_pct, stop_loss_pct, take_profit_pct, leverage = action
        # Map action_type from [-1, 1] to {0, 1, 2}
        action_type = int(np.clip((action_type + 1.0) * 1.0, 0, 2))  # -1->0 (hold), 0->1 (buy), 1->2 (sell/close)
        amount_pct = np.clip(amount_pct, 0.0, 1.0)
        if action_type == 1:
            amount_pct = max(amount_pct, 0.1)  # Ensure minimum amount for buy
        leverage = np.clip(leverage, 1.0, self.max_leverage)
        stop_loss_pct = np.clip(stop_loss_pct, 0.005, self.max_stop_loss_pct)
        take_profit_pct = np.clip(take_profit_pct, 0.005, self.max_take_profit_pct)

        current_price = self.df.iloc[self.current_step]['close']
        trade = {'type': 'none', 'entry_price': 0.0, 'amount': 0.0, 'leverage': 0.0, 'stop_loss_pct': 0.0, 'take_profit_pct': 0.0, 'profit': 0.0, 'holding_time': 0}

        if action_type == 1 and len(self.positions) < self.max_open_positions:
            amount = self.balance * amount_pct / current_price
            position = {
                'entry_price': current_price,
                'amount': amount,
                'leverage': leverage,
                'stop_loss': current_price * (1 - stop_loss_pct),
                'take_profit': current_price * (1 + take_profit_pct),
                'type': 'long',
                'entry_step': self.current_step
            }
            self.positions.append(position)
            trade = {
                'type': 'long',
                'entry_price': current_price,
                'amount': amount,
                'leverage': leverage,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'profit': 0.0,
                'holding_time': 0
            }

        elif action_type == 2:
            for pos in self.positions[:]:
                price_diff = current_price - pos['entry_price']
                profit = price_diff * pos['amount'] * pos['leverage']
                fee = (pos['amount'] * current_price * 0.001) * pos['leverage']
                self.total_fees += fee
                self.balance += profit - fee
                self.total_profit += profit
                self.positions.remove(pos)
                trade = {
                    'type': 'close',
                    'entry_price': pos['entry_price'],
                    'amount': pos['amount'],
                    'leverage': pos['leverage'],
                    'stop_loss_pct': pos['stop_loss'] / pos['entry_price'] - 1,
                    'take_profit_pct': pos['take_profit'] / pos['entry_price'] - 1,
                    'profit': profit,
                    'holding_time': self.current_step - pos['entry_step']
                }
                self.trade_history.append(trade)

        for pos in self.positions[:]:
            if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                price_diff = current_price - pos['entry_price']
                profit = price_diff * pos['amount'] * pos['leverage']
                fee = (pos['amount'] * current_price * 0.001) * pos['leverage']
                self.total_fees += fee
                self.balance += profit - fee
                self.total_profit += profit
                self.positions.remove(pos)
                trade = {
                    'type': 'close',
                    'entry_price': pos['entry_price'],
                    'amount': pos['amount'],
                    'leverage': pos['leverage'],
                    'stop_loss_pct': pos['stop_loss'] / pos['entry_price'] - 1,
                    'take_profit_pct': pos['take_profit'] / pos['entry_price'] - 1,
                    'profit': profit,
                    'holding_time': self.current_step - pos['entry_step']
                }
                self.trade_history.append(trade)

        self.current_step += 1
        drawdown = max(0.0, (self.peak_balance - self.balance) / self.peak_balance)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.peak_balance = max(self.peak_balance, self.balance)

        terminated = self.balance <= 0 or self.current_step >= len(self.df) - 1
        truncated = False

        data = self.df.iloc[self.current_step][self.feature_cols].values
        if self.scaler is not None:
            data = self.scaler.transform([data])[0]
        self.sequence.append(data)
        obs = self._get_augmented_observation()
        reward = 0.0
        info = {
            'balance': self.balance,
            'positions': len(self.positions),
            'profit': trade.get('profit', 0.0),
            'total_profit': self.total_profit,
            'total_fees': self.total_fees,
            'drawdown': self.max_drawdown,
            'trade': trade,
            'total_trades': len(self.trade_history)
        }

        return obs, reward, terminated, truncated, info

def make_env(**kwargs):
    """Creates a wrapped TradingEnv instance with a RewardWrapper."""
    def _init():
        trading_env_kwargs = {k: v for k, v in kwargs.items() if k != 'reward_params'}
        env = TradingEnv(**trading_env_kwargs)
        env = RewardWrapper(env, kwargs['reward_params'])
        return env
    return _init

async def run_trading_model():
    """
    Runs a trading simulation using a PPO model and a transformer-based predictor.

    Loads saved PPO and transformer models, a scaler, and market data (klines).
    If the scaler file is missing, creates a new StandardScaler and fits it to the data.
    Executes a test run for up to 1000 steps, forcing a buy action at step 5 for testing.
    Includes debugging logs to monitor trades, rewards, and model predictions.

    Returns:
        bool: True if the simulation completes successfully, False otherwise.

    Note:
        - Expects model files, scaler, and parameters in './backtesting/results/models/'.
        - Expects kline data in './data/klines/'.
        - Creates and saves a new scaler if none exists.
        - Users must provide their own trained models and data; see README for details.
    """
    scaler = None
    try:
        # Example paths relative to project root
        ppo_model_path = "./backtesting/results/models/ppo_checkpoint.zip"
        transformer_model_path = "./backtesting/results/models/transformer_model.pth"
        parameters_path = "./backtesting/results/parameters.json"
        scaler_path = "./backtesting/results/models/scaler.pkl"
        # Dynamic kline selection
        kline_files = glob.glob("./data/klines/klines_BTCUSDT_*.parquet")
        if kline_files:
            kline_path = max(kline_files, key=os.path.getmtime)
            print(f"✅ Using latest klines: {kline_path}")
        else:
            kline_path = "./data/klines/klines_BTCUSDT_default.parquet"
            print(f"⚠️ No klines found, falling back to: {kline_path}")

        # Verify file existence (except scaler, which can be created)
        for path in [ppo_model_path, transformer_model_path, parameters_path, kline_path]:
            if not os.path.exists(path):
                print(f"❌ File not found: {path}")
                return False

        with tqdm(total=5, desc="Initialization") as pbar:
            # Load parameters
            with open(parameters_path, 'r') as f:
                parameters = json.load(f)
            feature_cols = parameters.get('feature_cols', ['open', 'high', 'low', 'close', 'volume'])
            feature_cols = feature_cols[:16] if len(feature_cols) > 16 else feature_cols
            print(f"✅ Feature columns loaded: {feature_cols}")
            pbar.update(1)

            # Load market data
            df = pd.read_parquet(kline_path)
            print(f"✅ Klines loaded: {len(df)} rows")
            pbar.update(1)

            # Load transformer model
            checkpoint = torch.load(transformer_model_path, map_location=torch.device('cpu'))
            dl_model_params = {
                'input_size': checkpoint.get('input_size', len(feature_cols)),
                'd_model': checkpoint.get('d_model', 64),
                'nhead': checkpoint.get('nhead', 8),
                'num_layers': checkpoint.get('num_layers', 2),
                'dim_feedforward': checkpoint.get('dim_feedforward', 256),
                'dropout': checkpoint.get('dropout', 0.1),
                'output_size': checkpoint.get('output_size', 1),
                'max_seq_length': 32
            }
            transformer_model = TradingTransformer(**dl_model_params)
            transformer_model.load_state_dict(checkpoint['state_dict'])
            transformer_model.eval()
            print(f"✅ Transformer model loaded from: {transformer_model_path}")
            pbar.update(1)

            # Calculate technical indicators
            from ta import add_all_ta_features
            from ta.trend import SMAIndicator, EMAIndicator, MACD
            from ta.momentum import RSIIndicator
            from ta.volatility import AverageTrueRange, BollingerBands
            from ta.volume import VolumeWeightedAveragePrice

            with tqdm(total=9, desc="Calculating technical indicators") as ind_pbar:
                df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi().fillna(50.0)
                ind_pbar.update(1)
                macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['macd'] = macd.macd().fillna(0.0)
                ind_pbar.update(1)
                df['sma'] = SMAIndicator(close=df['close'], window=20).sma_indicator().fillna(df['close'])
                ind_pbar.update(1)
                df['ema'] = EMAIndicator(close=df['close'], window=20).ema_indicator().fillna(df['close'])
                ind_pbar.update(1)
                df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range().fillna(0.0)
                ind_pbar.update(1)
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                df['bb_upper'] = bb.bollinger_hband().fillna(df['close'])
                ind_pbar.update(1)
                df['bb_lower'] = bb.bollinger_lband().fillna(df['close'])
                ind_pbar.update(1)
                df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).volume_weighted_average_price().fillna(df['close'])
                ind_pbar.update(1)
                df['volume_profile'] = df['volume'].rolling(window=20).sum().fillna(0.0)
                df['fibonacci_levels'] = (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min()).fillna(0.0)
                ind_pbar.update(1)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"✅ Technical indicators calculated")
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'sma', 'ema', 'atr', 'bb_upper', 'bb_lower', 'vwap', 'volume_profile', 'fibonacci_levels']
            print(f"✅ Updated feature columns: {feature_cols}")
            pbar.update(1)

            # Load or create scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Scaler loaded from: {scaler_path}")
            else:
                scaler = StandardScaler()
                scaler.fit(df[feature_cols])
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✅ New scaler created and saved to: {scaler_path}")
            pbar.update(1)

        # Define reward parameters
        reward_params = {
            'profit_weight': 100.0,
            'risk_weight': 0.1,
            'holding_time_weight': 0.05,
            'drawdown_penalty': 2.0,
            'sl_tp_optimization_weight': 0.5,
            'position_management_weight': 0.2,
            'leverage_bonus_weight': 0.2,
            'high_leverage_threshold': 50.0,
            'high_leverage_bonus_multiplier': 0.4,
            'max_stop_loss_pct': parameters.get('max_stop_loss_pct', 0.03),
            'max_take_profit_pct': parameters.get('max_take_profit_pct', 0.04),
            'max_open_positions': parameters.get('max_open_positions', 10),
            'min_profit_threshold': parameters.get('min_profit_threshold', 0.0)
        }
        print(f"✅ Reward parameters loaded: {reward_params}")

        # Initialize trading environment
        with tqdm(total=1, desc="Initializing environment") as env_pbar:
            env = make_env(
                df=df,
                reward_params=reward_params,
                initial_balance=parameters.get('start_capital', 3000),
                max_leverage=parameters.get('max_leverage', 100),
                max_stop_loss_pct=parameters.get('max_stop_loss_pct', 0.03),
                max_take_profit_pct=parameters.get('max_take_profit_pct', 0.04),
                max_open_positions=parameters.get('max_open_positions', 10),
                min_profit_threshold=parameters.get('min_profit_threshold', 0.0),
                dl_model_state_dict=transformer_model.state_dict(),
                scaler=scaler,
                timeframe=parameters.get('timeframe', "1m"),
                seq_length=32,
                feature_cols=feature_cols,
                dl_model_params=dl_model_params,
                binance_client=None,
                symbol=parameters.get('symbol', "BTCUSDT")
            )()
            print(f"✅ Trading environment initialized")
            env_pbar.update(1)

        # Load PPO model
        with tqdm(total=1, desc="Loading PPO model") as ppo_pbar:
            ppo_model = PPO.load(ppo_model_path, env=env)
            print(f"✅ PPO model loaded from: {ppo_model_path}")
            ppo_pbar.update(1)

        # Initialize environment and run simulation
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        max_steps = 1000
        total_trades = 0

        with tqdm(total=max_steps, desc="Test run") as test_pbar:
            while not terminated and not truncated and step_count < max_steps:
                # Get transformer prediction for debugging
                sequence = env.env._get_sequence()  # Access inner TradingEnv
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred = env.env.dl_model(sequence_tensor).detach().numpy().flatten()[0]  # Use env's dl_model
                
                # Force a buy action at step 5 to ensure trading
                if step_count == 5:
                    action = np.array([0.0, 0.5, 0.01, 0.02, 50.0], dtype=np.float32)  # action_type=0 (buy), amount_pct=0.5
                    print(f"[Step {step_count}] Forcing buy action: {action}")
                else:
                    action, _ = ppo_model.predict(obs, deterministic=False)  # Non-deterministic to encourage buys
                    print(f"[Step {step_count}] Predicted Action: {action}")

                # Debug observation and prediction
                print(f"[Step {step_count}] Transformer Pred: {pred:.4f}")
                print(f"[Step {step_count}] Sample Obs Features: {obs[:5]}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                test_pbar.update(1)

                # Log reward components
                trade = info.get('trade', {})
                if trade.get('type') != 'none':
                    profit = trade.get('profit', 0.0)
                    leverage = trade.get('leverage', 1.0)
                    holding_time = trade.get('holding_time', 0)
                    drawdown = info.get('drawdown', 0.0)
                    stop_loss_pct = trade.get('stop_loss_pct', 0.0)
                    take_profit_pct = trade.get('take_profit_pct', 0.0)
                    open_positions = len(env.env.positions)
                    profit_reward = reward_params['profit_weight'] * profit if profit > reward_params['min_profit_threshold'] else 0.0
                    risk_reward = -reward_params['risk_weight'] * drawdown if drawdown > 0 else 0.0
                    holding_penalty = -reward_params['holding_time_weight'] * holding_time if holding_time > 0 else 0.0
                    drawdown_penalty = -reward_params['drawdown_penalty'] * drawdown if drawdown > 0 else 0.0
                    sl_tp_reward = reward_params['sl_tp_optimization_weight'] * (stop_loss_pct + take_profit_pct)
                    position_management_reward = -reward_params['position_management_weight'] * open_positions if open_positions > reward_params['max_open_positions'] else 0.0
                    leverage_bonus = reward_params['leverage_bonus_weight'] * leverage if leverage > reward_params['high_leverage_threshold'] else 0.0
                    high_leverage_bonus = reward_params['high_leverage_bonus_multiplier'] * leverage if leverage > reward_params['high_leverage_threshold'] else 0.0
                    print(f"[Step {step_count}] Reward Components: Profit={profit_reward:.4f}, Risk={risk_reward:.4f}, "
                          f"Holding={holding_penalty:.4f}, Drawdown={drawdown_penalty:.4f}, SL/TP={sl_tp_reward:.4f}, "
                          f"Pos_Mgmt={position_management_reward:.4f}, Leverage={leverage_bonus:.4f}, High_Lev={high_leverage_bonus:.4f}")

                # Log trade details
                print(
                    f"[Step {step_count}] Balance: {info['balance']:.2f}, "
                    f"Profit: {info['profit']:.2f}, Positions: {info['positions']}, "
                    f"Reward: {reward:.4f}, Total Trades: {info['total_trades']}"
                )
                
                if trade.get('type') != 'none':
                    print(
                        f"[Step {step_count}] Trade: {trade['type']}, "
                        f"Entry Price: {trade['entry_price']:.2f}, "
                        f"Amount: {trade['amount']:.6f}, "
                        f"Leverage: {trade['leverage']:.2f}, "
                        f"Stop Loss Pct: {trade['stop_loss_pct']:.4f}, "
                        f"Take Profit Pct: {trade['take_profit_pct']:.4f}"
                    )
                    if info['total_trades'] > total_trades:
                        print(f"[Step {step_count}] New trade executed! Total Trades: {info['total_trades']}")
                        total_trades = info['total_trades']

        print(
            f"✅ Test run completed: Steps={step_count}, "
            f"Final Balance={info['balance']:.2f}, "
            f"Total Profit={info['total_profit']:.2f}, "
            f"Total Fees={info['total_fees']:.2f}, "
            f"Total Trades={info['total_trades']}"
        )

        env.close()
        return True

    except Exception as e:
        print(f"❌ Error running trading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if scaler is not None:
            scaler_path = "./backtesting/results/models/scaler.pkl"
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✅ Scaler saved: {scaler_path}")

if __name__ == "__main__":
    asyncio.run(run_trading_model())
