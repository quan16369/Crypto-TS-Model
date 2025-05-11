import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import ta
from typing import Optional, Dict, Any
import yaml
from pathlib import Path
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split

class CryptoDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 config: Dict[str, Any],
                 train: bool = True,
                 scalers: Optional[Dict[str, TransformerMixin]] = None,
                 test_mode: bool = False):
        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['pred_len']
        self.freq = config['data']['freq']
        self.train = train
        self.test_mode = test_mode
        
        raw_df = self._load_and_clean(data_path)
        self.data = self._enhance_crypto_features(raw_df, self.freq)
        
        # Khởi tạo scalers với cải tiến mới
        self.scalers = scalers if scalers is not None else {
            'price': RobustScaler(),  # Thay StandardScaler bằng RobustScaler
            'volume': RobustScaler(),
            'indicators': MinMaxScaler(feature_range=(-1, 1))
        }
        self._fit_scalers()
        self._scale_data()

    def _load_and_clean(self, path: str) -> pd.DataFrame:

        df = pd.read_csv(path, float_precision='high')
        
        # Chuẩn hóa tên cột
        column_map = {
            'timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Xử lý timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Xử lý missing values và outliers
        df['volume'] = df['volume'].replace(0, np.nan)
        df['volume'] = df['volume'].fillna(df['volume'].rolling(12, min_periods=1).median())
        
        # Clip outliers cho các cột quan trọng
        for col in ['close', 'volume']:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q3)
            
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return df

    def _enhance_crypto_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Cải tiến feature engineering"""
        if freq != '5T':
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = df.resample(freq).apply(ohlc_dict).dropna()
        
        # Tính toán các features 
        df['log_returns'] = np.log1p(df['close'].pct_change())
        df['volatility'] = df['log_returns'].rolling(12).std()
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12).macd_diff()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume features 
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price features
        df['price_spread'] = (df['high'] - df['low']) / df['close']
        df['liquidity'] = np.log1p(df['volume'] * df['close'])  # Log transform
        
        # Momentum features 
        df['momentum_5_20'] = df['close'].rolling(5).mean() - df['close'].rolling(20).mean()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Encode cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df.dropna()

    def _fit_scalers(self):
        if not self.train:
            return
        
        # Nhóm các features để scale với logic mới
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'volume_zscore', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'volatility', 'log_returns', 'obv', 'price_spread', 'atr', 'momentum_5_20']
        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_month_start', 'is_quarter_end']
        
        # Fit scalers với data đã được xử lý outlier
        self.scalers['price'].fit(self.data[price_cols])
        self.scalers['volume'].fit(self.data[volume_cols])
        self.scalers['indicators'].fit(self.data[indicator_cols])
        self.scalers['time'] = MinMaxScaler()
        self.scalers['time'].fit(self.data[time_cols])  # Fit cho các time-based features

    def _scale_data(self):
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'volume_zscore', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'volatility', 'log_returns', 'obv', 'price_spread', 'atr', 'momentum_5_20']
        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_month_start', 'is_quarter_end']
        
        # Scale từng nhóm features
        scaled_data = pd.DataFrame(index=self.data.index)
        scaled_data[price_cols] = self.scalers['price'].transform(self.data[price_cols])
        scaled_data[volume_cols] = self.scalers['volume'].transform(self.data[volume_cols])
        scaled_data[indicator_cols] = self.scalers['indicators'].transform(self.data[indicator_cols])
        scaled_data[time_cols] = self.scalers['time'].transform(self.data[time_cols])  # Scale time features
        
        self.scaled_data = scaled_data.values
        self.feature_names = price_cols + volume_cols + indicator_cols + time_cols

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> dict:

        start_idx = idx
        end_idx = idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len 
        
        x = self.scaled_data[start_idx:end_idx]
        y = self.scaled_data[end_idx:pred_end_idx, self.feature_names.index('close')]
        
        # Thêm noise ngẫu nhiên khi train để tăng tính tổng quát
        if self.train and not self.test_mode:
            noise = np.random.normal(0, 0.01, size=x.shape)
            x = x + noise
        
        return {
            'x': torch.FloatTensor(x),  # [seq_len, num_features]
            'y': torch.FloatTensor(y).unsqueeze(-1)  # [pred_len, 1]
        }

class CryptoDataLoader:
    def __init__(self, config_path: str = None, data_path: Optional[str] = None):
        config_path = config_path or str(Path(__file__).parent.parent / "configs/train_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        if data_path:
            self.config['data']['path'] = data_path
            
        # Load full dataset
        full_data = CryptoDataset(
            data_path=self.config['data']['path'],
            config=self.config,
            train=True
        )
        
        # Chia dữ liệu theo thời gian (không dùng stratified sampling)
        split_idx = int(len(full_data) * self.config['data']['train_ratio'])
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, len(full_data))
        
        self.train_data = torch.utils.data.Subset(full_data, train_idx)
        self.test_data = torch.utils.data.Subset(full_data, test_idx)
        
        # Lưu scalers và feature names
        self.scalers = full_data.scalers
        self.feature_names = full_data.feature_names
        
        # Tạo data loaders với cải tiến
        self.batch_size = self.config['training']['batch_size']
        self.train_loader = self._create_loader(self.train_data, shuffle=True)
        self.test_loader = self._create_loader(self.test_data, shuffle=False)
    
    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,  
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            drop_last=True  # Bỏ batch cuối nếu không đủ size
        )
