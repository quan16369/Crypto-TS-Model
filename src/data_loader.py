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
        self.data = self._add_crypto_features(raw_df, self.freq)
        
        # Khởi tạo scalers
        self.scalers = scalers if scalers is not None else {
            'price': StandardScaler(),
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
        
        # Xử lý missing values
        df['volume'] = df['volume'].replace(0, np.nan)
        df['volume'] = df['volume'].fillna(df['volume'].rolling(12, min_periods=1).median())
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return df

    def _add_crypto_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
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
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(12).std()
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=10).rsi()
        df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(12).mean()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price features
        df['price_spread'] = (df['high'] - df['low']) / df['close']
        df['liquidity'] = df['volume'] * df['close']
        
        return df.dropna()

    def _fit_scalers(self):
        if not self.train:
            return
            
        # Nhóm các features để scale
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'volume_ma', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'volatility', 'returns', 'obv', 'price_spread']
        
        self.scalers['price'].fit(self.data[price_cols])
        self.scalers['volume'].fit(self.data[volume_cols])
        self.scalers['indicators'].fit(self.data[indicator_cols])

    def _scale_data(self):
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'volume_ma', 'liquidity']
        indicator_cols = ['rsi', 'macd', 'volatility', 'returns', 'obv', 'price_spread']
        
        # Scale từng nhóm features
        scaled_data = pd.DataFrame(index=self.data.index)
        scaled_data[price_cols] = self.scalers['price'].transform(self.data[price_cols])
        scaled_data[volume_cols] = self.scalers['volume'].transform(self.data[volume_cols])
        scaled_data[indicator_cols] = self.scalers['indicators'].transform(self.data[indicator_cols])
        
        self.scaled_data = scaled_data.values

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> dict:
        # Lấy dữ liệu theo sequence
        start_idx = idx
        end_idx = idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len
        
        x = self.scaled_data[start_idx:end_idx]
        y = self.scaled_data[end_idx:pred_end_idx, self.data.columns.get_loc('close')]
        
        return {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y)
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
        
        # Split data
        split_idx = int(len(full_data) * self.config['data']['train_ratio'])
        self.train_data = torch.utils.data.Subset(full_data, range(split_idx))
        self.test_data = torch.utils.data.Subset(full_data, range(split_idx, len(full_data)))
        
        # Lưu scalers và feature names
        self.scalers = full_data.scalers
        self.feature_names = full_data.data.columns.tolist()
        
        # Tạo data loaders
        self.batch_size = self.config['training']['batch_size']
        self.train_loader = self._create_loader(self.train_data, shuffle=True)
        self.test_loader = self._create_loader(self.test_data, shuffle=False)
    
    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
