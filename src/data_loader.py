import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import ta
from typing import Optional, Tuple, Dict, Any
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple 

class CryptoDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 config: Dict[str, Any],
                 train: bool = True,
                 scaler: Optional[RobustScaler] = None,
                 test_mode: bool = False):
        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['pred_len']
        self.freq = config['data']['freq']
        self.train = train
        self.test_mode = test_mode
        
        raw_df = self._load_and_clean(data_path)
        self.data = self._add_crypto_features(raw_df, self.freq)
        
        self.scaler = scaler or RobustScaler()
        self._fit_scaler()

    def _load_and_clean(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df.ffill().bfill()

    def _add_crypto_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df = df.resample(freq).apply(ohlc_dict).dropna()
        
        # Thêm các indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']
        df['liquidity'] = df['volume'] * df['close']
        
        return df.dropna()

    def __getitem__(self, idx: int) -> dict:
        x = self.scaled_data[idx:idx+self.seq_len]
        y = self.scaled_data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 3]
        
        timestamps = self.data.index[idx:idx+self.seq_len]
        time_features = np.column_stack([
            timestamps.minute.values / 59.0,
            timestamps.hour.values / 23.0
        ])
        
        return {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y),
            'time_features': torch.FloatTensor(time_features),
            'actuals': torch.FloatTensor(self.data.iloc[idx:idx+self.seq_len]['close'].values)
        }

class CryptoDataLoader:
    def __init__(self, config_path: str = None, data_path: Optional[str] = None):
        config_path = config_path or str(Path(__file__).parent.parent / "configs/train_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        if data_path:
            self.config['data']['path'] = data_path
            
        full_data = CryptoDataset(
            data_path=self.config['data']['path'],
            config=self.config,
            train=True
        )
        
        split_idx = int(len(full_data) * self.config['data']['train_ratio'])
        self.train_data = torch.utils.data.Subset(full_data, range(split_idx))
        self.test_data = torch.utils.data.Subset(full_data, range(split_idx, len(full_data)))
        
        self.scaler = full_data.scaler
        self.feature_names = full_data.data.columns.tolist()
        
        self.batch_size = self.config['training']['batch_size']
        self.train_loader = self._create_loader(self.train_data, shuffle=True)
        self.test_loader = self._create_loader(self.test_data, shuffle=False)
    
    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=1,
            pin_memory=True
        )