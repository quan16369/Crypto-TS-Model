import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import ta
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

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
        # Đọc dữ liệu
        df = pd.read_csv(path)
        
        # Debug: In ra các cột thực tế
        print("Các cột trong file gốc:", df.columns.tolist())
        
        # Chuẩn hóa tên cột (đổi tất cả về chữ thường)
        column_mapping = {
            'timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'  # Đảm bảo khớp với tên trong file
        }
        
        # Chỉ đổi tên các cột tồn tại
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Debug: Kiểm tra sau khi đổi tên
        print("Các cột sau chuẩn hóa:", df.columns.tolist())
        
        # Chuyển đổi timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Xử lý volume nếu cột tồn tại
        if 'volume' in df.columns:
            # Thay thế giá trị 0 bằng NaN rồi fill bằng giá trị trung bình
            df['volume'] = df['volume'].replace(0, np.nan)
            df['volume'] = df['volume'].fillna(df['volume'].rolling(12, min_periods=1).mean())
        else:
            print("Cảnh báo: Không tìm thấy cột volume")
            # Tạo cột volume mặc định nếu cần
            df['volume'] = 1.0
        
        # Đặt timestamp làm index và sắp xếp
        df = df.set_index('timestamp').sort_index()
        
        # Xử lý dữ liệu trùng lặp
        df = df[~df.index.duplicated(keep='first')]
        
        # Debug: Kiểm tra kết quả cuối cùng
        print("\n5 dòng đầu sau xử lý:")
        print(df.head())
        print("\nThống kê volume:" if 'volume' in df.columns else "\nKhông có cột volume")
        if 'volume' in df.columns:
            print(df['volume'].describe())
        
        return df.ffill().bfill()

    def _add_crypto_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        # Resample dữ liệu nếu cần (giữ nguyên khung 5 phút)
        if freq != '5T':
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = df.resample(freq).apply(ohlc_dict).dropna()
        
        # Thêm các technical indicators tối ưu cho khung 5 phút
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(12).std()  # Biến động 1 giờ (12*5phút)
        
        # RSI với tham số ngắn hơn phù hợp khung 5 phút
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=10).rsi()
        
        # MACD với tham số tối ưu
        df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        
        # Volume-based features
        df['volume_ma'] = df['volume'].rolling(12).mean()  # MA 1 giờ
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price spread và liquidity
        df['price_spread'] = (df['high'] - df['low']) / df['close']
        df['liquidity'] = df['volume'] * df['close']
        
        # Thêm features từ dữ liệu gốc của Binance nếu có
        if 'Taker buy base asset volume' in df.columns:
            df['buy_ratio'] = df['Taker buy base asset volume'] / df['volume']
        
        return df.dropna()

    def _fit_scaler(self):
        """Fit scaler trên training data"""
        if self.train:
            self.scaler.fit(self.data.values)
        self.scaled_data = self.scaler.transform(self.data.values)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> dict:
        x = self.scaled_data[idx:idx+self.seq_len]
        y = self.scaled_data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 3]  # Chỉ lấy cột close
        
        # Thêm features thời gian chi tiết
        timestamps = self.data.index[idx:idx+self.seq_len]
        time_features = np.column_stack([
            timestamps.minute.values / 59.0,
            timestamps.hour.values / 23.0,
            timestamps.dayofweek.values / 6.0,  # Ngày trong tuần
            (timestamps.hour * 60 + timestamps.minute) / 1439.0  # Phút trong ngày
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
            pin_memory=True if torch.cuda.is_available() else False
        )
