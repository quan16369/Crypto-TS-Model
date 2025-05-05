import os
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from model import CryptoRWKV_TS
from data_loader import CryptoDataLoader
from utils import TrainingTracker, EarlyStopper
from metrics import CryptoMetrics 
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple 

class TrainConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        # Training parameters
        self.epochs = config_dict['training']['epochs']
        self.batch_size = config_dict['training']['batch_size']
        self.lr = config_dict['training']['lr']
        self.device = torch.device(config_dict['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Paths
        self.data_path = config_dict['data']['path']
        self.log_dir = config_dict['training'].get('log_dir', 'logs')
        self.checkpoint_dir = config_dict['training'].get('checkpoint_dir', 'checkpoints')

def evaluate(model, data_loader):
    """Đánh giá model trên validation set"""
    model.eval()
    total_smape = 0
    metric_calculator = CryptoMetrics()
    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(model.device)
            y = batch['y'].to(model.device)
            pred = model(x)
            total_smape += metric_calculator.smape(pred, y).item()  
    return total_smape / len(data_loader)

def train(config_path: str = 'configs/tash.config.yaml'):
    # 1. Load cấu hình từ file YAML
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainConfig(config_dict)
    
    # 2. Khởi tạo hệ thống tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = TrainingTracker(config_dict) 
    stopper = EarlyStopper(config_dict)     
    
    # 3. Load dữ liệu và model
    data_loader = CryptoDataLoader(config_path=config_path)
    train_loader = data_loader.train_loader
    val_loader = data_loader.test_loader
    
    model = CryptoRWKV_TS(config_dict).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    stopper = EarlyStopper(patience=config_dict['training'].get('patience', 10))
    
    # 4. Vòng lặp training
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x = batch['x'].to(config.device)
            y = batch['y'].to(config.device)
            time_features = batch['time_features'].to(config.device)
            
            pred = model(x, time_features)
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 5. Tính metrics và logging
        avg_loss = total_loss / len(train_loader)
        val_smape = evaluate(model, val_loader)
        
        tracker.log("Loss/train", avg_loss, epoch)
        tracker.log("Metrics/val_smape", val_smape, epoch)
        
        # 6. Lưu checkpoint định kỳ
        if epoch % config_dict['training'].get('checkpoint_interval', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config_dict
            }, f"{config.checkpoint_dir}/{timestamp}/epoch_{epoch}.pt")
        
        # 7. Lưu model tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config_dict,
                'best_loss': best_loss
            }, f"{config.checkpoint_dir}/{timestamp}/best_model.pt")
            print(f"🔥 New best model (loss={best_loss:.4f})")
        
        # 8. Early stopping
        if stopper.check(avg_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    tracker.close()
    return model

if __name__ == "__main__":
    train()