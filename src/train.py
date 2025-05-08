import os
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm  # Thêm thư viện progress bar
import logging
from rwkv_ts_model import CryptoRWKV_TS
from lstm_attention_model import LSTMAttentionModel
from lstm_model import LSTMModel
from data_loader import CryptoDataLoader
from utils import TrainingTracker, EarlyStopper
from metrics import CryptoMetrics 
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple 
import warnings

# Cấu hình logging và suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt TensorFlow logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Debug CUDA tốt hơn
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

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

def evaluate(model, data_loader, device):
    """Đánh giá model trên validation set"""
    model.eval()
    total_loss = 0
    metric_calculator = CryptoMetrics()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            time_features = batch.get('time_features', None)
            time_features = time_features.to(device) if time_features is not None else None
            
            pred = model(x, time_features)
            total_loss += F.mse_loss(pred, y).item()
    
    return total_loss / len(data_loader)

def train(config_path: str = 'configs/train_config.yaml'):
    try:
        # 1. Load cấu hình từ file YAML
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        config = TrainConfig(config_dict)
        
        # 2. Khởi tạo hệ thống tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracker = TrainingTracker(config_dict) 
        stopper = EarlyStopper(config_dict)
        
        # 3. Load dữ liệu và model
        logger.info("Loading data...")
        data_loader = CryptoDataLoader(config_path=config_path)
        train_loader = data_loader.train_loader
        val_loader = data_loader.test_loader
        
        # Chọn model dựa trên config
        model_type = config_dict['model'].get('model_type', 'lstm').lower()
        if model_type == 'lstm_attention':
            model = LSTMAttentionModel(config_dict).to(config.device)
        elif model_type == 'rwkv':
            model = CryptoRWKV_TS(config_dict).to(config.device)
        else:
            model = LSTMModel(config_dict).to(config.device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        logger.info(f"Training {model_type.upper()} model on {config.device}")
        logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        
        # 4. Vòng lặp training
        best_loss = float('inf')
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            
            # Sử dụng tqdm cho progress bar
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{config.epochs}") as tepoch:
                for batch in tepoch:
                    x = batch['x'].to(config.device)
                    y = batch['y'].to(config.device)
                    time_features = batch.get('time_features', None)
                    time_features = time_features.to(config.device) if time_features is not None else None
                    
                    optimizer.zero_grad()
                    pred = model(x, time_features)
                    loss = F.mse_loss(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
            
            # Tính metrics
            avg_loss = epoch_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, config.device)
            scheduler.step(val_loss)
            
            # Logging
            tracker.log("Loss/train", avg_loss, epoch)
            tracker.log("Loss/val", val_loss, epoch)
            tracker.log("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch+1}/{config.epochs} | "
                       f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Lưu checkpoint
            if epoch % config_dict['training'].get('checkpoint_interval', 5) == 0:
                checkpoint_path = f"{config.checkpoint_dir}/{timestamp}/epoch_{epoch}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'config': config_dict
                }, checkpoint_path)
            
            # Lưu model tốt nhất
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = f"{config.checkpoint_dir}/{timestamp}/best_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config_dict,
                    'best_loss': best_loss,
                    'epoch': epoch
                }, best_model_path)
                logger.info(f"New best model saved (val_loss={best_loss:.4f})")
            
            # Early stopping
            if stopper.check(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        tracker.close()
        logger.info("Training completed")
    
    return model

if __name__ == "__main__":
    train()
