import os
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import glob
from rwkv_ts_model import CryptoRWKV_TS
from lstm_attention_model import LSTMAttentionModel
from lstm_model import LSTMModel
from cnn_lstm_model import CNNLSTMModel
from cnn_lstm_attention_model import LSTMWithCNNAttention
from data_loader import CryptoDataLoader
from utils import TrainingTracker, EarlyStopper
from metrics import CryptoMetrics 
import torch.nn.functional as F
from typing import Dict, Any
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình logging và suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
        self.epochs = config_dict['training']['epochs']
        self.batch_size = config_dict['training']['batch_size']
        self.lr = config_dict['training']['lr']
        self.device = torch.device(config_dict['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.data_path = config_dict['data']['path']
        self.log_dir = config_dict['training'].get('log_dir', 'logs')
        self.checkpoint_dir = config_dict['training'].get('checkpoint_dir', 'checkpoints')
        self.resume = config_dict['training'].get('resume', None)
        self.checkpoint_interval = config_dict['training'].get('checkpoint_interval', 5)

def evaluate(model, data_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            time_features = batch.get('time_features', None)
            time_features = time_features.to(device) if time_features is not None else None

            pred = model(x, time_features)

            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    r2 = r2_score(targets, preds)

    print(f"[Eval] MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")
    
    return mse  

def find_latest_checkpoint(checkpoint_dir):
    """Tìm checkpoint mới nhất trong thư mục"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*/epoch_*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Tải checkpoint và khôi phục trạng thái"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'val_loss': checkpoint.get('val_loss', float('inf'))
    }

def save_checkpoint(state, filename):
    """Lưu checkpoint"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")

def train(config_path: str = 'configs/train_config.yaml'):
    try:
        # 1. Load config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = TrainConfig(config_dict)

        # 2. Khởi tạo hệ thống
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracker = TrainingTracker(config_dict)
        stopper = EarlyStopper(config_dict)

        # 3. Chuẩn bị dữ liệu
        logger.info("Loading data...")
        data_loader = CryptoDataLoader(config_path=config_path)
        train_loader = data_loader.train_loader
        val_loader = data_loader.test_loader

        # 4. Khởi tạo model
        model_type = config_dict['model'].get('model_type', 'lstm').lower()
        if model_type == 'lstm_attention':
            model = LSTMAttentionModel(config_dict).to(config.device)
        elif model_type == 'rwkv':
            model = CryptoRWKV_TS(config_dict).to(config.device)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(config_dict).to(config.device)
        elif model_type == 'cnn_lstm_attention':
            model = LSTMWithCNNAttention(config_dict).to(config.device)
        else:
            model = LSTMModel(config_dict).to(config.device)

        optimizer = torch.optim.Adam(
                                        model.parameters(), 
                                        lr=config.lr, 
                                        weight_decay=0.001  
        )
        scheduler = scheduler = torch.optim.lr_scheduler.CyclicLR(
                                        optimizer, 
                                        base_lr=1e-4, 
                                        max_lr=1e-3,
                                        step_size_up=500
                                    )
        
        # 5. Resume training nếu có
        start_epoch = 0
        best_loss = float('inf')
        if config.resume == 'auto':
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        elif config.resume:
            checkpoint_path = config.resume
        
        if checkpoint_path:
            resume_state = load_checkpoint(model, optimizer, scheduler, checkpoint_path, config.device)
            start_epoch = resume_state['epoch'] + 1
            best_loss = resume_state['best_loss']
            logger.info(f"Resuming training from epoch {start_epoch}, checkpoint: {checkpoint_path}")

        logger.info(f"Training {model_type.upper()} model on {config.device}")
        logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

        # 6. Vòng lặp training
        train_losses = []  # List to store training losses
        val_losses = []  # List to store validation losses
        for epoch in range(start_epoch, config.epochs):
            model.train()
            epoch_loss = 0
            
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    tepoch.set_postfix(loss=loss.item())

            # 7. Đánh giá và logging
            avg_loss = epoch_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, config.device)
            scheduler.step(val_loss)

            train_losses.append(avg_loss)  # Append training loss
            val_losses.append(val_loss)  # Append validation loss

            tracker.log("Loss/train", avg_loss, epoch)
            tracker.log("Loss/val", val_loss, epoch)
            tracker.log("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)

            logger.info(f"Epoch {epoch+1}/{config.epochs} | "
                    f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # 8. Lưu checkpoint
            if epoch % config.checkpoint_interval == 0:
                checkpoint_path = f"{config.checkpoint_dir}/{timestamp}/epoch_{epoch}.pt"
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss,
                    'config': config_dict
                }, checkpoint_path)

            # 9. Lưu model tốt nhất
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = f"{config.checkpoint_dir}/{timestamp}/best_model.pt"
                save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'config': config_dict,
                    'best_loss': best_loss,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_model_path)

            # 10. Early stopping
            if stopper.check(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Plotting learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curve.png')
        plt.show()

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        tracker.close()
        logger.info("Training completed")

if __name__ == "__main__":
    train()
