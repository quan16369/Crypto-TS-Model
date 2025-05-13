import os
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import glob
from lstm_attention_model import LSTMAttentionModel
from rwkv_ts_model import CryptoRWKV_TS
from lstm_model import LSTMModel
from cnn_lstm_model import CNNLSTMModel
from cnn_lstm_attention_model import LSTMCNNAttentionModel
from lstm_attention_hybrid_model import LSTMAttentionHybrid
from data_loader import CryptoDataLoader
from ms_hitt import MSHiTT
from utils import TrainingTracker, EarlyStopper
import torch.nn.functional as F
from typing import Dict, Any
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

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

class AdaptiveHuberLoss(nn.Module):
    
    def __init__(self, initial_delta=1.0):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(initial_delta))
        self.delta.requires_grad = False  # Delta không học qua backprop
        
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        loss = torch.where(
            condition,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        return loss.mean()
    
    def update_delta(self, new_delta):
        self.delta.fill_(new_delta)

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
        self.loss_fn = config_dict['training'].get('loss_fn', 'huber')
        self.huber_delta = config_dict['training'].get('huber_delta', 0.5)

def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    preds = []
    targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validating', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            preds.append(pred.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mean_squared_error(targets, preds),
        'mae': mean_absolute_error(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'smape': 100 * np.mean(2.0 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-8)),
        'r2': r2_score(targets, preds)
    }
    
    print(f"[Eval] Loss: {metrics['loss']:.4f} | MSE: {metrics['mse']:.4f} | "
          f"MAE: {metrics['mae']:.4f} | SMAPE: {metrics['smape']:.2f}% | R2: {metrics['r2']:.4f}")
    
    return metrics['loss']

def find_latest_checkpoint(checkpoint_dir):
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*/epoch_*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    
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
        elif model_type == 'ms_hitt':
            model = MSHiTT(config_dict).to(config.device)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(config_dict).to(config.device)
        elif model_type == 'cnn_lstm_attention':
            model = LSTMCNNAttentionModel(config_dict).to(config.device)
        elif model_type == 'lstm_hybridattention':
            model = LSTMAttentionHybrid(config_dict).to(config.device)
        else:
            model = LSTMModel(config_dict).to(config.device)
        print(model_type)
        
        # 5. Khởi tạo loss function
        if config.loss_fn.lower() == "huber":
            loss_fn = AdaptiveHuberLoss(initial_delta=config.huber_delta)
            logger.info(f"Using HuberLoss with initial delta={config.huber_delta}")
            
            # Tính toán delta ban đầu từ dữ liệu
            with torch.no_grad():
                sample = next(iter(train_loader))
                pred = model(sample['x'].to(config.device))
                errors = torch.abs(pred - sample['y'].to(config.device))
                delta = torch.quantile(errors, 0.8).item()
                loss_fn.update_delta(delta)
                logger.info(f"Auto-adjusted delta to: {delta:.4f}")
        else:
            loss_fn = nn.MSELoss()
            logger.info("Using MSE Loss")

        # 6. Tối ưu hóa
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=1e-4
        )

        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                        optimizer,
                                                        mode='min',       # giảm val_loss
                                                        factor=0.5,       # mỗi lần giảm LR, chia cho 2
                                                        patience=5,       # nếu 5 epoch val_loss không giảm, giảm LR
                                                        min_lr= 1e-6
                                                    )

        # 7. Resume training nếu có
        start_epoch = 0
        best_loss = float('inf')
        if config.resume == 'auto':
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        elif config.resume:
            checkpoint_path = config.resume
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resumed training from epoch {start_epoch}")

        # 8. Vòng lặp training
        for epoch in range(start_epoch, config.epochs):
            model.train()
            epoch_loss = 0
            
            # 8.1 Training phase
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{config.epochs}") as tepoch:
                for batch in tepoch:
                    x = batch['x'].to(config.device)
                    y = batch['y'].to(config.device)
                    
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    
                    # Adaptive gradient clipping
                    max_norm = 2.0 * loss_fn.delta if hasattr(loss_fn, 'delta') else 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

            # 8.2 Evaluation phase
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, config.device, loss_fn)
            scheduler.step(val_loss)

            # 8.3 Cập nhật delta cho Huber Loss
            if isinstance(loss_fn, AdaptiveHuberLoss):
                with torch.no_grad():
                    preds = model(torch.cat([b['x'] for b in train_loader], dim=0).to(config.device))
                    targets = torch.cat([b['y'] for b in train_loader], dim=0).to(config.device)
                    errors = torch.abs(preds - targets)
                    new_delta = torch.quantile(errors, 0.8).item()
                    loss_fn.update_delta(new_delta)
                    tracker.log("Loss/delta", new_delta, epoch)

            # 8.4 Logging
            tracker.log("Loss/train", avg_train_loss, epoch)
            tracker.log("Loss/val", val_loss, epoch)
            tracker.log("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)

            logger.info(f"Epoch {epoch+1}/{config.epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # 8.5 Lưu checkpoint
            if epoch % config.checkpoint_interval == 0 or val_loss < best_loss:
                if val_loss < best_loss:
                    best_loss = val_loss
                    prefix = "best_"
                else:
                    prefix = ""
                
                checkpoint_path = f"{config.checkpoint_dir}/{timestamp}/{prefix}epoch_{epoch}.pt"
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss,
                    'config': config_dict
                }, checkpoint_path)

            # 8.6 Early stopping
            if stopper.check(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 9. Visualize kết quả
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
