"""
PREFACE-DDN Module 2 (Phase 3): Deep Autoencoder & Anomaly Signal Aggregator
Symmetric Neural Network Architecture: n -> n/2 -> n/4 -> n/8 -> n/4 -> n/2 -> n
Trained strictly on healthy operational data (unsupervised).
Phase 3 Update: Implements RobustScaler (Median/IQR) to handle non-Gaussian K8s memory drift.
Exports continuous per-service anomaly signals (a_t^s) for downstream DDN reasoning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super(DeepAutoencoder, self).__init__()
        
        # Bottleneck sizing (n -> n/2 -> n/4 -> n/8)
        h1 = max(16, input_dim // 2)
        h2 = max(8, input_dim // 4)
        bottleneck = max(4, input_dim // 8)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim) # Linear output for normalized rKPI regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class RobustAnomalyScorePipeline:
    def __init__(self, feature_names: List[str], services: List[str]):
        self.feature_names = feature_names
        self.services = services
        self.input_dim = len(feature_names)
        self.model = DeepAutoencoder(self.input_dim)
        
        # Phase 3: Robust Normalization parameters (Median and IQR)
        self.median_train = np.zeros(self.input_dim, dtype=np.float32)
        self.iqr_train = np.ones(self.input_dim, dtype=np.float32)
        
        # Per-service error normalization parameters
        # We continue to use standard Z-score for the reconstruction errors because
        # errors are approximately half-normal distributed (bound at 0).
        self.service_error_mean: Dict[str, float] = {s: 0.0 for s in services}
        self.service_error_std: Dict[str, float] = {s: 1.0 for s in services}
        
        # Map feature index to microservice name
        self.feature_service_map = []
        for feat in feature_names:
            srv = feat.split(".")[0]
            self.feature_service_map.append(srv)

    def fit_normalizer(self, healthy_data: np.ndarray):
        """Fit RobustScaler (Median/IQR) normalization on healthy baseline metrics."""
        self.median_train = np.median(healthy_data, axis=0).astype(np.float32)
        
        q75, q25 = np.percentile(healthy_data, [75, 25], axis=0)
        self.iqr_train = (q75 - q25).astype(np.float32)
        
        # Prevent division by zero
        self.iqr_train[self.iqr_train == 0.0] = 1.0

    def train_autoencoder(self, healthy_data: np.ndarray, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        """Train autoencoder on robustly standardized healthy dataset."""
        self.fit_normalizer(healthy_data)
        norm_data = (healthy_data - self.median_train) / self.iqr_train
        tensor_data = torch.tensor(norm_data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_data, tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

        # Compute healthy training reconstruction errors per service
        self.model.eval()
        with torch.no_grad():
            recon_train = self.model(tensor_data).numpy()
            sq_errors = (norm_data - recon_train) ** 2
            
            for s in self.services:
                s_indices = [i for i, srv in enumerate(self.feature_service_map) if srv == s]
                if len(s_indices) > 0:
                    s_errors = np.mean(sq_errors[:, s_indices], axis=1)
                    self.service_error_mean[s] = float(np.mean(s_errors))
                    self.service_error_std[s] = float(np.std(s_errors))
                    if self.service_error_std[s] == 0.0:
                        self.service_error_std[s] = 1.0

    def compute_anomaly_signals(self, x_t: np.ndarray) -> Dict[str, float]:
        """
        Calculates normalized continuous anomaly signal a_t^s per microservice.
        Does NOT apply hard thresholding; passes continuous signal to DDN.
        """
        self.model.eval()
        norm_x = (x_t - self.median_train) / self.iqr_train
        tensor_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            recon = self.model(tensor_x).squeeze(0).numpy()

        sq_error = (norm_x - recon) ** 2

        anomaly_signals: Dict[str, float] = {}
        for s in self.services:
            s_indices = [i for i, srv in enumerate(self.feature_service_map) if srv == s]
            if len(s_indices) > 0:
                raw_s_error = float(np.mean(sq_error[s_indices]))
                # Standardize error against healthy baseline distribution
                std_error = (raw_s_error - self.service_error_mean[s]) / self.service_error_std[s]
                anomaly_signals[s] = float(std_error)
            else:
                anomaly_signals[s] = 0.0

        return anomaly_signals

    def save_model(self, filepath: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'median_train': self.median_train,
            'iqr_train': self.iqr_train,
            'service_error_mean': self.service_error_mean,
            'service_error_std': self.service_error_std,
            'feature_names': self.feature_names,
            'services': self.services
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.median_train = checkpoint['median_train']
        self.iqr_train = checkpoint['iqr_train']
        self.service_error_mean = checkpoint['service_error_mean']
        self.service_error_std = checkpoint['service_error_std']
        self.feature_names = checkpoint['feature_names']
        self.services = checkpoint['services']
