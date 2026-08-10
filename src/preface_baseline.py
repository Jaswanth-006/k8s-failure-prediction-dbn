import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.autoencoder_phase3 import RobustAnomalyScorePipeline
from src.rectifier import Rectifier

class PrefaceBaseline:
    """
    Original PREFACE Baseline Implementation.
    - Detection: Binary threshold on total autoencoder reconstruction error (m_e + 3s_e).
    - Localization: Z-score ranking (returns service with max anomaly signal).
    - Decision: None (Memoryless/Instantaneous).
    """
    def __init__(self, ae_model_path: str, healthy_dataset_path: str):
        self.ae_model_path = ae_model_path

        # Load Phase 3 Autoencoder
        ckpt = torch.load(ae_model_path, map_location='cpu', weights_only=False)
        self.feature_names = ckpt['feature_names']
        self.services = ckpt['services']

        self.rectifier = Rectifier(self.services, ["cpu_usage"], ["node_cpu"])

        self.pipeline = RobustAnomalyScorePipeline(self.feature_names, self.services)
        self.pipeline.load_model(ae_model_path)

        # Calculate baseline total error m_e and s_e
        self.m_e, self.s_e = self._fit_threshold(healthy_dataset_path)
        self.threshold = self.m_e + 3 * self.s_e

    def _fit_threshold(self, dataset_path: str) -> Tuple[float, float]:
        df = pd.read_csv(dataset_path)
        # Assuming df has columns: timestamp, pod, namespace, service_name, kpi_name, value
        # We need to reshape into fixed feature vectors
        timestamps = sorted(df['timestamp'].unique())
        total_errors = []

        for ts in timestamps:
            ts_df = df[df['timestamp'] == ts]
            x_t, _ = self.rectifier.process_tick(ts_df)

            # Forward pass
            norm_x = (x_t - self.pipeline.median_train) / self.pipeline.iqr_train
            tensor_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                recon = self.pipeline.model(tensor_x).squeeze(0).numpy()

            sq_error = (norm_x - recon) ** 2
            total_error = float(np.mean(sq_error))
            total_errors.append(total_error)

        m_e = float(np.mean(total_errors))
        s_e = float(np.std(total_errors))
        if s_e == 0.0:
            s_e = 1.0

        return m_e, s_e

    def predict(self, x_t: np.ndarray) -> Tuple[bool, str, float]:
        """
        Returns:
            is_anomaly: True if total_error > threshold
            root_cause: Service with max Z-score (if anomaly) else None
            max_signal: The value of the max anomaly signal
        """
        # 1. Forward pass for total error
        norm_x = (x_t - self.pipeline.median_train) / self.pipeline.iqr_train
        tensor_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            recon = self.pipeline.model(tensor_x).squeeze(0).numpy()

        sq_error = (norm_x - recon) ** 2
        total_error = float(np.mean(sq_error))

        is_anomaly = total_error > self.threshold

        # 2. Z-score localization
        # We can just reuse compute_anomaly_signals which returns the Z-scores
        anomaly_signals = self.pipeline.compute_anomaly_signals(x_t)

        root_cause = "None"
        max_signal = 0.0

        if is_anomaly:
            # Rank by anomaly signal (Z-score)
            ranked = sorted(anomaly_signals.items(), key=lambda item: item[1], reverse=True)
            root_cause = ranked[0][0] if ranked else "None"
            max_signal = ranked[0][1] if ranked else 0.0

        return is_anomaly, root_cause, max_signal
