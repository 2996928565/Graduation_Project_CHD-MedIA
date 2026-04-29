"""
MLP 常模模型（AutoEncoder）工具模块
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn


class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须大于 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim 必须大于 0")

        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


@dataclass
class MLPNormalityResult:
    is_abnormal: bool
    score: float
    threshold: float
    abnormal_features: List[Dict]


def reconstruction_error_per_sample(x_std: torch.Tensor, x_recon_std: torch.Tensor) -> torch.Tensor:
    """
    返回每个样本的平均重建误差（MSE）
    """
    return ((x_std - x_recon_std) ** 2).mean(dim=1)


def eval_normality(
    model: MLPAutoEncoder,
    x: np.ndarray,
    feature_names: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    error_threshold: float,
    feature_residual_threshold: float = 2.0,
    topk: int = 8,
    device: str = "cpu",
) -> MLPNormalityResult:
    """
    单样本常模判别
    """
    model.eval()
    x_std = (x - mean) / (std + 1e-8)
    x_t = torch.from_numpy(x_std.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon_std = model(x_t)
        err = reconstruction_error_per_sample(x_t, x_recon_std).item()
        residual_std = (x_t - x_recon_std).squeeze(0).cpu().numpy()

    abs_res = np.abs(residual_std)
    indices = np.argsort(-abs_res)[:topk]
    abnormal_features: List[Dict] = []
    for i in indices:
        if abs_res[i] < feature_residual_threshold:
            continue
        abnormal_features.append(
            {
                "feature": feature_names[i],
                "value": float(x[i]),
                "residual_std": float(residual_std[i]),
                "abs_residual_std": float(abs_res[i]),
            }
        )

    return MLPNormalityResult(
        is_abnormal=bool(err > error_threshold),
        score=float(err),
        threshold=float(error_threshold),
        abnormal_features=abnormal_features,
    )
