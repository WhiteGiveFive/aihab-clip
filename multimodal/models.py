from __future__ import annotations

import torch
from torch import nn


class IdentityEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TabularProjectionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatFusion(nn.Module):
    def forward(self, modalities):
        return torch.cat([torch.flatten(modality, start_dim=1) for modality in modalities], dim=1)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionClassifier(nn.Module):
    def __init__(
        self,
        mode: str,
        image_encoder: nn.Module,
        tabular_encoder: nn.Module,
        head: nn.Module,
        fusion: nn.Module | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.head = head
        self.fusion = fusion

    def forward(self, image_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "image_only":
            return self.head(self.image_encoder(image_features))
        if self.mode in {"geo_only", "soil_only", "tabular_only"}:
            return self.head(self.tabular_encoder(tabular_features))
        if self.mode in {"raw_concat", "soil_raw_concat", "tabular_raw_concat", "soil_projected_concat", "tabular_projected_concat"}:
            if self.fusion is None:
                raise RuntimeError(f"{self.mode} mode requires a fusion module")
            fused = self.fusion([self.image_encoder(image_features), self.tabular_encoder(tabular_features)])
            return self.head(fused)
        raise ValueError(f"Unsupported fusion mode: {self.mode}")
