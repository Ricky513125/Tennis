import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossModalTranslate(nn.Module):
    def __init__(self, encoder_embed_dim=384):
        super(CrossModalTranslate, self).__init__()
        self.dim = encoder_embed_dim
        self.mlp_to_flow = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim)
        )
        self.mlp_to_rgb = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim)
        )
        self.mlp_to_skeleton = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim)
        )

    def forward(self, x):
        x_rgb = self.mlp_to_rgb(x)
        x_flow = self.mlp_to_flow(x)
        x_skeleton = self.mlp_to_skeleton(x)
        return x_rgb, x_flow, x_skeleton
