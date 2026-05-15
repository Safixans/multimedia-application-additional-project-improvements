import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import SpatialTransformer
from .config import ModelConfig


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super(Denoiser, self).__init__()

        self.backbone = SpatialTransformer(
            ModelConfig(
                n_genes=config.n_genes,
                d_input=config.feature_dim, 
                d_model=config.hidden_dim, 
                d_edge_model=config.pairwise_hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                n_neighbors=config.n_neighbors,
                act=config.activation,
            )
        )
        self.loss_func = nn.MSELoss()

        self.fourier_proj = TimestepEmbedder(config.hidden_dim)
        self.image_transform = nn.Linear(config.feature_dim, config.hidden_dim)

    def inference(self, noisy_exp, img_features, coords, t_steps, predict=False, cached_graph=None):
        # noisy_exp: [B, n_cells, n_genes]
        # img_features: [B, n_cells, n_features]
        # coords: [B, n_cells, 2]
        # t_steps: [B]

        img_features = self.image_transform(img_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(noisy_exp.shape[0], noisy_exp.shape[1], -1)
        features = img_features + time_emb

        prediction = self.backbone(
            gene_exp=noisy_exp,
            features=features,
            coords=coords,
            cached_graph=cached_graph,
        )
        return prediction

    def forward(self, exp, img_features, coords, labels, t_steps):
        prediction = self.inference(exp, img_features, coords, t_steps)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        return prediction, loss

    @torch.no_grad()
    def build_inference_cache(self, img_features, coords):
        return self.backbone.build_inference_cache(img_features, coords)