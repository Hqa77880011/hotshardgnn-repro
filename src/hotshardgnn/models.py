"""Compact TGAT-style temporal attention and temporal-window GraphSAGE.

These are independent PyTorch implementations, not the unpublished DGL/TGL code.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.frequency = nn.Parameter(1 / 10 ** torch.linspace(0, 9, dim))
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, delta):
        return torch.cos(delta.unsqueeze(-1) * self.frequency + self.phase)


class GraphModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden, layers, heads=2,
                 kind="tgat", classes=2):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(input_dim, hidden)
        if kind == "tgat":
            self.time = TimeEncoding(hidden)
            self.edge = nn.Linear(max(1, edge_dim), hidden, bias=False)
            self.attention = nn.ModuleList([
                nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
                for _ in range(layers)])
            self.merge = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
            self.predict = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.merge = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
            self.predict = nn.Linear(hidden, classes)

    def encode(self, tree, features, edge_features, depth):
        if tree.children is None:
            return self.project(features)
        child = self.encode(tree.children, features, edge_features, depth - 1)
        batch, k = tree.valid.shape
        child = child.reshape(batch, k + 1, -1)
        own, neighbor = child[:, 0], child[:, 1:]
        device = child.device
        valid = torch.as_tensor(tree.valid, device=device)
        if self.kind == "tgat":
            raw = np.asarray(edge_features[tree.edge_ids]).copy()
            if raw.shape[-1] == 0:
                raw = np.zeros((*raw.shape[:2], 1), dtype=np.float32)
            edge = torch.as_tensor(raw, device=device)
            delta = torch.as_tensor(tree.deltas, device=device, dtype=torch.float32)
            tokens = neighbor + self.edge(edge) + self.time(delta)
            # Self token makes attention defined even for an empty history.
            tokens = torch.cat([own[:, None], tokens], dim=1)
            mask = torch.cat([torch.zeros((batch, 1), dtype=torch.bool, device=device), ~valid], dim=1)
            result, _ = self.attention[depth - 1](own[:, None], tokens, tokens,
                                                  key_padding_mask=mask, need_weights=False)
            aggregate = result[:, 0]
        else:
            aggregate = (neighbor * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        return F.relu(self.merge[depth - 1](torch.cat([own, aggregate], dim=-1)))

    def forward(self, tree, features, edge_features, layers):
        embedding = self.encode(tree, features, edge_features, layers)
        if self.kind == "sage":
            return self.predict(embedding)
        source, positive, negative = embedding.chunk(3)
        return torch.cat([self.predict(torch.cat([source, positive], dim=-1)).flatten(),
                          self.predict(torch.cat([source, negative], dim=-1)).flatten()])
