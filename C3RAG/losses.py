#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveCCESLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, pos_scores, neg_scores):
        pos_logits = pos_scores / self.tau
        neg_logits = neg_scores / self.tau
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(pos_scores.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
