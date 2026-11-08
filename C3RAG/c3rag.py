#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from transformers import AutoModel

class CCES(torch.nn.Module):
    def __init__(self, encoder_model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(encoder_model_name, torch_dtype=torch.float32)
        self.dim = self.encoder.config.hidden_size

        self.attention = torch.nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.dim * 2, self.dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, query, query_mask, evidence, evidence_mask):
        batch_size = query.size(0)

        if evidence.dim() == 3:
            num_evidence = evidence.size(1)
            evidence_len = evidence.size(2)
            evidence_flat = evidence.view(batch_size * num_evidence, evidence_len)
            evidence_mask_flat = evidence_mask.view(batch_size * num_evidence, evidence_len)
        else:
            evidence_flat = evidence
            evidence_mask_flat = evidence_mask
            num_evidence = evidence_flat.size(0) // batch_size

        query_feat = self.encoder(input_ids=query, attention_mask=query_mask, output_hidden_states=True).hidden_states[-1]
        evidence_feat = self.encoder(input_ids=evidence_flat, attention_mask=evidence_mask_flat, output_hidden_states=True).hidden_states[-1]

        query_feat = (query_feat * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(1, keepdim=True).clamp(min=1)
        evidence_feat = (evidence_feat * evidence_mask_flat.unsqueeze(-1)).sum(1) / evidence_mask_flat.sum(1, keepdim=True).clamp(min=1)

        query_feat = F.normalize(query_feat, p=2, dim=-1)
        evidence_feat = F.normalize(evidence_feat, p=2, dim=-1)

        evidence_feat_reshaped = evidence_feat.view(batch_size, num_evidence, self.dim)
        query_expanded = query_feat.unsqueeze(1)

        h_E, attn_weights = self.attention(query_expanded, evidence_feat_reshaped, evidence_feat_reshaped)
        h_E = h_E.squeeze(1)
        attn_weights = attn_weights.squeeze(1)

        combined = torch.cat([query_feat, h_E], dim=-1)
        score = self.discriminator(combined).squeeze(-1)

        return score, attn_weights


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, pos_score, neg_scores):
        pos_logit = pos_score / self.tau
        neg_logits = neg_scores / self.tau
        logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(pos_score.size(0), dtype=torch.long, device=pos_score.device)
        return F.cross_entropy(logits, labels)
