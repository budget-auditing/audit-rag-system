#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
from c3rag import CCES
from losses import ContrastiveCCESLoss


class CCESDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_query_len=128, max_ev_len=512, max_ev_num=6):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_ev_len = max_ev_len
        self.max_ev_num = max_ev_num
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _extract_text(self, evidence):
        if isinstance(evidence, dict) and 'span' in evidence:
            return evidence['span']
        return str(evidence) if evidence else ""

    def _pad_evidence(self, ev_list):
        texts = [self._extract_text(e) for e in ev_list][:self.max_ev_num]
        while len(texts) < self.max_ev_num:
            texts.append("")
        return texts

    def __getitem__(self, idx):
        s = self.samples[idx]
        q_tok = self.tokenizer(s['query'], max_length=self.max_query_len, padding='max_length', truncation=True, return_tensors='pt')
        pos_texts = self._pad_evidence(s['positive_evidence'])
        pos_tok = self.tokenizer(pos_texts, max_length=self.max_ev_len, padding='max_length', truncation=True, return_tensors='pt')

        neg_sets = []
        for neg in s['negative_evidences']:
            neg_texts = self._pad_evidence(neg['evidence_set'])
            neg_tok = self.tokenizer(neg_texts, max_length=self.max_ev_len, padding='max_length', truncation=True, return_tensors='pt')
            neg_sets.append({'input_ids': neg_tok['input_ids'], 'attention_mask': neg_tok['attention_mask']})

        return {
            'query_input_ids': q_tok['input_ids'].squeeze(0),
            'query_attention_mask': q_tok['attention_mask'].squeeze(0),
            'pos_input_ids': pos_tok['input_ids'],
            'pos_attention_mask': pos_tok['attention_mask'],
            'neg_evidence_sets': neg_sets
        }


def collate_fn(batch):
    q_ids = torch.stack([x['query_input_ids'] for x in batch])
    q_mask = torch.stack([x['query_attention_mask'] for x in batch])
    pos_ids = torch.stack([x['pos_input_ids'] for x in batch])
    pos_mask = torch.stack([x['pos_attention_mask'] for x in batch])

    max_neg = max((len(x['neg_evidence_sets']) for x in batch), default=0)
    neg_sets = []
    for i in range(max_neg):
        input_batches = []
        mask_batches = []
        valid_flags = []
        for sample in batch:
            if i < len(sample['neg_evidence_sets']):
                input_batches.append(sample['neg_evidence_sets'][i]['input_ids'])
                mask_batches.append(sample['neg_evidence_sets'][i]['attention_mask'])
                valid_flags.append(True)
            else:
                input_batches.append(torch.zeros_like(sample['pos_input_ids']))
                mask_batches.append(torch.zeros_like(sample['pos_attention_mask']))
                valid_flags.append(False)
        neg_sets.append({
            'input_ids': torch.stack(input_batches),
            'attention_mask': torch.stack(mask_batches),
            'valid_mask': torch.tensor(valid_flags, dtype=torch.bool)
        })

    return {
        'query_input_ids': q_ids,
        'query_attention_mask': q_mask,
        'pos_input_ids': pos_ids,
        'pos_attention_mask': pos_mask,
        'neg_evidence_sets': neg_sets
    }


class CCESTrainer(pl.LightningModule):
    def __init__(self, encoder_model_name, lr=3e-5, warmup_ratio=0.1, temperature=0.07, freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = CCES(encoder_model_name)
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.criterion = ContrastiveCCESLoss(temperature)
        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def forward(self, q_ids, q_mask, ev_ids, ev_mask):
        return self.model(q_ids, q_mask, ev_ids, ev_mask)

    def training_step(self, batch, batch_idx):
        pos_score, _ = self(batch['query_input_ids'], batch['query_attention_mask'],
                            batch['pos_input_ids'], batch['pos_attention_mask'])
        neg_scores = []
        valid_masks = []
        for neg in batch['neg_evidence_sets']:
            neg_score, _ = self(batch['query_input_ids'], batch['query_attention_mask'],
                                neg['input_ids'], neg['attention_mask'])
            neg_scores.append(neg_score)
            valid_masks.append(neg['valid_mask'].to(neg_score.device))

        if neg_scores:
            neg_scores = torch.stack(neg_scores, dim=1)
            valid_mask = torch.stack(valid_masks, dim=1).to(neg_scores.device)
            neg_scores = neg_scores.masked_fill(~valid_mask, torch.finfo(neg_scores.dtype).min)
        else:
            neg_scores = torch.empty(pos_score.size(0), 0, device=pos_score.device)
        loss = self.criterion(pos_score, neg_scores)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("pos_score", pos_score.mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos_score, _ = self(batch['query_input_ids'], batch['query_attention_mask'],
                            batch['pos_input_ids'], batch['pos_attention_mask'])
        neg_scores = []
        valid_masks = []
        for neg in batch['neg_evidence_sets']:
            neg_score, _ = self(batch['query_input_ids'], batch['query_attention_mask'],
                                neg['input_ids'], neg['attention_mask'])
            neg_scores.append(neg_score)
            valid_masks.append(neg['valid_mask'].to(neg_score.device))

        if neg_scores:
            neg_scores = torch.stack(neg_scores, dim=1)
            valid_mask = torch.stack(valid_masks, dim=1).to(neg_scores.device)
            neg_scores = neg_scores.masked_fill(~valid_mask, torch.finfo(neg_scores.dtype).min)
        else:
            neg_scores = torch.empty(pos_score.size(0), 0, device=pos_score.device)
        loss = self.criterion(pos_score, neg_scores)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", 0)
        if total_steps:
            warmup_steps = int(total_steps * self.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


def train_cces(train_path, val_path, encoder="nghuyong/ernie-3.0-base-zh", batch_size=8, epochs=3, output="models/cces_model.pt"):
    import os
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(encoder, trust_remote_code=True)
    full_train_ds = CCESDataset(train_path, tokenizer)

    if os.path.exists(val_path):
        val_ds = CCESDataset(val_path, tokenizer)
        train_ds = full_train_ds
    else:
        if len(full_train_ds) < 2:
            raise ValueError("Insufficient training samples for validation split. Provide separate validation set or add more samples.")
        generator = torch.Generator().manual_seed(42)
        train_size = max(1, int(len(full_train_ds) * 0.9))
        val_size = len(full_train_ds) - train_size
        if val_size == 0:
            val_size = 1
            train_size = len(full_train_ds) - 1
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    model = CCESTrainer(encoder, lr=3e-5, temperature=0.07, freeze_encoder=False)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.model.state_dict(), output)
    print(f"Model saved: {output}")


if __name__ == "__main__":
    train_cces("data/train_data/cces_training_data.jsonl", "data/train_data/cces_validation_data.jsonl")
