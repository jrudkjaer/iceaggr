#!/usr/bin/env python3
"""
Training script for the PID transformer (noise / muon / neutrino).

Built from the pieces of train_flat.py, but with a classification loss
(cross-entropy) instead of the angular loss. There are no angles in this file.
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from iceaggr.data import make_collate_flat
from iceaggr.models import FlatTransformerV2
from iceaggr.utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: str) -> nn.Module:
    """Build FlatTransformerV2 with the PID head from config['model']."""
    m = config['model']
    model_config = {
        "max_pulses_per_dom": m['max_pulses_per_dom'],
        "d_model": m['d_model'],
        "max_doms": m['max_doms'],
        "num_heads": m['num_heads'],
        "num_layers": m['num_layers'],
        "hidden_dim": m['hidden_dim'],
        "head_hidden_dim": m['head_hidden_dim'],
        "dropout": m['dropout'],
        "input_mode": m.get('input_mode', 'none'),
        "head_type": "pid",
        "num_classes": m.get('num_classes', 3),
    }
    model = FlatTransformerV2(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PID model: {n_params:,} parameters, {model_config['num_classes']} classes")
    return model


def build_optimizer_and_scheduler(model, config, total_steps, pct_start):
    """AdamW + OneCycleLR (same settings as train_flat.py's default path)."""
    lr = float(config['training']['lr'])
    weight_decay = float(config['training']['weight_decay'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,          # initial LR = max_lr / 25
        final_div_factor=1000.0,  # final LR = max_lr / 1000
    )
    return optimizer, scheduler

def _worker_init_fn(worker_id: int) -> None:
    """One torch thread per DataLoader worker (avoids thread oversubscription)."""
    torch.set_num_threads(1)


def create_dataloader(dataset, geometry, config: dict, shuffle: bool) -> DataLoader:
    """Build a DataLoader around ANY dataset that yields the standard event dict.

    The dataset can be the fake one, the real parquet one, or one reading a
    database: the model side only sees what the collator produces.
    """
    collate_fn = make_collate_flat(
        geometry,
        max_pulses_per_dom=config['model']['max_pulses_per_dom'],
        max_doms=config['model']['max_doms'],
    )
    num_workers = int(config['data'].get('num_workers', 0))
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

def train_epoch(model, loader, optimizer, scheduler, device, epoch, config):
    """Train for one epoch. Returns (mean training loss, training accuracy)."""
    model.train()

    clip = float(config['training'].get('gradient_clip', 1.0))
    log_every = int(config['training'].get('log_every', 100))
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    use_amp = bool(config['training'].get('use_amp', False)) and device_type == 'cuda'

    n_batches = len(loader)
    n_trained = 0
    n_events = 0
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        dom_vectors = batch['dom_vectors'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)   # (B,) integers 0..num_classes-1

        # 1. predict, 2. score
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(dom_vectors, padding_mask)['logits']     # (B, num_classes)
        loss = F.cross_entropy(logits.float(), targets)

        # 3. find the blame (gradient for every weight)
        loss.backward()
        # 4. limit a too-large step
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # 5. update the weights, advance the learning-rate schedule, clear the gradients
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # bookkeeping
        loss_sum += loss.detach()
        correct += (logits.argmax(dim=1) == targets).sum()
        n_trained += 1
        n_events += targets.numel()

        if n_trained % log_every == 0 or n_trained == n_batches:
            logger.info(
                f"Epoch {epoch:3d} | batch {n_trained}/{n_batches} | "
                f"loss {loss_sum.item() / n_trained:.4f} | "
                f"acc {correct.item() / n_events:.3f} | "
                f"{time.time() - start_time:.0f}s"
            )

    return loss_sum.item() / max(n_trained, 1), correct.item() / max(n_events, 1)