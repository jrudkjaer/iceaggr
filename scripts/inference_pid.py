#!/usr/bin/env python3
"""
Run a trained PID model on a dataset and write per-event class probabilities.

Example:
    python scripts/inference_pid.py --checkpoint checkpoints/<run>/best.pt \
        --output predictions/pid_predictions.parquet

Output columns: event_no, pid_<class>_pred (one per class), pred_class and, when the
dataset has labels, true_class. With labels it also prints the accuracy and saves a
confusion-matrix plot next to the output file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iceaggr.data import GeometryLoader
from iceaggr.models.pid_metrics import (
    CLASS_NAMES,
    confusion_matrix,
    metrics_from_confusion,
    plot_confusion_matrix,
)
from iceaggr.utils import get_logger
from train_pid import build_datasets, create_dataloader, create_model

logger = get_logger(__name__)


def class_names(num_classes: int) -> list:
    """Readable class names (noise, muon, neutrino for 3 classes)."""
    if num_classes == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [f"class{i}" for i in range(num_classes)]


def predictions_table(event_ids, probs, true_classes, names) -> pd.DataFrame:
    """Build the output table from numpy arrays."""
    table = {"event_no": event_ids}
    for i, name in enumerate(names):
        table[f"pid_{name}_pred"] = probs[:, i]
    table["pred_class"] = probs.argmax(axis=1)
    if true_classes is not None:
        table["true_class"] = true_classes
    return pd.DataFrame(table)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    """Run the model over the loader. Returns (event_ids, probs, true_classes or None)."""
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    all_ids, all_probs, all_true = [], [], []

    for batch in loader:
        dom_vectors = batch["dom_vectors"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(dom_vectors, padding_mask)["logits"]

        all_ids.append(batch["event_ids"])
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        if "targets" in batch:
            all_true.append(batch["targets"])

    event_ids = torch.cat(all_ids).numpy()
    probs = torch.cat(all_probs).numpy()
    true = torch.cat(all_true).numpy() if all_true else None
    return event_ids, probs, true


def main():
    parser = argparse.ArgumentParser(description="PID inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--output", default="predictions/pid_predictions.parquet")
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="Which dataset from build_datasets to run on")
    parser.add_argument("--geometry", default=None, help="Override the geometry CSV path")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]          # the settings the model was trained with
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.geometry:
        config["data"]["geometry_path"] = args.geometry
    num_classes = int(config["model"].get("num_classes", 3))
    names = class_names(num_classes)
    logger.info(f"Checkpoint {args.checkpoint} (epoch {ckpt['epoch']}, val acc {ckpt['val_accuracy']:.3f})")

    # --- model: rebuild from the saved config, then load the trained weights ---
    model = create_model(config, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- data ---
    geometry_path = config["data"]["geometry_path"]
    if config["data"].get("source", "fake") == "fake" and not Path(geometry_path).exists():
        from fake_pid_data import write_fake_geometry
        write_fake_geometry(geometry_path)
    geometry = GeometryLoader(geometry_path)

    train_ds, val_ds = build_datasets(config)
    dataset = train_ds if args.split == "train" else val_ds
    loader = create_dataloader(dataset, geometry, config, shuffle=False)
    logger.info(f"Running on the '{args.split}' dataset: {len(dataset):,} events")

    # --- predict and save ---
    use_amp = bool(config["training"].get("use_amp", False)) and device == "cuda"
    event_ids, probs, true = predict(model, loader, device, use_amp)

    table = predictions_table(event_ids, probs, true, names)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        table.to_csv(out_path, index=False)
    else:
        table.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(table):,} events to {out_path}")

    # --- with labels: accuracy and confusion matrix ---
    if true is not None:
        cm = confusion_matrix(torch.from_numpy(probs), torch.from_numpy(true), num_classes)
        m = metrics_from_confusion(cm)
        per = ", ".join(f"{n} {a:.3f}" for n, a in zip(names, m["per_class_accuracy"]))
        logger.info(f"Accuracy {m['accuracy']:.3f} | per class: {per}")
        logger.info(f"Confusion matrix (rows = true, columns = predicted):\n{cm}")
        png = out_path.with_name(out_path.stem + "_confusion_matrix.png")
        plot_confusion_matrix(cm, names, path=str(png))
        logger.info(f"Saved {png}")


if __name__ == "__main__":
    main()