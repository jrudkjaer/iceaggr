"""
Fake data for testing the PID pipeline before the real dataset exists.

FakePIDDataset yields random events in exactly the format the collator expects.
The class changes the charge scale and the number of pulses, so a model can
actually learn to tell the classes apart (this lets us check that training works).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

N_SENSORS = 5160                 # IceCube: 86 strings x 60 DOMs
MEAN_PULSES = (15, 60, 200)      # noise, muon, neutrino
CHARGE_SCALE = (0.3, 1.0, 3.0)   # noise, muon, neutrino


def write_fake_geometry(path: str, seed: int = 0) -> None:
    """Write a CSV with columns sensor_id,x,y,z (positions random in [-1, 1])."""
    rng = np.random.default_rng(seed)
    rows = np.column_stack([np.arange(N_SENSORS), rng.uniform(-1, 1, (N_SENSORS, 3))])
    np.savetxt(path, rows, delimiter=",", header="sensor_id,x,y,z",
               comments="", fmt=["%d", "%.5f", "%.5f", "%.5f"])


def _make_pulses(rng, cls: int) -> np.ndarray:
    """One fake event: array of shape (n_pulses, 4) = [time, charge, sensor_id, auxiliary]."""
    n_pulses = int(rng.poisson(MEAN_PULSES[cls])) + 5
    time = np.sort(rng.uniform(9000, 15000, n_pulses))
    charge = rng.lognormal(np.log(CHARGE_SCALE[cls]), 0.4, n_pulses)
    sensor = rng.integers(0, N_SENSORS, n_pulses)
    aux = rng.integers(0, 2, n_pulses)
    return np.stack([time, charge, sensor, aux], axis=1).astype(np.float32)


class FakePIDDataset(Dataset):
    """Random events with the same dictionary format as IceCubeDataset."""

    def __init__(self, n_events: int = 2000, seed: int = 0):
        self.n_events = n_events
        self.seed = seed
        self.labels = np.random.default_rng(seed).integers(0, 3, n_events)

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        cls = int(self.labels[idx])
        rng = np.random.default_rng([self.seed, idx])    # same event for the same idx
        pulses = _make_pulses(rng, cls)
        return {
            "pulse_features": torch.from_numpy(pulses),
            "event_id": torch.tensor(idx, dtype=torch.long),
            "n_pulses": torch.tensor(len(pulses), dtype=torch.long),
            "target": torch.tensor(cls, dtype=torch.long),    # 0 noise, 1 muon, 2 neutrino
        }