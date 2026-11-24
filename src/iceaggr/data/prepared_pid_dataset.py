import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler


class PreparedPIDDataset(Dataset):
    """Events of one subset (train, val or test) written by prepare_pid_data.py."""

    def __init__(self, prepared_dir: str, subset: str, num_classes: int = 3):
        self.dir = Path(prepared_dir)
        self.subset = subset
        self.num_classes = num_classes

        meta_files = sorted(glob.glob(str(self.dir / f"{subset}_meta" / "*.parquet")))
        if not meta_files:
            raise FileNotFoundError(f"no meta files in {self.dir / (subset + '_meta')}")

        cols = ["event_id", "batch_id", "first_pulse_index", "last_pulse_index", "label"]
        meta = pa.concat_tables([pq.read_table(f, columns=cols) for f in meta_files])
        self.event_id = meta.column("event_id").to_numpy()
        self.batch_id = meta.column("batch_id").to_numpy()
        self.first = meta.column("first_pulse_index").to_numpy()
        self.last = meta.column("last_pulse_index").to_numpy()
        self.label = meta.column("label").to_numpy()
        self._cached_batch = None     # the part whose pulses are in memory right now
        self._cached = None

    def __len__(self) -> int:
        return len(self.event_id)

    def _load_part(self, batch_id: int):
        """Pulses of one part file, as numpy arrays. Only the last part is kept in memory."""
        if self._cached_batch != batch_id:
            t = pq.read_table(self.dir / self.subset / f"part-{batch_id:06d}.parquet")
            self._cached = np.stack([t.column(c).to_numpy().astype(np.float32)
                                     for c in ("time", "charge", "sensor_id", "auxiliary")], axis=1)
            self._cached_batch = batch_id
        return self._cached

    def __getitem__(self, idx: int) -> dict:
        pulses = self._load_part(int(self.batch_id[idx]))
        pulses = pulses[self.first[idx]: self.last[idx] + 1]       # the rows of this event

        label = int(self.label[idx])
        target = label if self.num_classes == 5 else min(label, 2)  # 3 classes: all neutrinos -> 2
        return {
            "pulse_features": torch.from_numpy(pulses.copy()),      # shape (n_pulses, 4)
            "event_id": torch.tensor(int(self.event_id[idx]), dtype=torch.long),
            "n_pulses": torch.tensor(len(pulses), dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }

class PartSampler(Sampler):
    """Visit the events part by part: random order of parts, random order of events inside a part."""

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        order = np.argsort(dataset.batch_id, kind="stable")             # event numbers, sorted by part
        _, starts = np.unique(dataset.batch_id[order], return_index=True)
        self.groups = np.split(order, starts[1:])                       # one array of events per part
        self.n = len(dataset)

    def set_epoch(self, epoch: int):
        self.epoch = epoch                                              # a new order every epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        groups = self.groups
        if self.shuffle:
            groups = [groups[i] for i in rng.permutation(len(groups))]  # shuffle the parts
            groups = [rng.permutation(g) for g in groups]               # shuffle the events in each part
        for g in groups:
            yield from g.tolist()