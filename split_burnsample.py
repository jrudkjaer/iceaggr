import glob
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ---- EDIT THESE IF NEEDED ----
SOURCE_ROOT = "/lustre/hpc/icecube/janikh/niels_2026/ic86_2011_2022_burnsample"
PULSE_TABLE = "SRTInIcePulses"          # or "SplitInIcePulses"
OUT_DIR = "/groups/icecube/julie/my_converted_data"
TEST_FRACTION = 0.2
SEED = 42
# --------------------------------

os.makedirs(f"{OUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/test", exist_ok=True)

# 1. Load truth (all parquet parts) and keep only events with a real label
print("Loading truth...")
truth_files = glob.glob(f"{SOURCE_ROOT}/truth/*.parquet")
truth = pd.concat([pd.read_parquet(f) for f in truth_files], ignore_index=True)
print(f"  {len(truth):,} total rows in truth")

labeled = truth.dropna(subset=["azimuth", "zenith"]).reset_index(drop=True)
print(f"  {len(labeled):,} rows have a real azimuth/zenith label")

# 2. Split labeled events into train/test
rng = np.random.default_rng(SEED)
event_nos = labeled["event_no"].to_numpy()
rng.shuffle(event_nos)
n_test = int(len(event_nos) * TEST_FRACTION)
test_events = set(event_nos[:n_test])
train_events = set(event_nos[n_test:])
print(f"  train: {len(train_events):,} events | test: {len(test_events):,} events")

labeled = labeled.set_index("event_no")

# 3. Load pulse table as a dataset (reads only what's needed, not everything at once)
print(f"Opening pulse table {PULSE_TABLE}...")
pulse_dataset = ds.dataset(f"{SOURCE_ROOT}/{PULSE_TABLE}", format="parquet")

# 4. Build a single consistent sensor_id mapping across BOTH splits
print("Building sensor geometry (this reads string/dom_number/position columns only)...")
geom_cols = pulse_dataset.to_table(columns=["string", "dom_number", "dom_x", "dom_y", "dom_z"])
geom_df = geom_cols.to_pandas().drop_duplicates(subset=["string", "dom_number"])
sensor_keys = list(zip(geom_df["string"], geom_df["dom_number"]))
unique_sensors = sorted(set(sensor_keys))
sensor_to_id = {key: i for i, key in enumerate(unique_sensors)}

geom_df["sensor_id"] = [sensor_to_id[k] for k in zip(geom_df["string"], geom_df["dom_number"])]
geom_out = (
    geom_df[["sensor_id", "dom_x", "dom_y", "dom_z"]]
    .drop_duplicates(subset=["sensor_id"])
    .sort_values("sensor_id")
    .rename(columns={"dom_x": "x", "dom_y": "y", "dom_z": "z"})
)
geom_out.to_csv(f"{OUT_DIR}/sensor_geometry.csv", index=False)
print(f"  wrote sensor_geometry.csv ({len(geom_out):,} sensors)")


def build_split(split_name, event_set):
    print(f"\nBuilding {split_name} split ({len(event_set):,} events)...")

    # Pull only the pulses belonging to this split's events
    filt = ds.field("event_no").isin(list(event_set))
    table = pulse_dataset.to_table(
        columns=["event_no", "dom_time", "charge", "string", "dom_number", "hlc"],
        filter=filt,
    )
    pulses = table.to_pandas()
    print(f"  {len(pulses):,} pulses loaded")

    pulses["sensor_id"] = [
        sensor_to_id[k] for k in zip(pulses["string"], pulses["dom_number"])
    ]
    pulses["auxiliary"] = 1.0 - pulses["hlc"].astype(np.float32)

    # Order pulses to match the event order we'll write into meta
    ordered_events = [e for e in event_nos if e in event_set]
    order = {ev: i for i, ev in enumerate(ordered_events)}
    pulses["_order"] = pulses["event_no"].map(order)
    pulses = pulses.sort_values("_order").reset_index(drop=True)

    counts = pulses.groupby("event_no", sort=False).size().reindex(ordered_events, fill_value=0)
    last_idx = counts.cumsum() - 1
    first_idx = last_idx - counts + 1
    first_idx = first_idx.where(counts > 0, -1)
    last_idx = last_idx.where(counts > 0, -1)

    n_zero = (counts == 0).sum()
    if n_zero:
        print(f"  WARNING: {n_zero:,} events have zero pulses (index set to -1)")

    meta = pd.DataFrame({
        "batch_id": 1,
        "event_id": ordered_events,
        "first_pulse_index": first_idx.values,
        "last_pulse_index": last_idx.values,
        "azimuth": labeled.loc[ordered_events, "azimuth"].values,
        "zenith": labeled.loc[ordered_events, "zenith"].values,
    })
    meta.to_parquet(f"{OUT_DIR}/{split_name}_meta.parquet", index=False)
    print(f"  wrote {split_name}_meta.parquet")

    batch = pulses[["dom_time", "charge", "sensor_id", "auxiliary"]].rename(columns={"dom_time": "time"})
    batch.to_parquet(f"{OUT_DIR}/{split_name}/batch_1.parquet", index=False)
    print(f"  wrote {split_name}/batch_1.parquet")


build_split("train", train_events)
build_split("test", test_events)

print("\nDone. In data_config.yaml, set:")
print(f"  root: {OUT_DIR}")
print(f"  train: {OUT_DIR}")
print(f"  test: {OUT_DIR}")
print('  batch_pattern: "batch_*.parquet"')
