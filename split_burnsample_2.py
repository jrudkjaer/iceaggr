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
MAX_LABELED_EVENTS = 5000  # TEST RUN LIMIT. Set to None only once this works end-to-end.
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

if MAX_LABELED_EVENTS is not None:
    labeled = labeled.sample(n=min(MAX_LABELED_EVENTS, len(labeled)), random_state=SEED).reset_index(drop=True)
    print(f"  TEST MODE: subsampled down to {len(labeled):,} events")

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

# Load pulses for ALL selected events (train + test) once, up front.
# Sensor geometry is then derived directly from these pulses -- no need to
# separately scan the whole (much bigger) detector-wide pulse table, since
# your selected events will already touch the large majority of sensors.
print("Loading pulses for selected events...")
all_selected_events = list(train_events | test_events)
filt = ds.field("event_no").isin(all_selected_events)
all_pulses = pulse_dataset.to_table(
    columns=["event_no", "dom_time", "charge", "string", "dom_number", "hlc", "dom_x", "dom_y", "dom_z"],
    filter=filt,
).to_pandas()
print(f"  {len(all_pulses):,} pulses loaded")

# 4. Build sensor_id mapping directly from these pulses
print("Building sensor geometry from selected pulses...")
geom_df = all_pulses.drop_duplicates(subset=["string", "dom_number"])
unique_sensors = sorted(zip(geom_df["string"], geom_df["dom_number"]))
sensor_to_id = {key: i for i, key in enumerate(unique_sensors)}

geom_out = pd.DataFrame(
    [{"sensor_id": sensor_to_id[(s, d)], "x": x, "y": y, "z": z}
     for s, d, x, y, z in zip(geom_df["string"], geom_df["dom_number"], geom_df["dom_x"], geom_df["dom_y"], geom_df["dom_z"])]
).sort_values("sensor_id")
geom_out.to_csv(f"{OUT_DIR}/sensor_geometry.csv", index=False)
print(f"  wrote sensor_geometry.csv ({len(geom_out):,} sensors found in this sample)")

all_pulses["sensor_id"] = [
    sensor_to_id[k] for k in zip(all_pulses["string"], all_pulses["dom_number"])
]
all_pulses["auxiliary"] = 1.0 - all_pulses["hlc"].astype(np.float32)


def build_split(split_name, event_set):
    print(f"\nBuilding {split_name} split ({len(event_set):,} events)...")

    pulses = all_pulses[all_pulses["event_no"].isin(event_set)].copy()
    print(f"  {len(pulses):,} pulses for this split")

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
