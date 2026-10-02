import glob
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ---- EDIT THESE IF NEEDED ----
SOURCE_ROOT = "/groups/icecube/julie/data/niels_2026/origin_mc_muongun_130000"
PULSE_TABLE = "SplitInIcePulses"          # or "SplitInIcePulses"
OUT_DIR = "/groups/icecube/julie/my_converted_data/origin_mc_muongun_130000"
TEST_FRACTION = 0.2
SEED = 42
MAX_LABELED_EVENTS = 5000  # TEST RUN LIMIT. Set to None only once this works end-to-end.
# --------------------------------

# Map raw pid -> class index. Sign is dropped (nu vs anti-nu), flavour is kept.
#   -1 noise | 13 muon | 12 nu_e | 14 nu_mu | 16 nu_tau
PID_TO_CLASS = {-1: 0, 13: 1, 12: 2, 14: 3, 16: 4}


def pid_to_class(pid_series):
    key = pid_series.astype(int)
    key = key.where(key == -1, key.abs())  # keep -1 as noise, fold +/- for the rest
    classes = key.map(PID_TO_CLASS)
    if classes.isna().any():
        bad = sorted(key[classes.isna()].unique())
        raise ValueError(f"Unexpected pid values not in PID_TO_CLASS: {bad}")
    return classes.astype(int).values


# MC-only guard: refuse to run on anything whose path is not marked as MC.
if "mc" not in os.path.basename(SOURCE_ROOT.rstrip("/")).lower():
    raise ValueError(f"SOURCE_ROOT does not look like an MC dataset: {SOURCE_ROOT}")

os.makedirs(f"{OUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/test", exist_ok=True)

# 1. Load truth (all parquet parts) and keep only events with a real label
print("Loading truth...")
truth_files = glob.glob(f"{SOURCE_ROOT}/truth/*.parquet")
truth = pd.concat([pd.read_parquet(f) for f in truth_files], ignore_index=True)
print(f"  {len(truth):,} total rows in truth")
if "pid" not in truth.columns:
    raise KeyError(f"'pid' not in truth columns: {list(truth.columns)}")
print("  pid value counts in truth:")
print(truth["pid"].value_counts(dropna=False).to_string())

# MC-only filter: real data has no truth pid, so require a valid pid as well.
n_no_pid = truth["pid"].isna().sum()
if n_no_pid:
    print(f"  WARNING: dropping {n_no_pid:,} rows with no truth pid (not MC)")
truth = truth.dropna(subset=["pid"])

labeled = truth.dropna(subset=["azimuth", "zenith"]).reset_index(drop=True)
print(f"  {len(labeled):,} rows have a real azimuth/zenith label")

if MAX_LABELED_EVENTS is not None:
    labeled = labeled.sample(n=min(MAX_LABELED_EVENTS, len(labeled)), random_state=SEED).reset_index(drop=True)
    print(f"  TEST MODE: subsampled down to {len(labeled):,} events")

# 2. Split labeled events into train/test
rng = np.random.default_rng(SEED)
event_nos = labeled["event_no"].to_numpy().copy()  # .copy() so shuffle does not modify the DataFrame
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
    columns=["event_no", "dom_time", "charge", "dom_x", "dom_y", "dom_z"],
    filter=filt,
).to_pandas()
print(f"  {len(all_pulses):,} pulses loaded")

# 4. Build sensor_id from DOM position (this table has no string/dom_number)
print("Building sensor geometry from selected pulses...")
for c in ["dom_x", "dom_y", "dom_z"]:
    all_pulses[c] = all_pulses[c].round(2)     # guard against float noise
all_pulses["sensor_id"] = all_pulses.groupby(["dom_x", "dom_y", "dom_z"]).ngroup()

geom_out = (
    all_pulses.drop_duplicates(subset="sensor_id")[["sensor_id", "dom_x", "dom_y", "dom_z"]]
    .rename(columns={"dom_x": "x", "dom_y": "y", "dom_z": "z"})
    .sort_values("sensor_id")
)
geom_out.to_csv(f"{OUT_DIR}/sensor_geometry.csv", index=False)
print(f"  wrote sensor_geometry.csv ({len(geom_out):,} sensors found in this sample)")

# hlc is not available in this dataset -> placeholder (all treated as non-auxiliary)
all_pulses["auxiliary"] = np.float32(0.0)


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
        "pid": labeled.loc[ordered_events, "pid"].astype(int).values,
        "pid_class": pid_to_class(labeled.loc[ordered_events, "pid"]),
    })
    print("  pid_class counts:", meta["pid_class"].value_counts().sort_index().to_dict())
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