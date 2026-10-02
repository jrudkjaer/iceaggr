import sqlite3
import pandas as pd

# ---- EDIT THESE THREE LINES ----
DB_PATH = "/lustre/hpc/project/icecube/Burnsample/databases/burnsample_oscNext_data_IC86.11-22_level3_v02.00_pass2.db"
PULSE_TABLE = "SRTInIcePulses"
OUT_DIR = "/groups/icecube/julie/my_converted_data"
# ---------------------------------

import os
os.makedirs(OUT_DIR, exist_ok=True)

con = sqlite3.connect(DB_PATH)

# 1. Load labels (one row per event)
truth = pd.read_sql_query("SELECT event_no, azimuth, zenith FROM truth", con)
truth = truth.sort_values("event_no").reset_index(drop=True)

# 2. Load pulses (many rows per event)
pulses = pd.read_sql_query(
    f"SELECT event_no, dom_time, charge, string, dom_number, hlc, dom_x, dom_y, dom_z FROM {PULSE_TABLE}",
    con,
)
con.close()

pulses = pulses[pulses["event_no"].isin(truth["event_no"])]

# 3. Give each sensor a simple number (0, 1, 2, ...)
sensor_keys = list(zip(pulses["string"], pulses["dom_number"]))
unique_sensors = sorted(set(sensor_keys))
sensor_to_id = {key: i for i, key in enumerate(unique_sensors)}
pulses["sensor_id"] = [sensor_to_id[k] for k in sensor_keys]

# 4. Write sensor_geometry.csv (one row per sensor: id, x, y, z)
geom = pulses.drop_duplicates("sensor_id")[["sensor_id", "dom_x", "dom_y", "dom_z"]]
geom = geom.sort_values("sensor_id").rename(columns={"dom_x": "x", "dom_y": "y", "dom_z": "z"})
geom.to_csv(f"{OUT_DIR}/sensor_geometry.csv", index=False)

# 5. auxiliary flag (best guess: opposite of hlc)
pulses["auxiliary"] = 1.0 - pulses["hlc"]

# 6. Sort pulses to match truth's event order, then find where each event starts/ends
order = {ev: i for i, ev in enumerate(truth["event_no"])}
pulses["_order"] = pulses["event_no"].map(order)
pulses = pulses.sort_values("_order").reset_index(drop=True)

counts = pulses.groupby("event_no", sort=False).size().reindex(truth["event_no"], fill_value=0)
last_idx = counts.cumsum() - 1
first_idx = last_idx - counts + 1

# 7. Write train_meta.parquet
meta = pd.DataFrame({
    "batch_id": 1,
    "event_id": truth["event_no"],
    "first_pulse_index": first_idx.values,
    "last_pulse_index": last_idx.values,
    "azimuth": truth["azimuth"],
    "zenith": truth["zenith"],
})
meta.to_parquet(f"{OUT_DIR}/train_meta.parquet", index=False)

# 8. Write batch_1.parquet
batch = pulses[["dom_time", "charge", "sensor_id", "auxiliary"]].rename(columns={"dom_time": "time"})
batch.to_parquet(f"{OUT_DIR}/batch_1.parquet", index=False)

print("Done! Created in", OUT_DIR)
print("- train_meta.parquet")
print("- batch_1.parquet")
print("- sensor_geometry.csv")
