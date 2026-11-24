#!/usr/bin/env python3
"""
Prepare the oscNext MC for the PID transformer: convert, mix, split and write training parts.

Reads   <base>/<genie|muongun|noise>/<dataset>/merged/{truth,retro,<pulse map>}/*.parquet
Writes  <out>/{train,val,test}/part-NNNNNN.parquet        pulses: time, charge, sensor_id, auxiliary
        <out>/{train,val,test}_meta/part-NNNNNN.parquet   one row per event (index into the pulses, label, weights)
        <out>/geometry.csv                                sensor_id,x,y,z (positions divided by 500)
        <out>/summary.csv                                 counts and weight bookkeeping per dataset
        <out>/settings.json

Why: every input file holds ONE dataset (one class), so batches read from one file would be single-class.
Here every event is assigned to a random group and a random subset (train/val/test), then each group is
shuffled and written out, so every output part is a random mix of all datasets.

Pass A (scatter): read each source file once, convert, route its events to groups (temporary pieces).
Pass B (gather):  per group, read the pieces, shuffle events, split by subset, sort pulses by time within
                  each event, and write parts.

Labels (5 classes): 0 noise, 1 muon, 2 nue, 3 numu, 4 nutau. 3 classes = min(label, 2).
Weights: osc_weight and class_weight are kept untouched in the meta files. To plot a subset, multiply
class_weight by  (n_events_total - n_dropped_no_pulses) / (events used)  (summary.csv, per dataset).
Events without pulses are dropped from the output but counted (they still count in the totals).
--max-events-per-file N keeps only a random N events (with pulses) per source file; n_events_total and
n_dropped_no_pulses are still counted on the whole file, so  (n_events_total - n_dropped) / n_events_used
stays the correct weight factor.

Usage:
  python prepare_pid_data.py --base /groups/icecube/julie/oscNext_datasets_new --out /path/to/prepared
  python prepare_pid_data.py --out /path/to/prepared --verify          (checks an existing output)
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KINDS = ("genie", "muongun", "noise")
SUBSETS = ("train", "val", "test")
N_SENSORS = 5160                       # 86 strings x 60 DOMs
ID_STRIDE = 10 ** 9                    # event_id = dataset * ID_STRIDE + event_no
PULSE_COLS = ["event_no", "dom_time", "charge", "string", "dom_number", "hlc"]
TRUTH_COLS = ["energy", "zenith", "azimuth", "interaction_type", "L3_oscNext_bool"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- helpers
def sensor_ids(string, dom_number):
    sid = (np.rint(string).astype(np.int64) - 1) * 60 + (np.rint(dom_number).astype(np.int64) - 1)
    if sid.size and (sid.min() < 0 or sid.max() >= N_SENSORS):
        raise ValueError(f"sensor ids outside 0..{N_SENSORS - 1}: min {sid.min()}, max {sid.max()}")
    return sid


def gather_rows(starts, lengths):
    """Row indices of the given events (start, length), concatenated in the given order."""
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.cumsum(lengths) - lengths
    return np.repeat(starts - offsets, lengths) + np.arange(total)


def make_labels(kind, pid):
    """5-class label from the dataset type; returns (label, number of events whose pid does not fit)."""
    a = np.abs(pid)
    if kind == "genie":
        label = np.full(pid.shape, -1, dtype=np.int8)
        label[a == 12], label[a == 14], label[a == 16] = 2, 3, 4
        if (label < 0).any():
            raise ValueError(f"genie dataset with unexpected pid values: {np.unique(pid[label < 0])}")
        return label, 0
    if kind == "muongun":
        return np.full(pid.shape, 1, dtype=np.int8), int((a != 13).sum())
    return np.full(pid.shape, 0, dtype=np.int8), int((pid != -1).sum())

def file_index(path):
    """The number at the end of a file name: truth_12.parquet -> 12."""
    return int(re.search(r"_(\d+)\.parquet$", os.path.basename(path)).group(1))

def discover(base, pulses, only=None):
    found = []
    for kind in KINDS:
        for merged in sorted(glob.glob(os.path.join(base, kind, "[0-9]*", "merged"))):
            dataset = os.path.basename(os.path.dirname(merged))
            name = f"{kind}/{dataset}"
            if only and name not in only:
                continue
            files = {t: sorted(glob.glob(os.path.join(merged, t, "*.parquet")), key=file_index) for t in ("truth", "retro", pulses)}
            counts = {t: len(v) for t, v in files.items()}
            if len(set(counts.values())) != 1 or 0 in counts.values():
                raise SystemExit(f"{name}: the tables need the same (non-zero) number of files, found {counts}")
            idx = [[file_index(p) for p in v] for v in files.values()]
            if any(i != idx[0] for i in idx):
                raise SystemExit(f"{name}: the file numbers differ between the tables, e.g. {[i[:5] for i in idx]}")
            found.append({"name": name, "kind": kind, "dataset": int(dataset), "files": files})
    if not found:
        raise SystemExit(f"no <type>/<dataset>/merged folders found under {base}")
    return found


# ----------------------------------------------------------------------------- geometry
def build_geometry(datasets, pulses, out_csv):
    pos = np.full((N_SENSORS, 3), np.nan)
    seen = np.zeros(N_SENSORS, dtype=bool)
    max_dev = 0.0
    for ds in datasets:
        for path in ds["files"][pulses]:
            t = pq.read_table(path, columns=["string", "dom_number", "dom_x", "dom_y", "dom_z"])
            sid = sensor_ids(t.column("string").to_numpy(), t.column("dom_number").to_numpy())
            xyz = np.column_stack([t.column(c).to_numpy() for c in ("dom_x", "dom_y", "dom_z")])
            ids, first = np.unique(sid, return_index=True)
            new = ~seen[ids]
            pos[ids[new]] = xyz[first[new]]
            seen[ids[new]] = True
            if (~new).any():
                max_dev = max(max_dev, float(np.abs(pos[ids[~new]] - xyz[first[~new]]).max()))
            if seen.all():
                break
        if seen.all():
            break
    if not seen.all():
        log(f"WARNING: {int((~seen).sum())} of {N_SENSORS} sensors never appear in the data (position 0,0,0 in geometry.csv)")
    if max_dev > 1e-2:
        log(f"WARNING: the same sensor has different positions in different files (max difference {max_dev:.3f})")
    table = np.column_stack([np.arange(N_SENSORS), np.nan_to_num(pos) / 500.0])
    np.savetxt(out_csv, table, delimiter=",", header="sensor_id,x,y,z", comments="", fmt=["%d", "%.6f", "%.6f", "%.6f"])
    log(f"geometry written: {out_csv} ({int(seen.sum())} sensors seen)")


# ----------------------------------------------------------------------------- pass A
def scatter_file(ds, k, args, n_groups, tmp, truth_cols):
    """Read source file k of a dataset, convert it, and write one piece per group."""
    tag = f"{ds['dataset']}_{k:05d}"
    stats_path = os.path.join(tmp, "stats", tag + ".json")
    if os.path.exists(stats_path):
        return False

    truth_path, retro_path, pulse_path = ds["files"]["truth"][k], ds["files"]["retro"][k], ds["files"][args.pulses][k]
    present = pq.ParquetFile(truth_path).schema_arrow.names
    cols = ["event_no", "pid"] + [c for c in truth_cols if c in present]
    truth = pq.read_table(truth_path, columns=cols)
    retro = pq.read_table(retro_path, columns=["event_no", "osc_weight", "class_weight"])
    pul = pq.read_table(pulse_path, columns=PULSE_COLS)

    ev = truth.column("event_no").to_numpy()
    n = len(ev)
    if len(np.unique(ev)) != n:
        raise ValueError(f"{tag}: event_no is not unique inside the truth file")
    if ev.min() < 0 or ev.max() >= ID_STRIDE:
        raise ValueError(f"{tag}: event_no outside 0..{ID_STRIDE - 1}")

    # weights, aligned to the truth order
    r_ev = retro.column("event_no").to_numpy()
    o = np.argsort(r_ev)
    p = np.clip(np.searchsorted(r_ev[o], ev), 0, len(r_ev) - 1)
    if len(r_ev) != n or not (r_ev[o][p] == ev).all():
        raise ValueError(f"{tag}: truth and retro do not contain the same events")
    osc_w = retro.column("osc_weight").to_numpy()[o][p]
    cls_w = retro.column("class_weight").to_numpy()[o][p]
    ratio = np.unique(np.round(osc_w / cls_w, 3)).tolist()

    pid = truth.column("pid").to_numpy()
    label, n_bad_pid = make_labels(ds["kind"], pid)

    # pulses: events must be contiguous; find the row range of every event
    pev = pul.column("event_no").to_numpy()
    if len(pev):
        change = np.flatnonzero(np.r_[True, pev[1:] != pev[:-1]])
        run_ev, run_len = pev[change], np.diff(np.r_[change, len(pev)])
        if len(np.unique(run_ev)) != len(run_ev):
            raise ValueError(f"{tag}: the pulses of an event are not contiguous")
        if not np.isin(run_ev, ev).all():
            raise ValueError(f"{tag}: pulses for events that are not in the truth table")
        ro = np.argsort(run_ev)
        rp = np.clip(np.searchsorted(run_ev[ro], ev), 0, len(run_ev) - 1)
        has = run_ev[ro][rp] == ev
        run_idx = ro[rp]
        start = np.where(has, change[run_idx], 0)
        length = np.where(has, run_len[run_idx], 0)
    else:
        has = np.zeros(n, dtype=bool)
        start = length = np.zeros(n, dtype=np.int64)

    time_ = pul.column("dom_time").to_numpy().astype(np.float32)
    charge = pul.column("charge").to_numpy().astype(np.float32)
    sensor = sensor_ids(pul.column("string").to_numpy(), pul.column("dom_number").to_numpy()).astype(np.int16)
    aux = (np.nan_to_num(pul.column("hlc").to_numpy()) == 0).astype(np.int8)   # 1 = not a hard-local-coincidence hit

    # random subset and group for EVERY event (also for the dropped ones, so the counts are complete)
    rng = np.random.default_rng([args.seed, 1, ds["dataset"], k])
    fractions = np.array(args.fractions, dtype=float) / np.sum(args.fractions)
    subset = rng.choice(len(SUBSETS), size=n, p=fractions).astype(np.int8)
    group = rng.integers(0, n_groups, size=n)

    extra = {c: truth.column(c).to_numpy().astype(np.float32) for c in truth_cols if c in present}
    event_id = ds["dataset"] * ID_STRIDE + ev

    keep = np.flatnonzero(has)
    if args.max_events_per_file and len(keep) > args.max_events_per_file:
        # random cap (own rng, so the subset/group draws above stay the same). The counts above (n_total,
        # n_dropped) were taken BEFORE this cut, which is what keeps the weights correct.
        cap_rng = np.random.default_rng([args.seed, 3, ds["dataset"], k])
        keep = np.sort(cap_rng.choice(keep, size=args.max_events_per_file, replace=False))
    kept = np.zeros(n, dtype=bool)
    kept[keep] = True
    order = keep[np.argsort(group[keep], kind="stable")]
    bounds = np.searchsorted(group[order], np.arange(n_groups + 1))
    for g in range(n_groups):
        sel = order[bounds[g]:bounds[g + 1]]
        if len(sel) == 0:
            continue
        idx = gather_rows(start[sel], length[sel])
        gdir = os.path.join(tmp, f"g{g:04d}")
        os.makedirs(gdir, exist_ok=True)
        pq.write_table(pa.table({"time": time_[idx], "charge": charge[idx], "sensor_id": sensor[idx], "auxiliary": aux[idx]}),
                       os.path.join(gdir, tag + "_pulses.parquet"), compression="zstd")
        meta = {"event_id": event_id[sel], "event_no": ev[sel], "dataset": np.full(len(sel), ds["dataset"], dtype=np.int32),
                "label": label[sel], "pid": pid[sel].astype(np.int16), "subset": subset[sel],
                "osc_weight": osc_w[sel], "class_weight": cls_w[sel], "n_pulses": length[sel].astype(np.int64)}
        meta.update({c: v[sel] for c, v in extra.items()})
        pq.write_table(pa.table(meta), os.path.join(gdir, tag + "_meta.parquet"), compression="zstd")

    stats = {
        "name": ds["name"], "dataset": ds["dataset"], "label": int(label[0]) if n else None, "n_total": int(n),
        "n_with_pulses": int(has.sum()), "n_dropped": int((~has).sum()),
        "assigned": [int((subset == s).sum()) for s in range(3)],
        "written": [int(((subset == s) & kept).sum()) for s in range(3)],
        "n_cut_max_events": int(has.sum() - kept.sum()),
        "sum_class_weight_all": float(cls_w.sum()), "sum_class_weight_dropped": float(cls_w[~has].sum()),
        "ratio_osc_over_class": ratio, "n_unexpected_pid": n_bad_pid,
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f)               # written last: it is the "done" marker of this source file
    return True


# ----------------------------------------------------------------------------- pass B
def gather_group(g, args, tmp, out, truth_cols):
    marker = os.path.join(out, ".done", f"g{g:04d}")
    if os.path.exists(marker):
        return
    gdir = os.path.join(tmp, f"g{g:04d}")
    metas = sorted(glob.glob(os.path.join(gdir, "*_meta.parquet")))
    if not metas:
        return
    meta = pa.concat_tables([pq.read_table(p) for p in metas])
    pulses = pa.concat_tables([pq.read_table(p.replace("_meta.parquet", "_pulses.parquet")) for p in metas])
    m = {c: meta.column(c).to_numpy() for c in meta.column_names}
    time_, charge = pulses.column("time").to_numpy(), pulses.column("charge").to_numpy()
    sensor, aux = pulses.column("sensor_id").to_numpy(), pulses.column("auxiliary").to_numpy()

    lengths = m["n_pulses"]
    ev_start = np.cumsum(lengths) - lengths                  # start of each event in the concatenated pulses
    if int(lengths.sum()) != len(time_):
        raise ValueError(f"group {g}: meta and pulse pieces do not fit together")

    rng = np.random.default_rng([args.seed, 2, g])
    perm = rng.permutation(len(lengths))
    for s, sname in enumerate(SUBSETS):
        ev_s = perm[m["subset"][perm] == s]
        if len(ev_s) == 0:
            continue
        n_parts = math.ceil(len(ev_s) / args.events_per_part)
        if n_parts >= 1000:
            raise ValueError(f"group {g}: more than 999 parts, use a larger --events-per-part")
        for j, chunk in enumerate(np.array_split(ev_s, n_parts)):
            batch_id = g * 1000 + j
            ln = lengths[chunk]
            idx = gather_rows(ev_start[chunk], ln)
            pos = np.repeat(np.arange(len(chunk)), ln)
            order = np.lexsort((time_[idx], pos))                 # events stay together, pulses sorted by time
            idx = idx[order]
            first = np.cumsum(ln) - ln
            name = f"part-{batch_id:06d}.parquet"
            os.makedirs(os.path.join(out, sname), exist_ok=True)
            os.makedirs(os.path.join(out, sname + "_meta"), exist_ok=True)
            pq.write_table(pa.table({"time": time_[idx], "charge": charge[idx], "sensor_id": sensor[idx], "auxiliary": aux[idx]}),
                           os.path.join(out, sname, name), compression="zstd")
            cols = {"event_id": m["event_id"][chunk], "batch_id": np.full(len(chunk), batch_id, dtype=np.int32),
                    "first_pulse_index": first, "last_pulse_index": first + ln - 1, "label": m["label"][chunk],
                    "dataset": m["dataset"][chunk], "event_no": m["event_no"][chunk], "pid": m["pid"][chunk],
                    "osc_weight": m["osc_weight"][chunk], "class_weight": m["class_weight"][chunk]}
            cols.update({c: m[c][chunk] for c in truth_cols if c in m})
            pq.write_table(pa.table(cols), os.path.join(out, sname + "_meta", name), compression="zstd")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").close()


# ----------------------------------------------------------------------------- summary
def write_summary(out, tmp):
    per = {}
    for p in sorted(glob.glob(os.path.join(tmp, "stats", "*.json"))):
        s = json.load(open(p))
        d = per.setdefault(s["name"], {"name": s["name"], "label": s["label"], "n_src_files": 0, "n_events_total": 0,
                                        "n_dropped_no_pulses": 0, "assigned": [0, 0, 0], "written": [0, 0, 0],
                                        "sum_class_weight_all": 0.0, "ratios": set(), "bad_pid": 0, "n_cut": 0})
        d["n_src_files"] += 1
        d["n_events_total"] += s["n_total"]
        d["n_dropped_no_pulses"] += s["n_dropped"]
        d["sum_class_weight_all"] += s["sum_class_weight_all"]
        d["bad_pid"] += s["n_unexpected_pid"]
        d["n_cut"] += s.get("n_cut_max_events", 0)
        d["ratios"].update(s["ratio_osc_over_class"])
        for i in range(3):
            d["assigned"][i] += s["assigned"][i]
            d["written"][i] += s["written"][i]
    header = ["dataset", "label5", "n_src_files", "n_i3_files_from_weights", "n_events_total", "n_dropped_no_pulses"] \
        + [f"assigned_{s}" for s in SUBSETS] + [f"written_{s}" for s in SUBSETS] + ["sum_class_weight_all", "n_cut_max_events"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for name, d in sorted(per.items()):
            ratios = sorted(d["ratios"])
            n_i3 = ratios[0] if len(ratios) == 1 else float("nan")
            if len(ratios) != 1:
                log(f"WARNING {name}: osc_weight/class_weight is not constant ({ratios[:5]})")
            if d["bad_pid"]:
                log(f"WARNING {name}: {d['bad_pid']} events have a pid that does not fit the folder type")
            w.writerow([name, d["label"], d["n_src_files"], n_i3, d["n_events_total"], d["n_dropped_no_pulses"]]
                       + d["assigned"] + d["written"] + [d["sum_class_weight_all"], d["n_cut"]])
    log("summary written: " + os.path.join(out, "summary.csv"))


# ----------------------------------------------------------------------------- verify
def verify(out):
    problems = []
    summ = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(out, "summary.csv")))}
    by_num = {int(k.split("/")[1]): k for k in summ}
    all_ids, rows = [], {}
    for s in SUBSETS:
        for mf in sorted(glob.glob(os.path.join(out, s + "_meta", "*.parquet"))):
            m = pq.read_table(mf, columns=["event_id", "batch_id", "first_pulse_index", "last_pulse_index", "dataset", "class_weight", "label"])
            first, last = m.column("first_pulse_index").to_numpy(), m.column("last_pulse_index").to_numpy()
            batch = int(m.column("batch_id").to_numpy()[0])
            n_rows = pq.ParquetFile(os.path.join(out, s, f"part-{batch:06d}.parquet")).metadata.num_rows
            if first[0] != 0 or (first[1:] != last[:-1] + 1).any() or last[-1] + 1 != n_rows or (last < first).any():
                problems.append(f"{s} part {batch}: pulse index does not fit the pulse file")
            all_ids.append(m.column("event_id").to_numpy())
            ds, w = m.column("dataset").to_numpy(), m.column("class_weight").to_numpy()
            for d in np.unique(ds):
                r = rows.setdefault((s, int(d)), [0, 0.0])
                r[0] += int((ds == d).sum())
                r[1] += float(w[ds == d].sum())
    ids = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        problems.append("some event_id appears more than once (inside or across subsets)")

    print(f"\n{'dataset':18s} {'subset':6s} {'written':>10s} {'expected':>10s} {'weight check':>14s}")
    for (s, d), (n, wsum) in sorted(rows.items(), key=lambda kv: (by_num[kv[0][1]], kv[0][0])):
        row = summ[by_num[d]]
        expected = int(row[f"written_{s}"])
        total_w = float(row["sum_class_weight_all"])
        n_full = int(row["n_events_total"]) - int(row["n_dropped_no_pulses"])      # events that can appear in data
        scaled = wsum * n_full / max(n, 1)                                          # same formula as in the notebook
        print(f"{by_num[d]:18s} {s:6s} {n:10,d} {expected:10,d} {scaled / total_w:14.3f}")
        if n != expected:
            problems.append(f"{by_num[d]} {s}: {n} events written, summary says {expected}")
    print("(weight check = sum of class_weight in the subset x (n_events_total - n_dropped) / written, divided by the dataset's total;"
          " it should be close to 1, a little below because of the dropped events, plus statistical scatter)")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Prepare the oscNext MC for the PID transformer")
    ap.add_argument("--base", default=None, help="Top folder with genie/, muongun/, noise/")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--pulses", default="SRTInIcePulses", help="Pulse map folder to use")
    ap.add_argument("--datasets", nargs="*", default=None, help="Only these, e.g. genie/161293 noise/888003")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-events-per-file", type=int, default=None,
                    help="Keep only a random N events (with pulses) of every source file. For quick tests / small data. "
                         "Counts and weights stay correct. For small runs also give --groups")
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="train val test")
    ap.add_argument("--events-per-part", type=int, default=200_000)
    ap.add_argument("--groups", type=int, default=None, help="Number of mixing groups (default: from the data size)")
    ap.add_argument("--rows-per-group", type=int, default=40_000_000, help="Pulse rows per group when --groups is not given")
    ap.add_argument("--truth-cols", nargs="*", default=TRUTH_COLS, help="Extra truth columns to keep in the meta files")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--verify", action="store_true", help="Only check an existing output folder")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if args.verify:
        raise SystemExit(verify(out))
    if not args.base:
        ap.error("--base is required")

    datasets = discover(os.path.abspath(args.base), args.pulses, args.datasets)
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for ds in datasets for p in ds["files"][args.pulses])
    n_groups = args.groups or max(1, math.ceil(total_rows / args.rows_per_group))
    log(f"{len(datasets)} datasets, {sum(len(d['files']['truth']) for d in datasets)} source files, "
        f"{total_rows:,} pulse rows -> {n_groups} mixing group(s)")

    os.makedirs(out, exist_ok=True)
    settings = {"datasets": [d["name"] for d in datasets], "pulses": args.pulses, "seed": args.seed,
                "fractions": args.fractions, "events_per_part": args.events_per_part, "groups": n_groups,
                "truth_cols": args.truth_cols, "max_events_per_file": args.max_events_per_file}
    sp = os.path.join(out, "settings.json")
    if os.path.exists(sp) and json.load(open(sp)) != settings:
        raise SystemExit(f"{sp} exists with different settings: use a new --out folder (or delete the old one)")
    json.dump(settings, open(sp, "w"), indent=1)

    tmp = os.path.join(out, "tmp")
    build_geometry(datasets, args.pulses, os.path.join(out, "geometry.csv"))

    for ds in datasets:
        n_files = len(ds["files"]["truth"])
        done = [scatter_file(ds, k, args, n_groups, tmp, args.truth_cols) for k in range(n_files)]
        log(f"pass A: {ds['name']}: {sum(done)} of {n_files} files converted" + ("" if all(done) else f" ({n_files - sum(done)} already done)"))
    for g in range(n_groups):
        gather_group(g, args, tmp, out, args.truth_cols)
        log(f"pass B: group {g + 1}/{n_groups} done")

    write_summary(out, tmp)
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    code = verify(out)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
