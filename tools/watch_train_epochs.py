"""Continuously emit epoch progress for an in-progress RVC training run.

Polls the main TensorBoard events file every POLL_INTERVAL seconds, parses
`learning_rate` scalar entries to compute current step, divides by
steps_per_epoch to derive epoch number. Also watches G_2333333.pth mtime
to detect checkpoint saves.

Usage:  python tools/watch_train_epochs.py logs/<exp_name>
"""
import sys, os, time, glob, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

POLL_INTERVAL = 60          # seconds between polls
EXP = sys.argv[1] if len(sys.argv) > 1 else "logs/Obito"

# Load config to compute steps_per_epoch
with open(os.path.join(EXP, "config.json")) as f:
    cfg = json.load(f)
bs = cfg["train"]["batch_size"]
with open(os.path.join(EXP, "filelist.txt")) as f:
    n_files = sum(1 for _ in f)
steps_per_epoch = (n_files + bs - 1) // bs
print(f"[init] exp={EXP} batch={bs} files={n_files} steps_per_epoch={steps_per_epoch}", flush=True)

def latest_events_file(exp):
    files = glob.glob(os.path.join(exp, "events.out.tfevents.*"))
    files = [f for f in files if f.endswith(".0")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def read_step(events_path):
    try:
        ea = EventAccumulator(events_path, size_guidance={"scalars": 0})
        ea.Reload()
        ev = ea.Scalars("learning_rate")
        return ev[-1].step if ev else 0
    except Exception:
        return 0

ckpt = os.path.join(EXP, "G_2333333.pth")
last_ckpt_mtime = os.path.getmtime(ckpt) if os.path.exists(ckpt) else 0
last_epoch_reported = -1

while True:
    ef = latest_events_file(EXP)
    if not ef:
        print("[wait] no events file yet", flush=True)
        time.sleep(POLL_INTERVAL)
        continue

    step = read_step(ef)
    epoch = step // steps_per_epoch
    if epoch > last_epoch_reported:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] epoch ~{epoch} (global_step={step})", flush=True)
        last_epoch_reported = epoch

    if os.path.exists(ckpt):
        m = os.path.getmtime(ckpt)
        if m > last_ckpt_mtime:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] CHECKPOINT SAVED at epoch ~{epoch} (file mtime updated)", flush=True)
            last_ckpt_mtime = m

    time.sleep(POLL_INTERVAL)
