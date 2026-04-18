"""
Experiment sweep for weight-tied ECNet.

Phase 1 — LR search at depth=4 with AdamW (batch 256, 200 epochs).
           Tries 7 learning rates spanning 1e-4 to 1e-1.

Phase 2 — Depth sweep at depths [1, 2, 4, 8, 12] using the best LR from Phase 1.

Results are saved as individual history_{run_name}.json files (one per run).
A summary table is printed at the end of each phase.

Usage:
    python sweep.py                        # full sweep (both phases)
    python sweep.py --phase 1              # LR search only
    python sweep.py --phase 2 --lr 3e-3   # depth sweep with a fixed LR
    python sweep.py --epochs 50            # quick smoke-test
    python sweep.py --dry-run              # print commands without executing
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ── Sweep configuration ───────────────────────────────────────────────────────

PHASE1_LRS    = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
PHASE2_DEPTHS = [1, 2, 4, 8, 12]
DEPTH_BASELINE = 4
DEFAULT_EPOCHS = 200
DEFAULT_BATCH  = 256
DEFAULT_OPT    = "adamw"
DEFAULT_WD     = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    """Format LR the same way train.py does for run-name construction."""
    return f"{lr:.0e}".replace('-0', '-').replace('+0', '')


def run_name(depth: int, lr: float, opt: str = DEFAULT_OPT) -> str:
    return f"cifar10_depth{depth}_dil1-1-2-1-2-3_{opt}_lr{lr_str(lr)}"


def history_path(depth: int, lr: float, opt: str = DEFAULT_OPT) -> Path:
    return Path(f"history_{run_name(depth, lr, opt)}.json")


def build_cmd(depth: int, lr: float, epochs: int, opt: str = DEFAULT_OPT,
              wd: float = DEFAULT_WD, batch: int = DEFAULT_BATCH) -> list[str]:
    return [
        sys.executable, "train.py",
        "--iterations", str(depth),
        "--lr",         str(lr),
        "--optimizer",  opt,
        "--weight-decay", str(wd),
        "--batch-size", str(batch),
        "--epochs",     str(epochs),
    ]


def load_best_acc(depth: int, lr: float, opt: str = DEFAULT_OPT) -> float | None:
    p = history_path(depth, lr, opt)
    if not p.exists():
        return None
    with open(p) as f:
        h = json.load(f)
    return max(h["test_acc"]) if h["test_acc"] else None


def load_history(depth: int, lr: float, opt: str = DEFAULT_OPT) -> dict | None:
    p = history_path(depth, lr, opt)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def run_experiment(cmd: list[str], label: str, dry_run: bool) -> bool:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    if dry_run:
        print("  [DRY RUN — skipping]")
        return True
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "DONE" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n  {status} — wall-clock: {elapsed/60:.1f} min")
    return ok


def print_table(rows: list[dict], title: str):
    """Print a simple ASCII summary table."""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    if not rows:
        print("  (no results yet)")
        return
    # Header
    keys = list(rows[0].keys())
    widths = [max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*keys))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(row[k]) for k in keys]))
    print(f"{'─'*60}")


# ── Phase 1: LR search ────────────────────────────────────────────────────────

def phase1(epochs: int, dry_run: bool) -> float | None:
    print("\n" + "█"*70)
    print("  PHASE 1 — LR search at depth=4 (AdamW, batch=256)")
    print("█"*70)

    for lr in PHASE1_LRS:
        # Skip if already done
        if not dry_run and history_path(DEPTH_BASELINE, lr).exists():
            print(f"\n  [SKIP] depth={DEPTH_BASELINE} lr={lr_str(lr)} — history file exists")
            continue
        cmd   = build_cmd(DEPTH_BASELINE, lr, epochs)
        label = f"depth={DEPTH_BASELINE}  lr={lr_str(lr)}  opt=adamw"
        ok    = run_experiment(cmd, label, dry_run)
        if not ok:
            print(f"  WARNING: run failed, continuing sweep...")

    # Summary
    rows = []
    for lr in PHASE1_LRS:
        best = load_best_acc(DEPTH_BASELINE, lr)
        h    = load_history(DEPTH_BASELINE, lr)
        avg_epoch_time = (sum(h["epoch_time"]) / len(h["epoch_time"])) if h else None
        rows.append({
            "lr":           lr_str(lr),
            "best_test_acc": f"{best:.2f}%" if best is not None else "—",
            "avg_s/epoch":  f"{avg_epoch_time:.1f}" if avg_epoch_time else "—",
        })
    print_table(rows, "Phase 1 results (depth=4, AdamW)")

    # Pick best LR
    best_lr, best_acc = None, -1.0
    for lr in PHASE1_LRS:
        acc = load_best_acc(DEPTH_BASELINE, lr)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_lr  = lr

    if best_lr is not None:
        print(f"\n  Best LR: {lr_str(best_lr)}  (test acc = {best_acc:.2f}%)")
    return best_lr


# ── Phase 2: Depth sweep ──────────────────────────────────────────────────────

def phase2(best_lr: float, epochs: int, dry_run: bool):
    print("\n" + "█"*70)
    print(f"  PHASE 2 — Depth sweep  (AdamW, lr={lr_str(best_lr)}, batch=256)")
    print(f"  Depths: {PHASE2_DEPTHS}")
    print("█"*70)

    for depth in PHASE2_DEPTHS:
        if not dry_run and history_path(depth, best_lr).exists():
            print(f"\n  [SKIP] depth={depth} lr={lr_str(best_lr)} — history file exists")
            continue
        cmd   = build_cmd(depth, best_lr, epochs)
        label = f"depth={depth}  lr={lr_str(best_lr)}  opt=adamw"
        ok    = run_experiment(cmd, label, dry_run)
        if not ok:
            print(f"  WARNING: run failed, continuing sweep...")

    # Summary
    rows = []
    for depth in PHASE2_DEPTHS:
        best = load_best_acc(depth, best_lr)
        h    = load_history(depth, best_lr)
        avg_epoch_time = (sum(h["epoch_time"]) / len(h["epoch_time"])) if h else None
        rows.append({
            "depth":         depth,
            "best_test_acc": f"{best:.2f}%" if best is not None else "—",
            "avg_s/epoch":   f"{avg_epoch_time:.1f}" if avg_epoch_time else "—",
        })
    print_table(rows, f"Phase 2 results (AdamW, lr={lr_str(best_lr)})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECTiedNet experiment sweep")
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Run only phase 1 or phase 2 (default: both)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Fixed LR for phase 2 (skips phase 1 selection)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Epochs per run (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)

    best_lr = args.lr

    if run_phase1:
        found_lr = phase1(args.epochs, args.dry_run)
        if best_lr is None:
            best_lr = found_lr

    if run_phase2:
        if best_lr is None:
            print("\nERROR: No LR specified for phase 2 and phase 1 did not complete.")
            print("       Re-run with --lr <value> or run phase 1 first.")
            sys.exit(1)
        phase2(best_lr, args.epochs, args.dry_run)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
