"""
Experiment sweep for ECTiedNet on ImageNet.

Phase 1 — LR search at the default architecture (stage_iterations=4 4 4).
           Tests 5 learning rates with AdamW.

Phase 2 — Iteration-depth sweep at the best LR from Phase 1.
           Varies total processing depth by changing stage_iterations uniformly.
           Configs: (2,2,2) (3,3,3) (4,4,4) (6,6,6) → 6/9/12/18 total iterations.

Results are saved as individual history_{run_name}.json files (one per run).
A summary table is printed at the end of each phase.

Usage:
    python sweep.py                          # full sweep (both phases)
    python sweep.py --phase 1                # LR search only
    python sweep.py --phase 2 --lr 1e-3     # iteration sweep with a fixed LR
    python sweep.py --dataset imagenet-mini-50 --epochs 50   # quick smoke-test
    python sweep.py --dry-run                # print commands without executing
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ── Sweep configuration ───────────────────────────────────────────────────────

PHASE1_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
PHASE1_ITERS = (4, 4, 4)   # fixed architecture for LR search

# Uniform depth sweep: (N,N,N) for N in [2,3,4,6]
# (4,4,4) is the default and overlaps with phase 1 — reused, not re-run
PHASE2_ITER_CONFIGS = [
    (2, 2, 2),
    (3, 3, 3),
    (4, 4, 4),
    (6, 6, 6),
]

DEFAULT_DATASET = "imagenet"
DEFAULT_EPOCHS  = 200
DEFAULT_BATCH   = 256
DEFAULT_OPT     = "adamw"
DEFAULT_WD      = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def lr_str(lr: float) -> str:
    """Format LR the same way train.py does for run-name construction."""
    return f"{lr:.0e}".replace('-0', '-').replace('+0', '')


def iters_str(iters: tuple[int, int, int]) -> str:
    return '-'.join(map(str, iters))


def run_name(iters: tuple[int, int, int], lr: float, channels: int = 128,
             dataset: str = DEFAULT_DATASET, opt: str = DEFAULT_OPT) -> str:
    return f"{dataset}_ch{channels}_iter{iters_str(iters)}_dil1-2-3-2_{opt}_lr{lr_str(lr)}"


def history_path(iters: tuple[int, int, int], lr: float,
                 dataset: str = DEFAULT_DATASET, opt: str = DEFAULT_OPT) -> Path:
    return Path(f"history_{run_name(iters, lr, dataset, opt)}.json")


def build_cmd(iters: tuple[int, int, int], lr: float, epochs: int,
              dataset: str = DEFAULT_DATASET, opt: str = DEFAULT_OPT,
              wd: float = DEFAULT_WD, batch: int = DEFAULT_BATCH) -> list[str]:
    n1, n2, n3 = iters
    return [
        sys.executable, "train.py",
        "--dataset",          dataset,
        "--stage-iterations", str(n1), str(n2), str(n3),
        "--lr",               str(lr),
        "--optimizer",        opt,
        "--weight-decay",     str(wd),
        "--batch-size",       str(batch),
        "--epochs",           str(epochs),
    ]


def load_best_acc(iters: tuple[int, int, int], lr: float,
                  dataset: str = DEFAULT_DATASET) -> float | None:
    p = history_path(iters, lr, dataset)
    if not p.exists():
        return None
    with open(p) as f:
        h = json.load(f)
    return max(h["test_acc"]) if h["test_acc"] else None


def load_history(iters: tuple[int, int, int], lr: float,
                 dataset: str = DEFAULT_DATASET) -> dict | None:
    p = history_path(iters, lr, dataset)
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
    print(f"\n  {'DONE' if ok else f'FAILED (exit {result.returncode})'} — {elapsed/60:.1f} min")
    return ok


def print_table(rows: list[dict], title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    if not rows:
        print("  (no results yet)")
        return
    keys = list(rows[0].keys())
    widths = [max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*keys))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(row[k]) for k in keys]))
    print(f"{'─'*60}")


# ── Phase 1: LR search ────────────────────────────────────────────────────────

def phase1(epochs: int, dataset: str, dry_run: bool) -> float | None:
    print("\n" + "█"*70)
    print(f"  PHASE 1 — LR search  (iters=4-4-4, AdamW, {dataset})")
    print("█"*70)

    for lr in PHASE1_LRS:
        if not dry_run and history_path(PHASE1_ITERS, lr, dataset).exists():
            print(f"\n  [SKIP] iters=4-4-4 lr={lr_str(lr)} — history file exists")
            continue
        cmd   = build_cmd(PHASE1_ITERS, lr, epochs, dataset)
        label = f"iters=4-4-4  lr={lr_str(lr)}  dataset={dataset}"
        ok    = run_experiment(cmd, label, dry_run)
        if not ok:
            print("  WARNING: run failed, continuing sweep...")

    rows = []
    for lr in PHASE1_LRS:
        best = load_best_acc(PHASE1_ITERS, lr, dataset)
        h    = load_history(PHASE1_ITERS, lr, dataset)
        avg_t = (sum(h["epoch_time"]) / len(h["epoch_time"])) if h else None
        rows.append({
            "lr":            lr_str(lr),
            "best_test_acc": f"{best:.2f}%" if best is not None else "—",
            "avg_s/epoch":   f"{avg_t:.1f}" if avg_t else "—",
        })
    print_table(rows, f"Phase 1 results (iters=4-4-4, AdamW, {dataset})")

    best_lr, best_acc = None, -1.0
    for lr in PHASE1_LRS:
        acc = load_best_acc(PHASE1_ITERS, lr, dataset)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_lr  = lr

    if best_lr is not None:
        print(f"\n  Best LR: {lr_str(best_lr)}  (test acc = {best_acc:.2f}%)")
    return best_lr


# ── Phase 2: Iteration-depth sweep ───────────────────────────────────────────

def phase2(best_lr: float, epochs: int, dataset: str, dry_run: bool):
    print("\n" + "█"*70)
    print(f"  PHASE 2 — Iteration sweep  (AdamW, lr={lr_str(best_lr)}, {dataset})")
    print(f"  Configs (N1,N2,N3): {PHASE2_ITER_CONFIGS}")
    print("█"*70)

    for cfg in PHASE2_ITER_CONFIGS:
        if not dry_run and history_path(cfg, best_lr, dataset).exists():
            print(f"\n  [SKIP] iters={iters_str(cfg)} lr={lr_str(best_lr)} — history file exists")
            continue
        cmd   = build_cmd(cfg, best_lr, epochs, dataset)
        label = f"iters={iters_str(cfg)} ({sum(cfg)} total)  lr={lr_str(best_lr)}"
        ok    = run_experiment(cmd, label, dry_run)
        if not ok:
            print("  WARNING: run failed, continuing sweep...")

    rows = []
    for cfg in PHASE2_ITER_CONFIGS:
        best = load_best_acc(cfg, best_lr, dataset)
        h    = load_history(cfg, best_lr, dataset)
        avg_t = (sum(h["epoch_time"]) / len(h["epoch_time"])) if h else None
        rows.append({
            "iters (N1,N2,N3)": iters_str(cfg),
            "total":             sum(cfg),
            "best_test_acc":    f"{best:.2f}%" if best is not None else "—",
            "avg_s/epoch":      f"{avg_t:.1f}" if avg_t else "—",
        })
    print_table(rows, f"Phase 2 results (AdamW, lr={lr_str(best_lr)}, {dataset})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECTiedNet experiment sweep")
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Run only phase 1 or phase 2 (default: both)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Fixed LR for phase 2 (skips phase 1 selection)')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        choices=["imagenet", "imagenet-mini-50", "cifar10"],
                        help=f'Dataset (default: {DEFAULT_DATASET})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Epochs per run (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)
    best_lr = args.lr

    if run_phase1:
        found_lr = phase1(args.epochs, args.dataset, args.dry_run)
        if best_lr is None:
            best_lr = found_lr

    if run_phase2:
        if best_lr is None:
            print("\nERROR: No LR for phase 2. Run phase 1 first or pass --lr <value>.")
            sys.exit(1)
        phase2(best_lr, args.epochs, args.dataset, args.dry_run)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
