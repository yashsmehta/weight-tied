"""SQLite results storage for eval runs, ported as-is from visreps/utils.py.

Schema (including the pca_* columns) is kept exactly as visreps' — this
project's configs simply never set pca_labels/pca_n_classes/etc, so those
columns land as False/None/1 on every row. Keeping the schema identical
means results.db stays diffable/mergeable against a visreps-produced one.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils import rprint

_RESULTS_DB_PATH = Path("results.db")

_IDENTITY_FIELDS = (
    "seed", "epoch", "region", "subject_idx", "neural_dataset", "cfg_id",
    "pca_labels", "pca_n_classes", "pca_labels_folder", "checkpoint_dir",
    "analysis", "compare_method", "reconstruct_from_pcs", "pca_k", "model_name",
)


def _to_plain(value):
    """Unwrap OmegaConf List/DictConfig so json.dumps can serialize it."""
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _compute_run_id(cfg) -> str:
    """Deterministic hash of experiment identity fields."""
    identity = {f: _to_plain(cfg.get(f)) for f in _IDENTITY_FIELDS}
    identity["subject_idx"] = str(identity.get("subject_idx"))
    raw = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _init_db(db_path) -> sqlite3.Connection:
    """Open (or create) the results SQLite database with WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            run_id              TEXT NOT NULL,
            compare_method      TEXT NOT NULL,
            layer               TEXT NOT NULL,
            score               REAL,
            ci_low              REAL,
            ci_high             REAL,
            analysis            TEXT NOT NULL,
            seed                INTEGER NOT NULL,
            epoch               INTEGER NOT NULL,
            region              TEXT,
            subject_idx         TEXT,
            neural_dataset      TEXT NOT NULL,
            cfg_id              INTEGER,
            pca_labels          BOOLEAN NOT NULL,
            pca_n_classes       INTEGER,
            pca_labels_folder   TEXT,
            model_name          TEXT NOT NULL,
            checkpoint_dir      TEXT,
            reconstruct_from_pcs BOOLEAN DEFAULT 0,
            pca_k               INTEGER DEFAULT 1,
            UNIQUE(run_id, compare_method, layer)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_configs (
            run_id      TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS layer_selection_scores (
            run_id          TEXT NOT NULL,
            compare_method  TEXT NOT NULL,
            layer           TEXT NOT NULL,
            score           REAL,
            UNIQUE(run_id, compare_method, layer)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bootstrap_distributions (
            run_id          TEXT NOT NULL,
            compare_method  TEXT NOT NULL,
            scores          TEXT,
            UNIQUE(run_id, compare_method)
        )
    """)
    conn.commit()
    return conn


def _get_float(row, col):
    """Safely extract a float from a DataFrame row, returning None if missing/NaN."""
    if col in row.index and pd.notna(row.get(col)):
        return float(row[col])
    return None


def save_results(df, cfg, timeout=60):
    """Save evaluation results to SQLite database at results.db.

    Uses a normalized "long" format: each comparison metric (Spearman, Kendall,
    Pearson) gets its own row, distinguished by `compare_method`, across all
    three tables (`results`, `layer_selection_scores`, `bootstrap_distributions`).
    Re-running the same eval replaces old rows for the same run_id.
    """
    run_id = _compute_run_id(cfg)
    conn = _init_db(_RESULTS_DB_PATH)

    config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    conn.execute(
        "INSERT OR REPLACE INTO run_configs (run_id, config_json) VALUES (?, ?)",
        (run_id, config_json),
    )

    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        layer = row.get("layer")
        score = _get_float(row, "score")
        ci_low = _get_float(row, "ci_low")
        ci_high = _get_float(row, "ci_high")
        if score is None:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO results
               (run_id, compare_method, layer, score, ci_low, ci_high,
                analysis, seed, epoch, region, subject_idx,
                neural_dataset, cfg_id, pca_labels, pca_n_classes, pca_labels_folder,
                model_name, checkpoint_dir, reconstruct_from_pcs, pca_k)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, method, layer, score, ci_low, ci_high,
                row.get("analysis", cfg.get("analysis")),
                int(cfg.get("seed")),
                int(cfg.get("epoch", 0)),
                str(cfg.get("region")),
                str(cfg.get("subject_idx")),
                cfg.get("neural_dataset"),
                cfg.get("cfg_id"),
                bool(cfg.get("pca_labels", False)),
                cfg.get("pca_n_classes"),
                cfg.get("pca_labels_folder"),
                cfg.get("model_name"),
                cfg.get("checkpoint_dir"),
                bool(cfg.get("reconstruct_from_pcs", False)),
                cfg.get("pca_k", 1),
            ),
        )

    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        entries = row.get("layer_selection_scores") or []
        for entry in entries:
            conn.execute(
                """INSERT OR REPLACE INTO layer_selection_scores
                   (run_id, compare_method, layer, score) VALUES (?, ?, ?, ?)""",
                (run_id, method, entry["layer"], float(entry["score"])),
            )

    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        bs = row.get("bootstrap_scores")
        if bs is not None:
            conn.execute(
                """INSERT OR REPLACE INTO bootstrap_distributions
                   (run_id, compare_method, scores) VALUES (?, ?, ?)""",
                (run_id, method, json.dumps(bs)),
            )

    conn.commit()
    conn.close()
    rprint(f"Saved {len(df)} results to {_RESULTS_DB_PATH} (run_id={run_id})", style="success")
    return str(_RESULTS_DB_PATH)
