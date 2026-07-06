import hashlib
import json
from collections.abc import Callable, Iterable
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from lookback.backtest.vectorised import VectorisedBacktester
from lookback.strategies.base import Strategy

# Per-worker globals, set once by the pool initializer so the (large) price
# frame and the factory are shipped to each process a single time.
_WORKER: dict = {}


def _param_hash(params: dict) -> str:
    """Deterministic id for a param combo (order-independent)."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _init_worker(build: Callable[..., Strategy], prices: pd.DataFrame,
                 cost_bps: float) -> None:
    _WORKER["build"] = build
    _WORKER["prices"] = prices
    _WORKER["backtester"] = VectorisedBacktester(cost_bps=cost_bps)


def run_one(params: dict) -> dict | None:
    """Backtest one combo in a worker. Top-level so it pickles. None if invalid."""
    try:
        strategy = _WORKER["build"](**params)
    except ValueError:
        return None
    result = _WORKER["backtester"].run(strategy, _WORKER["prices"])
    return {**params, **result.summary(), "param_hash": _param_hash(params)}


def pending_params(grid: Iterable[dict], checkpoint_dir: Path) -> list[dict]:
    """Combos not yet checkpointed — the resume set."""
    done = {p.stem for p in Path(checkpoint_dir).glob("*.parquet")}
    return [p for p in grid if _param_hash(p) not in done]


def _collect(checkpoint_dir: Path, rank_by: str) -> pd.DataFrame:
    files = sorted(Path(checkpoint_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop(columns=["param_hash"], errors="ignore")
    if rank_by in df.columns:
        df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    return df


def run_sweep_parallel(build: Callable[..., Strategy], prices: pd.DataFrame,
                       grid: Iterable[dict], checkpoint_dir: str | Path, *,
                       cost_bps: float = 0.0, rank_by: str = "sharpe",
                       processes: int | None = None) -> pd.DataFrame:
    """Run a sweep across processes, checkpointing each result for resume.

    Re-running with the same checkpoint_dir skips already-finished combos and
    computes only the rest, then returns the full ranked table.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    todo = pending_params(list(grid), checkpoint_dir)
    if todo:
        with Pool(processes, initializer=_init_worker,
                  initargs=(build, prices, cost_bps)) as pool:
            for row in pool.imap_unordered(run_one, todo):
                if row is None:
                    continue
                # Persist immediately: a crash keeps everything already done.
                path = checkpoint_dir / f"{row['param_hash']}.parquet"
                pd.DataFrame([row]).to_parquet(path)

    return _collect(checkpoint_dir, rank_by)
