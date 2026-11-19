"""Terminal heartbeat for long blocking calls; no algorithm state is touched."""
from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time


@contextmanager
def terminal_heartbeat(label: str, *, enabled: bool = True, interval: float = 15.0):
    """Print elapsed wall time while waiting, and always join the helper thread."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("heartbeat interval must be finite and positive")
    if not enabled:
        yield
        return
    stopped = threading.Event()
    started = time.perf_counter()

    def report() -> None:
        while not stopped.wait(interval):
            print(f"[进度] {label}：仍在运行，已用时 {time.perf_counter()-started:.1f}s", flush=True)

    worker = threading.Thread(target=report, name="snd-terminal-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
