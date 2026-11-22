"""The simulation, as a function RHAPSODY's pool workers run.

    run_simulation(store_spec, key, index, temperature, doping, bench)

It spins a CPU (or sleeps, with `bench.sim_burn` off) for the duration drawn from the candidate and
the parameters, then writes the result tensor to the store under `key` with `put_tensor`, and
returns only the key with its timestamps. Each worker opens its store once (`open_store`): a client
per task would attach to the dictionary each time, and every attach leaks Dragon channels.
"""

from __future__ import annotations

import socket
import time

from autonmi.store import open_store

from . import physics
from .benchmark import Benchmark


def _spin(seconds: float) -> int:
    """Keep one CPU busy for `seconds` of wall time."""
    deadline = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < deadline:
        for _ in range(2000):
            x += 1
    return x


def run_simulation(
    store_spec: str, key: str, index: int, temperature: float, doping: float, bench: Benchmark
) -> dict:
    t_start = time.time()
    c = physics.candidate(index, bench)
    seconds = physics.sim_seconds(c, temperature, doping, bench)
    if bench.sim_burn:
        _spin(seconds)
    elif seconds > 0:
        time.sleep(seconds)
    t_compute_end = time.time()
    import numpy as np

    tensor = np.asarray(
        physics.spectrum(c, temperature, doping, bench.sim_out_elems), dtype=np.float64
    )
    store = open_store(store_spec)
    t = time.perf_counter()
    store.put_tensor(key, tensor)
    put_s = time.perf_counter() - t
    return {
        "key": key,
        "seconds": seconds,
        "host": socket.gethostname(),
        "t_start": t_start,
        "t_compute_end": t_compute_end,
        "t_end": time.time(),
        "put_s": put_s,
    }
