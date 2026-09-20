"""Monte Carlo sweep over the number of nuclear-armed nations n."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .engine import Simulation, RunResult
from .jev import JevClient

DEFAULT_NS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]


@dataclass
class NStats:
    n: int
    runs: int
    p_any_launch: float
    p_world_destroyed: float
    p_false_alarm_launch: float
    mean_nations_destroyed: float
    mean_first_launch_turn: Optional[float]
    mean_warheads_detonated: float
    ci95_world_destroyed: float
    per_turn_hazard: float  # P(first launch on a given turn | no launch yet), pooled
    runs_detail: list[dict[str, Any]] = field(default_factory=list)


def aggregate(n: int, results: list[RunResult]) -> NStats:
    r = len(results)
    if r == 0:
        return NStats(n, 0, 0, 0, 0, 0, None, 0, 0, 0)
    p_launch = sum(x.any_launch for x in results) / r
    p_dead = sum(x.world_destroyed for x in results) / r
    p_fa = sum(x.launch_on_false_alarm for x in results) / r
    firsts = [x.first_launch_turn for x in results if x.first_launch_turn is not None]
    # pooled hazard: launches / nation-turns at risk before first launch
    exposure = sum((x.first_launch_turn or x.turns_played) for x in results)
    hazard = len(firsts) / exposure if exposure else 0.0
    return NStats(
        n=n,
        runs=r,
        p_any_launch=p_launch,
        p_world_destroyed=p_dead,
        p_false_alarm_launch=p_fa,
        mean_nations_destroyed=sum(x.nations_destroyed for x in results) / r,
        mean_first_launch_turn=(sum(firsts) / len(firsts)) if firsts else None,
        mean_warheads_detonated=sum(x.warheads_detonated for x in results) / r,
        ci95_world_destroyed=1.96 * math.sqrt(p_dead * (1 - p_dead) / r),
        per_turn_hazard=hazard,
        runs_detail=[x.summary() for x in results],
    )


async def sweep(
    ns: list[int] = DEFAULT_NS,
    runs: int = 6,
    turns: int = 12,
    jev: Optional[JevClient] = None,
    mode: str = "sample",
    seed0: int = 1,
    on_run: Optional[Callable[[RunResult], Awaitable[None]]] = None,
    on_n: Optional[Callable[[NStats], Awaitable[None]]] = None,
    run_concurrency: int = 4,
) -> list[NStats]:
    out: list[NStats] = []
    for n in ns:
        sem = asyncio.Semaphore(run_concurrency)

        async def one(k: int) -> RunResult:
            async with sem:
                sim = Simulation(n, seed0 + 1000 * n + k, jev, turns=turns, mode=mode)
                res = await sim.run()
                if on_run:
                    await on_run(res)
                return res

        results = await asyncio.gather(*[one(k) for k in range(runs)])
        st = aggregate(n, list(results))
        out.append(st)
        if on_n:
            await on_n(st)
    return out


def save(stats: list[NStats], path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "stats": [asdict(s) for s in stats]}, indent=2))
