"""CLI: run a Monte Carlo sweep over n and write results/ + a chart.

    python run_sweep.py --ns 0,1,2,3,4,6,8,12,16,24 --runs 8 --turns 12
    python run_sweep.py --no-jev   # offline fallback policy (no API calls)
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from madsim.engine import SCENARIOS, DEFAULT_SCENARIO
from madsim.jev import JevClient
from madsim.sweep import sweep, save, DEFAULT_NS
from madsim.report import chart, write_analysis


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)))
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--mode", choices=["sample", "argmax"], default=None, help="default: the scenario's decision mode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=DEFAULT_SCENARIO)
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ns = [int(x) for x in a.ns.split(",") if x.strip()]
    jev = None if a.no_jev else JevClient()
    t0 = time.time()

    async def on_n(st):
        print(f"n={st.n:3d}  P(launch)={st.p_any_launch:.2f}  P(world destroyed)={st.p_world_destroyed:.2f} "
              f"±{st.ci95_world_destroyed:.2f}  P(false-alarm launch)={st.p_false_alarm_launch:.2f}  hazard/turn={st.per_turn_hazard:.3f}")

    stats = await sweep(ns, a.runs, a.turns, jev, a.mode, a.seed, on_n=on_n, scenario=a.scenario)
    meta = {"ns": ns, "runs": a.runs, "turns": a.turns, "mode": a.mode, "seed": a.seed, "scenario": a.scenario,
            "model": "jev-latest" if jev else "fallback", "jev_calls": jev.calls if jev else 0,
            "input_tokens": jev.input_tokens if jev else 0, "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    stamp = int(t0)
    save(stats, out / f"sweep_{stamp}.json", meta)
    chart(stats, out / f"sweep_{stamp}.png")
    write_analysis(stats, meta, out / f"sweep_{stamp}.md")
    print(f"wrote {out}/sweep_{stamp}.{{json,png,md}}  ({meta['jev_calls']} Jev calls, {meta['seconds']}s)")
    if jev:
        await jev.close()


if __name__ == "__main__":
    asyncio.run(main())
