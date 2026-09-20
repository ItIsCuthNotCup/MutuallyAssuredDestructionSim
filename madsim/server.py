"""FastAPI dashboard: start sweeps / single runs and stream events over SSE."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .engine import Simulation, TurnResult, RunResult
from .jev import JevClient
from .sweep import sweep, save, NStats, DEFAULT_NS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
app = FastAPI(title="MAD Sim")

_subscribers: set[asyncio.Queue] = set()
_state: dict[str, Any] = {"running": False, "stats": [], "log": [], "meta": {}, "world": None}
LIVE_TURN_DELAY = 1.2  # seconds between years in a single live world so viewers can follow
_task: Optional[asyncio.Task] = None


def _load_latest() -> None:
    files = sorted(RESULTS.glob("sweep_*.json")) if RESULTS.exists() else []
    if not files:
        return
    data = json.loads(files[-1].read_text())
    _state["stats"] = [{k: v for k, v in s.items() if k != "runs_detail"} for s in data["stats"]]
    _state["meta"] = data["meta"]
    _state["log"] = [f"Loaded saved sweep {files[-1].name} ({data['meta'].get('jev_calls', 0)} Jev judgments)"]


_load_latest()


async def publish(kind: str, data: Any) -> None:
    msg = {"kind": kind, "data": data, "t": time.time()}
    if kind == "log":
        _state["log"] = (_state["log"] + [data])[-400:]
    for q in list(_subscribers):
        await q.put(msg)


class SweepParams(BaseModel):
    ns: list[int] = DEFAULT_NS
    runs: int = 6
    turns: int = 12
    mode: str = "sample"
    use_jev: bool = True
    seed: int = 1


class RunParams(BaseModel):
    n: int = 5
    turns: int = 12
    mode: str = "sample"
    use_jev: bool = True
    seed: int = 42


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (ROOT / "madsim" / "static" / "index.html").read_text()


@app.get("/state")
async def state() -> JSONResponse:
    return JSONResponse(_state)


@app.get("/results")
async def results() -> JSONResponse:
    files = sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []
    return JSONResponse([f.name for f in files])


@app.get("/results/{name}")
async def result_file(name: str) -> JSONResponse:
    return JSONResponse(json.loads((RESULTS / name).read_text()))


@app.get("/events")
async def events() -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.add(q)

    async def gen():
        try:
            yield f"data: {json.dumps({'kind': 'snapshot', 'data': _state})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _stats_payload(st: NStats) -> dict[str, Any]:
    d = asdict(st)
    d.pop("runs_detail")
    return d


@app.post("/sweep")
async def start_sweep(p: SweepParams) -> JSONResponse:
    global _task
    if _state["running"]:
        return JSONResponse({"error": "already running"}, status_code=409)
    _state.update(running=True, stats=[], log=[], meta=p.model_dump())
    _task = asyncio.create_task(_run_sweep(p))
    return JSONResponse({"ok": True})


@app.post("/run")
async def start_run(p: RunParams) -> JSONResponse:
    global _task
    if _state["running"]:
        return JSONResponse({"error": "already running"}, status_code=409)
    _state.update(running=True, log=[], meta=p.model_dump())
    _task = asyncio.create_task(_run_single(p))
    return JSONResponse({"ok": True})


@app.post("/stop")
async def stop() -> JSONResponse:
    if _task and not _task.done():
        _task.cancel()
    _state["running"] = False
    await publish("done", {"cancelled": True})
    return JSONResponse({"ok": True})


async def _run_single(p: RunParams) -> None:
    jev = JevClient() if p.use_jev else None
    try:
        async def on_turn(tr: TurnResult) -> None:
            payload = {
                "turn": tr.turn,
                "decisions": [asdict(d) for d in tr.decisions],
                "events": tr.events,
                "warheads_detonated": tr.warheads_detonated,
                "nations": [{k: v for k, v in n.items() if k != "history"} for n in tr.nations],
            }
            _state["world"] = payload
            await publish("turn", payload)
            for e in tr.events:
                await publish("log", f"[n={p.n} turn {tr.turn}] {e}")
            await asyncio.sleep(LIVE_TURN_DELAY)

        sim = Simulation(p.n, p.seed, jev, turns=p.turns, mode=p.mode, on_turn=on_turn)
        await publish("log", f"Single run: n={p.n}, seed={p.seed}, mode={p.mode}, jev={'on' if jev else 'off'}")
        _state["world"] = {"turn": 0, "decisions": [], "events": [], "warheads_detonated": 0, "nations": [asdict(x) for x in sim.nations]}
        await publish("world", {"nations": [asdict(x) for x in sim.nations]})
        res = await sim.run()
        await publish("run", res.summary())
        await publish("done", {"jev_calls": jev.calls if jev else 0})
    except asyncio.CancelledError:
        pass
    except Exception as e:  # surface to UI
        await publish("log", f"ERROR: {e!r}")
        await publish("done", {"error": str(e)})
    finally:
        _state["running"] = False
        if jev:
            await jev.close()


async def _run_sweep(p: SweepParams) -> None:
    jev = JevClient() if p.use_jev else None
    started = time.time()
    try:
        async def on_run(res: RunResult) -> None:
            s = res.summary()
            await publish("run", s)
            tag = "WORLD DESTROYED" if res.world_destroyed else ("launch" if res.any_launch else "peace")
            await publish("log", f"[n={res.n} seed={res.seed}] {tag}; nations destroyed={res.nations_destroyed}, warheads={res.warheads_detonated}, turns={res.turns_played}")

        async def on_n(st: NStats) -> None:
            payload = _stats_payload(st)
            _state["stats"].append(payload)
            await publish("nstats", payload)

        await publish("log", f"Sweep start: ns={p.ns} runs={p.runs} turns={p.turns} mode={p.mode} jev={'on' if jev else 'off'}")
        stats = await sweep(p.ns, p.runs, p.turns, jev, p.mode, p.seed, on_run=on_run, on_n=on_n)
        meta = {**p.model_dump(), "jev_calls": jev.calls if jev else 0, "input_tokens": jev.input_tokens if jev else 0,
                "seconds": round(time.time() - started, 1), "model": "jev-latest" if jev else "fallback"}
        fname = f"sweep_{int(started)}.json"
        save(stats, RESULTS / fname, meta)
        await publish("log", f"Saved results/{fname}; Jev calls={meta['jev_calls']} in {meta['seconds']}s")
        await publish("done", meta)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        await publish("log", f"ERROR: {e!r}")
        await publish("done", {"error": str(e)})
    finally:
        _state["running"] = False
        if jev:
            await jev.close()
