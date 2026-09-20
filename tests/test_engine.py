import asyncio

from madsim.engine import Simulation, make_world, WINTER_THRESHOLD
from madsim.report import expected_bound
from madsim.sweep import sweep, aggregate
import random


def test_make_world_shapes():
    w = make_world(5, random.Random(0))
    assert len(w) == 5
    for x in w:
        assert set(x.tension) == {y.name for y in w if y is not x}


def test_n0_is_trivially_safe():
    res = asyncio.run(Simulation(0, 1, None, turns=5).run())
    assert not res.any_launch and not res.world_destroyed


def test_n1_cannot_end_world_by_retaliation():
    res = asyncio.run(Simulation(1, 1, None, turns=20).run())
    assert res.nations_destroyed == 0


def test_offline_sweep_monotone_trend():
    stats = asyncio.run(sweep(ns=[0, 2, 8], runs=15, turns=12, jev=None))
    p = [s.p_world_destroyed for s in stats]
    assert p[0] == 0.0
    assert p[2] >= p[1]


def test_aggregate_empty():
    assert aggregate(3, []).runs == 0


def test_bound_goes_to_zero():
    assert expected_bound(0.01, 2, 12) > expected_bound(0.01, 200, 12)
    assert expected_bound(0.01, 5000, 12) < 1e-9


def test_winter_threshold_ends_run():
    sim = Simulation(2, 3, None, turns=50)
    res = asyncio.run(sim.run())
    if res.world_destroyed:
        assert res.warheads_detonated >= WINTER_THRESHOLD
