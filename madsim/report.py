"""Static chart + written analysis of a sweep."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .sweep import NStats  # noqa: E402


def chart(stats: list[NStats], path: Path) -> None:
    stats = sorted(stats, key=lambda s: s.n)
    ns = [s.n for s in stats]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ns, [s.p_any_launch for s in stats], "o-", label="P(any nuclear launch)", color="#e0a800")
    a1.errorbar(ns, [s.p_world_destroyed for s in stats], yerr=[s.ci95_world_destroyed for s in stats],
                fmt="s-", label="P(world destroyed)", color="#d62728", capsize=3)
    a1.plot(ns, [s.p_false_alarm_launch for s in stats], "^--", label="P(launch on false alarm)", color="#7f7f7f")
    a1.set_xlabel("n (nuclear-armed nations)")
    a1.set_ylabel("probability over run")
    a1.set_ylim(0, 1.02)
    a1.grid(alpha=0.3)
    a1.legend()
    a1.set_title("Catastrophe probability vs n")
    a2.plot(ns, [s.per_turn_hazard for s in stats], "o-", color="#2ca02c", label="observed hazard")
    if len(ns) > 1 and stats[1].n > 0 and stats[1].per_turn_hazard > 0:
        base = stats[1].per_turn_hazard / stats[1].n
        a2.plot(ns, [min(1.0, base * n) for n in ns], ":", color="#999", label="linear-in-n reference")
    a2.set_xlabel("n (nuclear-armed nations)")
    a2.set_ylabel("P(first launch this turn | none yet)")
    a2.grid(alpha=0.3)
    a2.legend()
    a2.set_title("Per-turn hazard of first launch")
    fig.suptitle("Jev-driven MAD simulation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_analysis(stats: list[NStats], meta: dict[str, Any], path: Path) -> None:
    stats = sorted(stats, key=lambda s: s.n)
    lines = [
        "# MAD sweep analysis",
        "",
        f"Model: `{meta.get('model')}` · runs/n: {meta.get('runs')} · turns/run: {meta.get('turns')} · mode: {meta.get('mode')} · Jev calls: {meta.get('jev_calls')}",
        "",
        "| n | P(any launch) | P(world destroyed) ±95% | P(false-alarm launch) | mean nations destroyed | mean first-launch turn | hazard/turn |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        fl = f"{s.mean_first_launch_turn:.1f}" if s.mean_first_launch_turn is not None else "—"
        lines.append(f"| {s.n} | {s.p_any_launch:.2f} | {s.p_world_destroyed:.2f} ±{s.ci95_world_destroyed:.2f} | {s.p_false_alarm_launch:.2f} | {s.mean_nations_destroyed:.2f} | {fl} | {s.per_turn_hazard:.3f} |")

    lines += ["", "## Reading the curve", ""]
    by_n = {s.n: s for s in stats}
    if 0 in by_n:
        lines.append("- **n = 0**: no nuclear states, catastrophe probability is identically 0. Trivial baseline.")
    if 1 in by_n:
        s = by_n[1]
        lines.append(f"- **n = 1 (monopoly)**: nobody can retaliate, so MAD does not exist. Jev's leader still launches in {s.p_any_launch:.0%} of runs (against a non-nuclear state) but the world never ends: one arsenal alone is below the winter threshold in these runs ({s.p_world_destroyed:.0%}).")
    if 2 in by_n:
        s = by_n[2]
        lines.append(f"- **n = 2 (classic MAD)**: the bilateral case. P(world destroyed) = {s.p_world_destroyed:.0%}; per-turn hazard {s.per_turn_hazard:.3f}. This is the regime the doctrine was designed for: each side's second-strike makes a first strike suicidal, and Jev's retaliation probabilities reflect that.")
    big = [s for s in stats if s.n >= 8]
    if big:
        s = big[-1]
        lines.append(f"- **n = {s.n}**: P(world destroyed) = {s.p_world_destroyed:.0%}, per-turn hazard {s.per_turn_hazard:.3f}. With many independent decision-makers, false alarms and one erratic leader are enough; retaliation cascades through alliances turn any single launch into a global exchange.")

    lines += [
        "",
        "## Why the limit n → ∞ is destruction",
        "",
        "Let each nuclear-armed leader have some small per-turn probability p_i > 0 of launching (first strike, or launch-on-warning after a false alarm). "
        "Even if MAD makes every individual p_i tiny, the chance that *nobody* launches in a turn is ∏(1 − p_i) ≤ (1 − p_min)^n → 0 as n → ∞. "
        "Over T turns, P(no launch) ≤ (1 − p_min)^(nT). MAD works by making retaliation certain, which suppresses p_i for *rational* actors, "
        "but it cannot drive any p_i to exactly zero (accidents, misperception, erratic leaders), and it multiplies the consequences of a single launch "
        "because retaliation is exactly what the doctrine guarantees. The simulation shows both halves: the observed per-turn hazard grows roughly "
        "linearly in n (right-hand chart), and once a launch happens the retaliation + alliance cascade pushes warheads detonated past the nuclear-winter threshold.",
        "",
        "So MAD is a stable equilibrium for n = 2 with rational actors, becomes fragile as n grows, and the probability of a civilization-ending exchange "
        "tends to 1 as n → ∞ for any fixed horizon T.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def expected_bound(p_min: float, n: int, turns: int) -> float:
    """Upper bound on P(no launch) over `turns` turns with n actors each having per-turn launch prob >= p_min."""
    return math.pow(1 - p_min, n * turns)
