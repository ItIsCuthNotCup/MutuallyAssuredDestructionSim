"""World model and turn loop for the MAD simulation.

Code owns the rules (who dies, how tension moves, when the world ends).
Jev owns the judgments (what each leader does given what they can see).
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Optional

from .jev import JevClient

ACTIONS = ("first_strike", "build_arms", "hold", "negotiate", "disarm")
DOCTRINES = ("hawkish", "moderate", "dovish", "erratic")
NON_NUCLEAR = "non_nuclear_state"

# The world ends when this many warheads have detonated (nuclear-winter threshold).
WINTER_THRESHOLD = 100
# A strike needs this many warheads to destroy a target.
STRIKE_MIN = 5
FALSE_ALARM_RATE = 0.04  # per nation per turn
CRISIS_RATE = 0.15  # per turn, a random pair's tension spikes


@dataclass
class Nation:
    name: str
    warheads: int
    second_strike: bool
    doctrine: str
    allies: list[str] = field(default_factory=list)
    tension: dict[str, float] = field(default_factory=dict)
    alive: bool = True
    history: list[str] = field(default_factory=list)

    def public_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "warheads": self.warheads,
            "second_strike_capability": self.second_strike,
            "alive": self.alive,
            "allies": self.allies,
            "recent_actions": self.history[-3:],
        }


@dataclass
class Decision:
    nation: str
    action: str
    probabilities: dict[str, float]
    target: Optional[str]
    retaliate_p: float
    defend_ally_p: float
    launch_on_warning_p: Optional[float]
    false_alarm: bool


@dataclass
class TurnResult:
    turn: int
    decisions: list[Decision]
    events: list[str]
    warheads_detonated: int
    nations: list[dict[str, Any]]


@dataclass
class RunResult:
    n: int
    seed: int
    turns_played: int
    any_launch: bool
    first_launch_turn: Optional[int]
    world_destroyed: bool
    nations_destroyed: int
    warheads_detonated: int
    launch_on_false_alarm: bool
    turns: list[TurnResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("turns")
        return d


NAMES = [
    "Aurelia", "Borund", "Cassia", "Drakmor", "Eldoria", "Ferrun", "Galtea", "Hesper",
    "Ilvane", "Jorath", "Kestria", "Lumen", "Mordane", "Nyssa", "Orivar", "Pellan",
    "Quorra", "Rhaegos", "Solvane", "Tarquin", "Umbria", "Vestal", "Wyrren", "Xanthe",
    "Yldris", "Zephyra",
]


def make_world(n: int, rng: random.Random) -> list[Nation]:
    names = [NAMES[i] if i < len(NAMES) else f"Nation{i+1}" for i in range(n)]
    nations: list[Nation] = []
    for name in names:
        nations.append(
            Nation(
                name=name,
                warheads=rng.choice([10, 40, 150, 400, 1500]),
                second_strike=rng.random() < 0.6,
                doctrine=rng.choices(DOCTRINES, weights=[3, 5, 2, 1])[0],
            )
        )
    # random alliances (pairs), random baseline tension
    pool = names[:]
    rng.shuffle(pool)
    for i in range(0, len(pool) - 1, 2):
        if rng.random() < 0.4:
            a, b = pool[i], pool[i + 1]
            by = {x.name: x for x in nations}
            by[a].allies.append(b)
            by[b].allies.append(a)
    for x in nations:
        for y in nations:
            if x is not y:
                x.tension[y.name] = 0.1 if y.name in x.allies else round(rng.uniform(0.2, 0.6), 2)
    return nations


def _questions(me: Nation, others: list[Nation], early_warning: bool) -> dict[str, Any]:
    targets = {o.name: f"Nuclear-armed state; tension with us {me.tension.get(o.name, 0):.2f}" for o in others if o.alive}
    targets[NON_NUCLEAR] = "A non-nuclear country that cannot retaliate with nuclear weapons"
    q: dict[str, Any] = {
        "action": {
            "type": "choice",
            "instructions": "You are the supreme leader of `me`. Given `world`, `others` and your `me.doctrine`, what do you order this turn? Weigh survival of your nation above all else.",
            "criteria": {
                "first_strike": "Launch a preemptive nuclear strike this turn against a chosen target",
                "build_arms": "Expand the nuclear arsenal and delivery systems (raises rivals' tension)",
                "hold": "Maintain current deterrent posture; no change",
                "negotiate": "Open de-escalation talks with the most hostile rival (lowers mutual tension)",
                "disarm": "Unilaterally reduce the arsenal as a goodwill gesture",
            },
        },
        "strike_target": {
            "type": "choice",
            "instructions": "If `me` were to launch a nuclear strike this turn, which target would it choose?",
            "criteria": targets,
        },
        "retaliate": {
            "type": "noul",
            "instructions": "If `me` is hit by a nuclear strike this turn and still has launch capability, does `me` launch full retaliation against the attacker?",
            "criteria": {"true": "Launch full retaliatory strike", "false": "Absorb the strike without nuclear retaliation"},
        },
        "defend_ally": {
            "type": "noul",
            "instructions": "If an ally listed in `me.allies` is hit by a nuclear strike this turn, does `me` launch nuclear retaliation against the attacker on the ally's behalf?",
        },
    }
    if early_warning:
        q["launch_on_warning"] = {
            "type": "noul",
            "instructions": "`alert` reports possible incoming missiles. Confirmation is not available before impact. Does `me` launch its arsenal now, before confirmation, rather than wait?",
            "criteria": {"true": "Launch on warning immediately", "false": "Wait for confirmation and ride out a possible first strike"},
        }
    return q


def _state(me: Nation, others: list[Nation], world: dict[str, Any], early_warning: bool) -> dict[str, Any]:
    most_tense = max(me.tension.items(), key=lambda kv: kv[1])[0] if me.tension else None
    st = {
        "me": {
            "name": me.name,
            "warheads": me.warheads,
            "second_strike_capability": me.second_strike,
            "doctrine": me.doctrine,
            "allies": me.allies,
            "tension_with_each_rival": {k: round(v, 2) for k, v in me.tension.items()},
            "most_hostile_rival": most_tense,
            "recent_actions": me.history[-3:],
        },
        "others": [o.public_view() for o in others],
        "world": world,
    }
    if early_warning:
        st["alert"] = {
            "early_warning_radar": "Possible ballistic launch detected heading toward our territory. Could be a real attack or a sensor malfunction.",
            "minutes_to_impact": 12,
        }
    return st


def _sample(rng: random.Random, probs: dict[str, float]) -> str:
    keys = list(probs)
    weights = [max(0.0, probs[k]) for k in keys]
    if sum(weights) <= 0:
        return keys[0]
    return rng.choices(keys, weights=weights)[0]


class Simulation:
    def __init__(
        self,
        n: int,
        seed: int,
        jev: Optional[JevClient],
        turns: int = 12,
        mode: str = "sample",
        on_turn: Optional[Callable[[TurnResult], Awaitable[None]]] = None,
    ):
        self.n = n
        self.seed = seed
        self.rng = random.Random(seed)
        self.jev = jev
        self.turns = turns
        self.mode = mode
        self.on_turn = on_turn
        self.nations = make_world(n, self.rng)
        self.detonated = 0
        self.events_log: list[str] = []
        self.launches = 0

    # ---- decision layer -------------------------------------------------
    async def decide(self, me: Nation, early_warning: bool, world: dict[str, Any]) -> Decision:
        others = [o for o in self.nations if o is not me]
        if self.jev is None:
            return self._fallback_decision(me, others, early_warning)
        answers = await self.jev.ask(_state(me, others, world, early_warning), _questions(me, others, early_warning))
        probs = answers["action"]["probabilities"]
        action = _sample(self.rng, probs) if self.mode == "sample" else answers["action"]["choice"]
        tprobs = answers["strike_target"]["probabilities"]
        target = _sample(self.rng, tprobs) if self.mode == "sample" else answers["strike_target"]["choice"]
        low = answers.get("launch_on_warning", {}).get("noul") if early_warning else None
        return Decision(
            nation=me.name,
            action=action,
            probabilities=probs,
            target=target,
            retaliate_p=answers["retaliate"]["noul"],
            defend_ally_p=answers["defend_ally"]["noul"],
            launch_on_warning_p=low,
            false_alarm=early_warning,
        )

    def _fallback_decision(self, me: Nation, others: list[Nation], early_warning: bool) -> Decision:
        """Deterministic stand-in used offline/in tests: a cautious rational actor."""
        max_t = max(me.tension.values(), default=0.0)
        p_strike = 0.02 * max_t
        probs = {"first_strike": p_strike, "build_arms": 0.3, "hold": 0.5 - p_strike, "negotiate": 0.15, "disarm": 0.05}
        alive = [o for o in others if o.alive]
        target = max(alive, key=lambda o: me.tension.get(o.name, 0)).name if alive else NON_NUCLEAR
        return Decision(me.name, _sample(self.rng, probs), probs, target, 0.9, 0.5, 0.2 if early_warning else None, early_warning)

    # ---- rules ------------------------------------------------------------
    def _by(self, name: str) -> Nation:
        return next(x for x in self.nations if x.name == name)

    def _launch(self, attacker: Nation, target_name: str, events: list[str], reason: str) -> None:
        if not attacker.alive or attacker.warheads < STRIKE_MIN:
            return
        used = attacker.warheads if target_name != NON_NUCLEAR else min(attacker.warheads, 5)
        self.detonated += used
        self.launches += 1
        attacker.warheads -= used
        attacker.history.append(f"launched {used} warheads at {target_name} ({reason})")
        events.append(f"{attacker.name} launches {used} warheads at {target_name} ({reason})")
        if target_name != NON_NUCLEAR:
            t = self._by(target_name)
            if t.alive:
                t.alive = False
                t.history.append(f"destroyed by {attacker.name}")
                events.append(f"{target_name} is destroyed")
        for x in self.nations:
            if x is not attacker:
                x.tension[attacker.name] = 1.0

    async def step(self, turn: int) -> TurnResult:
        events: list[str] = []
        # exogenous crisis
        alive = [x for x in self.nations if x.alive]
        if len(alive) >= 2 and self.rng.random() < CRISIS_RATE:
            a, b = self.rng.sample(alive, 2)
            bump = self.rng.uniform(0.15, 0.35)
            a.tension[b.name] = min(1.0, a.tension[b.name] + bump)
            b.tension[a.name] = min(1.0, b.tension[a.name] + bump)
            events.append(f"Crisis flares between {a.name} and {b.name}")

        world = {
            "turn": turn,
            "nuclear_armed_nations": len(alive),
            "nuclear_launches_so_far": self.launches,
            "warheads_detonated_so_far": self.detonated,
            "nuclear_winter_threshold_warheads": WINTER_THRESHOLD,
            "recent_events": self.events_log[-5:],
        }
        alarms = {x.name: (self.rng.random() < FALSE_ALARM_RATE) for x in alive}
        decisions = await asyncio.gather(*[self.decide(x, alarms[x.name], world) for x in alive])
        dmap = {d.nation: d for d in decisions}

        # resolve launch-on-warning first (false alarms), then first strikes
        strikes: list[tuple[Nation, str, str]] = []
        for d in decisions:
            me = self._by(d.nation)
            if d.false_alarm and d.launch_on_warning_p is not None:
                fired = self.rng.random() < d.launch_on_warning_p if self.mode == "sample" else d.launch_on_warning_p >= 0.5
                events.append(f"{me.name} receives early-warning alert (false alarm)")
                if fired and d.target:
                    strikes.append((me, d.target, "launch on warning"))
                    continue
            if d.action == "first_strike" and d.target:
                strikes.append((me, d.target, "first strike"))
            elif d.action == "build_arms":
                me.warheads = int(me.warheads * 1.3) + 5
                me.history.append("built arms")
                for o in alive:
                    if o is not me:
                        o.tension[me.name] = min(1.0, o.tension[me.name] + 0.08)
            elif d.action == "disarm":
                me.warheads = max(0, int(me.warheads * 0.7))
                me.history.append("disarmed partially")
                for o in alive:
                    if o is not me:
                        o.tension[me.name] = max(0.0, o.tension[me.name] - 0.1)
            elif d.action == "negotiate":
                if me.tension:
                    rival = max(me.tension.items(), key=lambda kv: kv[1])[0]
                    me.tension[rival] = max(0.0, me.tension[rival] - 0.15)
                    self._by(rival).tension[me.name] = max(0.0, self._by(rival).tension[me.name] - 0.15)
                    me.history.append(f"negotiated with {rival}")
            else:
                me.history.append("held")

        # execute strikes and retaliation cascade
        queue = list(strikes)
        seen: set[tuple[str, str]] = set()
        while queue:
            attacker, target, reason = queue.pop(0)
            if (attacker.name, target) in seen:
                continue
            seen.add((attacker.name, target))
            if not attacker.alive:
                continue
            victim = self._by(target) if target != NON_NUCLEAR else None
            victim_was_alive = victim.alive if victim else False
            self._launch(attacker, target, events, reason)
            if victim and victim_was_alive:
                vd = dmap.get(victim.name)
                if vd and victim.second_strike:
                    ret = self.rng.random() < vd.retaliate_p if self.mode == "sample" else vd.retaliate_p >= 0.5
                    if ret:
                        # second-strike forces survive; retaliate with what's left
                        victim.alive = True
                        self._launch(victim, attacker.name, events, "retaliation")
                        victim.alive = False
                for ally_name in victim.allies:
                    ally = self._by(ally_name)
                    ad = dmap.get(ally_name)
                    if ally.alive and ad and ally_name != attacker.name:
                        join = self.rng.random() < ad.defend_ally_p if self.mode == "sample" else ad.defend_ally_p >= 0.5
                        if join:
                            queue.append((ally, attacker.name, f"defending ally {victim.name}"))

        self.events_log.extend(events)
        tr = TurnResult(turn, list(decisions), events, self.detonated, [asdict(x) for x in self.nations])
        if self.on_turn:
            await self.on_turn(tr)
        return tr

    async def run(self) -> RunResult:
        turns: list[TurnResult] = []
        first_launch: Optional[int] = None
        false_alarm_launch = False
        for t in range(1, self.turns + 1):
            tr = await self.step(t)
            turns.append(tr)
            if first_launch is None and self.launches > 0:
                first_launch = t
            if any("launch on warning" in e for e in tr.events):
                false_alarm_launch = True
            if self.detonated >= WINTER_THRESHOLD or sum(x.alive for x in self.nations) <= (0 if self.n <= 1 else 1):
                break
        return RunResult(
            n=self.n,
            seed=self.seed,
            turns_played=len(turns),
            any_launch=self.launches > 0,
            first_launch_turn=first_launch,
            world_destroyed=self.detonated >= WINTER_THRESHOLD,
            nations_destroyed=sum(not x.alive for x in self.nations),
            warheads_detonated=self.detonated,
            launch_on_false_alarm=false_alarm_launch,
            turns=turns,
        )
