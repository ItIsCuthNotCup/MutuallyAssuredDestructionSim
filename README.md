# Mutually Assured Destruction Simulator

A Monte Carlo simulation of `n` nuclear-armed nations where **Jev** (TypeSafe's System One
model) is the decision-maker for every leader. Sweep `n` from 0 upward to see whether MAD
holds, and watch it break as `n → ∞`.

## How Jev is used

Jev does not generate text; it returns typed judgments with calibrated probabilities. Each
turn, every living nation gets one Jev request whose `state` is that leader's view of the
world (own arsenal, doctrine, allies, tension with each rival, recent events, any early-warning
alert) and whose `questions` are:

| question | type | meaning |
|---|---|---|
| `action` | choice | first_strike / build_arms / hold / negotiate / disarm |
| `strike_target` | choice | whom to hit if launching (speculative, asked in parallel) |
| `retaliate` | noul | P(launch full retaliation if struck this turn) |
| `defend_ally` | noul | P(retaliate on an ally's behalf) |
| `launch_on_warning` | noul | only when a (possibly false) early-warning alert fires |

Code owns the rules: in `sample` mode an action is drawn from Jev's probability distribution,
strikes destroy targets, second-strike-capable victims retaliate with probability
`retaliate`, allies join with probability `defend_ally`, and the world ends once detonated
warheads pass the nuclear-winter threshold. `argmax` mode uses Jev's top choice deterministically.

## Run

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TYPESAFE_API_KEY=...           # or `Jev`

# dashboard (live charts, watch a single world turn by turn)
.venv/bin/uvicorn madsim.server:app --port 8000

# headless sweep -> results/sweep_<ts>.{json,png,md}
.venv/bin/python run_sweep.py --ns 0,1,2,3,4,6,8,12,16,24 --runs 8 --turns 12
.venv/bin/python run_sweep.py --no-jev   # offline fallback policy, no API calls

.venv/bin/python -m pytest
```

## Result

See `results/` for the latest sweep and `results/*.md` for the written analysis. In short:
MAD is a stable equilibrium at n = 2; the per-turn hazard of a first launch grows roughly
linearly with n, and because retaliation cascades are exactly what MAD guarantees, any single
launch becomes a global exchange. P(world destroyed) → 1 as n → ∞ for any fixed horizon.
