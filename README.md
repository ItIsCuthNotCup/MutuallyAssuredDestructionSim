# Mutually Assured Destruction Simulator

A Monte Carlo simulation of `n` nuclear-armed nations where **Jev** (TypeSafe's System One
model) is the decision-maker for every leader. Sweep `n` from 0 upward to see whether MAD
holds as the number of nuclear states grows, under assumptions you can inspect and change.

## Calibration and the research it follows

Nine states have held nuclear weapons for decades with no use in war since 1945, so a model
that predicts frequent launches is wrong, not insightful. The engine therefore:

- gives every leader a **military briefing** before deciding: whether the rival's forces
  survive a first strike and how many warheads would come back. With survivable forces a
  first strike is never a best response (Schelling 1960; Intriligator & Brito 1984), and Jev
  sees exactly that;
- makes retaliation size depend on **force survivability** (`survivable_fraction`), not a
  boolean;
- treats **false alarms** as alerts that are cross-checked (`alarm_uncorroborated`), since
  every real alert in the record was vetoed before launch;
- lets crises **decay** and occur at a per-pair rate, so "more states" does not mechanically
  mean "a crisis every year";
- by default takes Jev's **most likely** action (`decision_mode=argmax`), so a 2% residual
  probability is not rolled as dice for every leader every year. The pessimist scenario
  samples instead, modelling erratic decision-making.

Three scenarios (`madsim/engine.py::SCENARIOS`) bracket the academic debate:

| scenario | camp | what it assumes |
|---|---|---|
| `historical` | fitted to 1945–today | ~1 serious false alarm per 100 state-years, 90% caught; ~1 crisis per rival pair per century; 90% of states have survivable forces |
| `optimist` | Waltz, *More May Be Better* | assured second strike everywhere, professional command and control |
| `pessimist` | Sagan, *More Will Be Worse* | vulnerable arsenals, fragile warning systems, erratic leaders, sampled decisions |

Any knob can be overridden from the dashboard or the API (`overrides`). The takeaway is
conditional: n only matters through the accident channel, and the scenario sets its size.

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

Code owns the rules: strikes destroy targets, second-strike-capable victims retaliate with
probability `retaliate` using the warheads that survived, allies join with probability
`defend_ally`, and the world ends once detonated warheads pass the nuclear-winter threshold.
`argmax` mode uses Jev's top choice; `sample` mode draws from its distribution.

## Run

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TYPESAFE_API_KEY=...           # or `Jev`

# dashboard (live charts, watch a single world turn by turn)
.venv/bin/uvicorn madsim.server:app --port 8000

# headless sweep -> results/sweep_<ts>.{json,png,md}
.venv/bin/python run_sweep.py --ns 0,1,2,3,4,6,8,12,16,24 --runs 8 --turns 25
.venv/bin/python run_sweep.py --scenario pessimist
.venv/bin/python run_sweep.py --no-jev   # offline fallback policy, no API calls

.venv/bin/python -m pytest
```

## Result

See `results/` for the latest sweep and `results/*.md` for the written analysis. In short:
under historically calibrated assumptions deterrence holds at every n tested, matching the
real record. Under pessimist assumptions each extra state adds an independent accident risk
and P(no launch in T years) ≈ (1 − p)^(nT) falls toward zero. Whether more nuclear states
means more danger is a question about p (force survivability, warning discipline, leader
restraint), not about n itself.
