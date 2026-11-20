# Era Basketball — Python Simulation Engine, v0.1 prototype

**STATUS: RUNNABLE RESEARCH PROTOTYPE, NOT A COMPLETE OR VALIDATED GAME ENGINE.**

This source package implements a small, dependency-free Python/SQLite basketball statistics prototype. It includes a single permanent ID for each fictional player, era-specific settings, possession outcomes, 10-game simulated averages, optional Monte Carlo 82-game scoring estimates, event records for future animated highlights, and automated consistency tests.

## Run (Python 3.10+; no packages required)

Open Terminal in this folder:

```bash
python3 -m era_engine.cli init era.db
python3 -m era_engine.cli simulate era.db --era 1995-96 --games 10 --seed 7 --output results.json
python3 -m era_engine.cli project era.db --era 1995-96 --runs 20 --output projection.json
python3 -m unittest discover -s tests -v
```

`results.json` contains every game's player stats, averages, and a sample of replay-worthy event metadata. `projection.json` contains Monte Carlo variability bounds **under prototype assumptions**. It is NOT a historically calibrated confidence interval or a genuine medical risk forecast.

## Formula details

* **Possessions:** `round(Normal(era.pace, 4))` per team, clamped to a minimum; this is a simplified approximation, not a clock-based simulator.
* **Shot zone selection:** 60–75% of propensity weight initially from the player's own distribution; adaptation moves toward an era distribution, with coaching style modifiers.
* **Make probability:** zone baseline + shooting skill contribution − defender contest − physicality penalty + situational openness − fatigue + adaptation + season geometry/spacing proxy; probability clamped from 5% to 88%.
* **Turnovers:** stochastic chance influenced by defensive steals, offensive IQ and fatigue; some credited steals.
* **Assists:** a teammate pass plus a made basket and a conditional assist-credit event.
* **Rebounds:** one credited rebound after a miss; offensive or defensive chosen probabilistically. **Follow-up possessions from offensive rebounds are not yet modeled.**
* **Fouls:** simplified shooting fouls with 2 or 3 free throws; NBA bonus and multiple kinds of fouls not yet implemented.
* **Statistics:** individual per-game counts summed over simulated games; percentages computed from total makes / total attempts, not mean of game percentages.
* **Season projection:** repeated 82-game statistical simulations, returning the 5th/95th percentile of run-level scoring averages, not yet a calibrated predictive interval.

## Data design

SQLite tables: `players`, `nba_research_links`, `eras`, `teams`, `simulations`. NBA reference IDs are **intentionally separate** from fictional player IDs; the research mapping should remain server-side in a rights-cleared deployment. The demo database has ten entirely fictional players, two five-player lineups, five parameterized seasons, no real player mapping, and **unverified illustrative** period parameters.

## What remains before this can be production-ready

1. **NBA data use/licensing:** `nba_api` is a third-party client, not a grant of data reuse rights. Obtain proper NBA statistical permissions and player-identity rights where needed. No official NBA data is shipped in this package.
2. **Historical calibration:** Acquire authorized player-season, shooting distance and team datasets. Research missing pre-tracking era data and report model assumptions.
3. **Full possession state machine:** live offensive rebounds, shot clock, clock progression, detailed screens/drives/passes, lineup substitutions, bench minutes, team foul bonus and overtime.
4. **Era-specific referee logic:** validated rules-by-season table; season-specific court dimensions, fouls, violations, enforcement. Current `Era` settings are only placeholders.
5. **Advanced defense:** contest quality and expected points prevented vs rigorous baseline; separate defensive play events; current defense/steal/block probabilities are placeholders.
6. **Workload/injuries:** `risk_injury` is a demonstrative uncalibrated function and does not yet remove players from games or seasons; add medical clearance, rest, reinjury, projections for games played.
7. **Player intelligence and coaching:** strategy adaptation, lineups, tactics, familiarity/chemistry, team-specific coaches, AI-versus-human decision logic.
8. **Statistical rigor:** back-testing and holdouts across decades, calibrated prediction intervals, feature importance, diagnostics for bias, confidence levels only after historical evaluation.
9. **3D event renderer and Apple device integration:** no animation/video generation or iPhone/iPad interface is included.
10. **Scaling:** optimized batch simulations, versioned immutable player/era profiles, server-side job queue, automated QA and rollback controls.

## Known limitations to avoid misinterpretation

This first release is an **executable technical foundation**, not the entire multi-year development scope. Demo attributes, team names and era pace values are only illustrations. Its results should **not** be advertised as realistic predictions yet. It deliberately does not link fictional names to living players, automatically obtain NBA data, or bypass data licenses. The engine calculates game events, not video. Health/medical numbers must not be interpreted as genuine injury predictions.

## Optional NBA API historical-data import (research only)

If permitted to retrieve and reuse NBA.com statistics for your application, install the community-maintained `nba_api` library:

```bash
python3 -m pip install nba_api pandas requests
python3 -m era_engine.nba_import era.db 201939
```

This imports player-season raw records for research into `nba_season_research`; it does **not** replace fictional profiles, publish a real-to-fictional search alias, or automatically establish calibrated skill ratings. It is untested against live NBA.com endpoints in this offline environment. API availability, rate limits, response fields, and use permissions can change; the tool intentionally reports failures instead of inventing data. Avoid shipping the research mapping or source data to clients without authorization.
