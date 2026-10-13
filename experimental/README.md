# Experimental (Phase 2 prototypes — not used, not tested)

These modules were written for a planned multi-agent extension and are **not**
part of the Phase-1 study or its test suite:

| File | What it is |
|---|---|
| `pheromone.py` | Multi-channel pheromone grid with evaporation + diffusion (note: diffusion currently ignores walls) |
| `comms.py` | Message channel / buffer / aggregator for learned inter-agent communication |
| `memory.py` | Spatial visit memory and a success-rate-driven curriculum controller |
| `reward_shaping.py` | Composable reward shapers (superseded by `ants_rl/env.py` presets) |
| `message_probe.py` | Linear probes for analysing learned messages |

The runnable colony PPO pilot is now in [`colony/`](colony/README.md) with
its own integrated environment (`ants_rl.colony`) and config. The original
modules listed above remain prototypes and are not dependencies of that pilot.

Imports still refer to the old package layout and will need updating before use.
