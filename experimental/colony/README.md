# Colony study add-on (Phase 2)

This add-on lives in the existing ANTS-RL repository and imports its seeded
maze generator (`ants_rl.maze`). It does not replace the original one-ant
environment or alter its results. It is an initial, CPU-scale study of
shared-pheromone learning with one shared PPO policy selecting actions for a
fixed-size colony.

## Run

From the repository root, with the dependencies in `requirements.txt`:

```bash
python scripts/run_colony.py --config configs/experiments/colony.yaml --limit 2
python scripts/run_colony.py --config configs/experiments/colony.yaml
```

The first command runs a small slice of the grid. Results and one model per
condition/seed go to `runs/colony/`; completed runs are skipped on rerun.
Set `--runs` to choose another output directory. Increase `total_timesteps`
and `eval_episodes` in the YAML before treating results as paper evidence.

## Design

Each ant gets a local 7x7 view with wall, food, nest, outbound trail,
return-trail, and carrying-state channels. Actions are simultaneous. Food
pickup changes the ant's target to the nest; delivery returns it to outbound
foraging. Team reward is one per delivery plus a team-level geodesic
potential-based shaping term (mean progress toward each ant's current target).
This reuses the distance information used by ANTS-RL's PBRS preset to make a
sparse-reward colony learnable; a matching step cost prevents discounting from
rewarding idle ants. The episode step limit is treated as truncation, so PBRS
does not create a terminal bonus at the time limit. Both terms are held fixed
across the pheromone ablation.
Trails evaporate and diffuse only through adjacent free cells. The two trail
channels record outbound and return traffic. Pheromone ablation leaves the same task, policy, and reward in
place, but removes trail input and deposition.

The initial comparison varies maze width (11/21), colony size (1/4), maze
topology (perfect/braided at 0.5), and pheromones (off/on): 48 runs across
three seeds. This topology factor matters because perfect mazes admit simple
complete wall-following, while loops remove that guarantee. It reports
deliveries per fixed-length episode,
probability of at least one delivery, and pheromone mass. Keep train/evaluation
seeds disjoint when extending the protocol, and summarize variation across
seeds rather than episodes alone. Increase the number of seeds before
interpreting small differences. Colony sizes 8/16 are a later extension once
the pilot learns reliably.

## Interpretation limits

This is a centralized joint-action PPO baseline with local per-ant
observations, not a biological model or a decentralized multi-agent algorithm.
The fixed-size joint action space changes with colony size. In particular,
the 1-ant pheromone condition tests within-episode trail feedback, while
larger colonies test shared feedback. The current default is a pilot and the
metrics are not directly comparable to Phase-1 success/SPL.
