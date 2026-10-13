#!/usr/bin/env python
"""Train/evaluate the optional shared-pheromone colony study.

Run from the repository root:
    python scripts/run_colony.py --config configs/experiments/colony.yaml
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ants_rl.colony import ColonyMazeEnv


def evaluate(model, kwargs, seed, episodes, deterministic=True):
    rows = []
    env = ColonyMazeEnv(**kwargs)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rows.append((info["deliveries"], total, info["pheromone_mass"]))
    env.close()
    a = np.asarray(rows, dtype=float)
    return {"mean_deliveries": float(a[:, 0].mean()),
            "delivery_sd": float(a[:, 0].std(ddof=1)) if len(a) > 1 else 0.0,
            "episodes_with_delivery": float(np.mean(a[:, 0] > 0)),
            "mean_pheromone_mass": float(a[:, 2].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/colony.yaml")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N combinations (0 = all)")
    ap.add_argument("--runs", default="runs/colony")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    grid = cfg["grid"]
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if args.limit:
        combos = combos[:args.limit]
    out = Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    fields = ["width", "n_ants", "braid", "pheromones", "seed", "timesteps", "mean_deliveries",
              "delivery_sd", "episodes_with_delivery", "mean_pheromone_mass"]
    append = summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not append:
            writer.writeheader()
        for spec in combos:
            for seed in cfg["seeds"]:
                kwargs = dict(cfg["env"], width=int(spec["width"]), height=int(spec["width"]),
                              n_ants=int(spec["n_ants"]), braid=float(spec["braid"]),
                              pheromones=bool(spec["pheromones"]))
                tag = f"w{spec['width']}_ants{spec['n_ants']}_b{spec['braid']}_pher{int(spec['pheromones'])}_s{seed}"
                run = out / tag
                done_file, model_file = run / "eval.yaml", run / "model.zip"
                if done_file.exists() and model_file.exists():
                    print(f"[skip] {tag}")
                    continue
                run.mkdir(parents=True, exist_ok=True)
                vec = make_vec_env(ColonyMazeEnv, n_envs=int(cfg["n_envs"]), seed=int(seed), env_kwargs=kwargs)
                model = PPO("MlpPolicy", vec, seed=int(seed), verbose=0,
                            learning_rate=float(cfg["learning_rate"]), n_steps=int(cfg["n_steps"]),
                            batch_size=int(cfg["batch_size"]), n_epochs=int(cfg["n_epochs"]),
                            gamma=float(cfg["gamma"]), ent_coef=float(cfg["ent_coef"]))
                model.learn(total_timesteps=int(cfg["total_timesteps"]))
                model.save(model_file)
                vec.close()
                metrics = evaluate(model, kwargs, int(seed) + 2_000_000, int(cfg["eval_episodes"]))
                writer.writerow({**spec, "seed": seed, "timesteps": cfg["total_timesteps"], **metrics})
                f.flush()
                done_file.write_text(yaml.safe_dump({"environment": kwargs, "seed": seed,
                                                      "timesteps": cfg["total_timesteps"], "metrics": metrics}, sort_keys=False))
                print(f"[done] {tag}: {metrics}")


if __name__ == "__main__":
    main()
