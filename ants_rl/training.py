"""Model construction, training loop and the standard evaluation protocol."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .env import AntMazeEnv
from .evaluation import (evaluate_model, heldout_episodes, save_results, summarise,
                         train_episodes)
from .policy import make_policy_kwargs

PPO_KEYS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "device")


def env_kwargs(cfg: dict) -> dict:
    return dict(cfg["env"])


def heldout_for(env_cfg: dict, n: int):
    return heldout_episodes(n, width=env_cfg.get("width", 11), height=env_cfg.get("height", 11),
                            algorithm=env_cfg.get("maze_algorithm", "dfs"),
                            braid=env_cfg.get("braid", 0.0))


def build_vec_env(cfg: dict, seed: int):
    a, p = cfg["agent"], cfg["ppo"]
    venv = make_vec_env(
        AntMazeEnv, n_envs=p["n_envs"], seed=seed, vec_env_cls=DummyVecEnv,
        env_kwargs=dict(gamma=p["gamma"], **env_kwargs(cfg)),
    )
    if a["memory"] == "stack":
        venv = VecFrameStack(venv, n_stack=a["n_stack"], channels_order="first")
    return venv


def build_model(cfg: dict, venv, seed: int, log_dir: str | None = None):
    a, p = cfg["agent"], cfg["ppo"]
    kwargs = {k: p[k] for k in PPO_KEYS if k in p}
    pk = make_policy_kwargs(a["extractor"], a["features_dim"])
    if a["memory"] == "lstm":
        from sb3_contrib import RecurrentPPO
        pk.update(lstm_hidden_size=a["lstm_hidden"], n_lstm_layers=1)
        return RecurrentPPO("MlpLstmPolicy", venv, policy_kwargs=pk, seed=seed,
                            verbose=0, **kwargs)
    if a["memory"] not in ("none", "stack"):
        raise ValueError(f"Unknown memory type {a['memory']!r}")
    return PPO("MlpPolicy", venv, policy_kwargs=pk, seed=seed, verbose=0, **kwargs)


def load_model(run_dir: str | Path, cfg: dict):
    path = Path(run_dir) / "model.zip"
    if cfg["agent"]["memory"] == "lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device="cpu")
    return PPO.load(path, device="cpu")


def _eval(model, cfg, env_cfg, episodes, mode):
    a = cfg["agent"]
    return evaluate_model(model, env_cfg, episodes, memory=a["memory"], n_stack=a["n_stack"],
                          deterministic=(mode == "deterministic"), gamma=cfg["ppo"]["gamma"])


class LearningCurveCallback(BaseCallback):
    """Evaluates on held-out layouts every ``every`` env steps -> curves.csv.
    Used for plotting only; the *final* model is reported (no checkpoint
    selection on held-out data)."""

    def __init__(self, cfg: dict, path: Path, every: int, n_episodes: int):
        super().__init__()
        self.cfg, self.path, self.every = cfg, path, every
        self.episodes = heldout_for(cfg["env"], n_episodes)
        self.next_at = 0
        self.last_eval = -1
        self.t0 = time.time()

    def _evaluate(self):
        row = {"timesteps": self.num_timesteps, "wall_time": time.time() - self.t0}
        for mode in self.cfg["eval"]["modes"]:
            s = summarise(_eval(self.model, self.cfg, env_kwargs(self.cfg), self.episodes, mode))
            row[f"{mode}_success"] = s["success_rate"]
            row[f"{mode}_spl"] = s["spl"]
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        self.last_eval = self.num_timesteps
        while self.next_at <= self.num_timesteps:
            self.next_at += self.every

    def _on_training_start(self):
        if self.every > 0:
            self._evaluate()

    def _on_step(self) -> bool:
        if self.every > 0 and self.num_timesteps >= self.next_at:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.every > 0 and self.num_timesteps != self.last_eval:
            self._evaluate()


def full_evaluation(model, cfg: dict) -> tuple[dict, dict]:
    """Standard protocol: held-out layouts, training distribution, and larger
    mazes; each in every mode in ``cfg['eval']['modes']``."""
    e = cfg["eval"]
    summary: dict = {"heldout": {}, "train_dist": {}, "transfer": {}}
    episodes: dict = {}
    sets = [("heldout", env_kwargs(cfg), heldout_for(cfg["env"], e["n_episodes"])),
            ("train_dist", env_kwargs(cfg), train_episodes(cfg["env"], e["n_train_episodes"]))]
    for size in e.get("transfer_sizes") or []:
        ecfg = dict(env_kwargs(cfg), width=size, height=size, max_steps=None, n_train_mazes=0)
        sets.append((f"transfer_{size}", ecfg, heldout_for(ecfg, e["n_episodes"])))
    for name, ecfg, eps in sets:
        for mode in e["modes"]:
            res = _eval(model, cfg, ecfg, eps, mode)
            s = summarise(res)
            if name.startswith("transfer_"):
                summary["transfer"].setdefault(name.split("_")[1], {})[mode] = s
            else:
                summary[name][mode] = s
            episodes[f"{name}/{mode}"] = {
                "success": np.array([r.success for r in res]),
                "steps": np.array([r.steps for r in res]),
                "spl": np.array([r.spl for r in res]),
                "optimal_steps": np.array([r.optimal_steps for r in res]),
            }
    return summary, episodes


def save_episode_arrays(path: Path, episodes: dict) -> None:
    flat = {f"{k}/{field}": v for k, d in episodes.items() for field, v in d.items()}
    np.savez_compressed(path, **flat)


def train(cfg: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = int(cfg["seed"])
    set_random_seed(seed)

    venv = build_vec_env(cfg, seed)
    model = build_model(cfg, venv, seed)
    fmts = ["csv"] + (["tensorboard"] if cfg["output"].get("tensorboard") else [])
    model.set_logger(configure(str(run_dir), fmts))

    cb = LearningCurveCallback(cfg, run_dir / "curves.csv", cfg["eval"]["curve_every"],
                               cfg["eval"]["curve_episodes"])
    t0 = time.time()
    model.learn(total_timesteps=int(cfg["ppo"]["total_timesteps"]), callback=cb)
    train_time = time.time() - t0
    if cfg["output"].get("save_model", True):
        model.save(run_dir / "model.zip")

    summary, episodes = full_evaluation(model, cfg)
    summary["train_time_s"] = train_time
    summary["timesteps"] = int(model.num_timesteps)
    save_results(run_dir / "eval.json", summary)
    save_episode_arrays(run_dir / "episodes.npz", episodes)
    venv.close()
    return summary
