"""Model construction, training loop and the standard evaluation protocol."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

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
    """``ppo.vec_env``: ``dummy`` steps all mazes in this process (best when
    several runs share the machine, e.g. ``--jobs 4``); ``subproc`` gives each
    maze its own process (best for one run at a time on a multi-core machine)."""
    a, p = cfg["agent"], cfg["ppo"]
    kind = p.get("vec_env", "dummy")
    if kind not in ("dummy", "subproc"):
        raise ValueError(f"ppo.vec_env must be dummy or subproc, got {kind!r}")
    venv = make_vec_env(
        AntMazeEnv, n_envs=p["n_envs"], seed=seed,
        vec_env_cls=SubprocVecEnv if kind == "subproc" else DummyVecEnv,
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


def _algo(cfg: dict):
    if cfg["agent"]["memory"] == "lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    return PPO


def load_model(run_dir: str | Path, cfg: dict, path: str | Path | None = None, env=None):
    """Load ``model.zip`` of a run (or any checkpoint/snapshot ``path``)."""
    path = Path(path) if path is not None else Path(run_dir) / "model.zip"
    return _algo(cfg).load(path, env=env, device="cpu")


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
        if path.exists():  # resuming: continue the existing curve
            rows = list(csv.DictReader(open(path)))
            if rows:
                self.last_eval = int(float(rows[-1]["timesteps"]))
                self.next_at = self.last_eval + every

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
        if self.every > 0 and self.num_timesteps != self.last_eval and self.num_timesteps >= self.next_at:
            self._evaluate()

    def _on_step(self) -> bool:
        if self.every > 0 and self.num_timesteps >= self.next_at:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.every > 0 and self.num_timesteps != self.last_eval:
            self._evaluate()


class SnapshotCallback(BaseCallback):
    """Saves the policy at chosen training steps -> snapshots/step_XXXXXXXX.zip,
    so behaviour can be replayed at different stages of learning
    (scripts/render_progress.py)."""

    def __init__(self, folder: Path, steps):
        super().__init__()
        self.folder = folder
        self.todo = sorted(int(x) for x in (steps or []))

    def _save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.model.save(self.folder / f"step_{self.num_timesteps:08d}.zip")

    def _on_training_start(self):
        self.todo = [s for s in self.todo if s > self.num_timesteps or s == 0 and self.num_timesteps == 0]
        if self.todo and self.todo[0] == 0:
            self._save()
            self.todo.pop(0)

    def _on_step(self) -> bool:
        if self.todo and self.num_timesteps >= self.todo[0]:
            self._save()
            while self.todo and self.todo[0] <= self.num_timesteps:
                self.todo.pop(0)
        return True


class CheckpointAndBudgetCallback(BaseCallback):
    """Saves ``checkpoint.zip`` every ``every`` steps and stops training
    (after saving) once ``deadline`` (a ``time.time()`` value) has passed."""

    def __init__(self, path: Path, every: int, deadline: float | None):
        super().__init__()
        self.path, self.every, self.deadline = path, every, deadline
        self.next_at = None
        self.out_of_time = False

    def _on_training_start(self):
        self.next_at = self.num_timesteps + self.every if self.every > 0 else None

    def _on_step(self) -> bool:
        if self.next_at is not None and self.num_timesteps >= self.next_at:
            self.model.save(self.path)
            self.next_at += self.every
        if self.deadline is not None and time.time() >= self.deadline:
            self.model.save(self.path)
            self.out_of_time = True
            return False
        return True


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


def _truncate_curves(path: Path, upto: int) -> None:
    """Drop learning-curve points recorded after the checkpoint (hard kill)."""
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    keep = [r for r in rows if int(float(r["timesteps"])) <= upto]
    if len(keep) != len(rows) and rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)


def train(cfg: dict, run_dir: str | Path, max_hours: float | None = None) -> dict | None:
    """Train (or resume) one run, then evaluate.

    If ``<run_dir>/checkpoint.zip`` exists the run resumes from it. With
    ``max_hours`` training stops once the budget is used up, after saving a
    checkpoint; the function then returns ``None`` and the same call later
    continues where it stopped. A resumed run is equivalent in budget but not
    bit-identical to an uninterrupted one (environment RNG states restart).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = cfg["output"]
    total = int(cfg["ppo"]["total_timesteps"])
    ckpt = run_dir / "checkpoint.zip"
    state_path = run_dir / "train_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sessions": 0, "train_time_s": 0.0}
    seed = int(cfg["seed"])
    set_random_seed(seed + 1000 * state["sessions"])
    venv = build_vec_env(cfg, seed + 1000 * state["sessions"])

    if ckpt.exists():
        model = load_model(run_dir, cfg, path=ckpt, env=venv)
        prev = run_dir / "progress.csv"
        if prev.exists():
            prev.rename(run_dir / f"progress_part{state['sessions']}.csv")
        _truncate_curves(run_dir / "curves.csv", int(model.num_timesteps))
        # Record every resume immediately, so a run that was hard-killed
        # (Ctrl-C, sleep, Kaggle timeout) is still marked as resumed.
        state.setdefault("resumed_at", []).append(int(model.num_timesteps))
        state_path.write_text(json.dumps(state, indent=1))
        print(f"[train] resuming from {ckpt} at {model.num_timesteps:,} / {total:,} steps")
    else:
        model = build_model(cfg, venv, seed)
    fmts = ["csv"] + (["tensorboard"] if out.get("tensorboard") else [])
    model.set_logger(configure(str(run_dir), fmts))

    deadline = time.time() + 3600 * max_hours if max_hours else None
    budget_cb = CheckpointAndBudgetCallback(ckpt, int(out.get("checkpoint_every", 0) or 0), deadline)
    cbs = [LearningCurveCallback(cfg, run_dir / "curves.csv", cfg["eval"]["curve_every"],
                                 cfg["eval"]["curve_episodes"]),
           SnapshotCallback(run_dir / "snapshots", out.get("snapshots")),
           budget_cb]
    remaining = total - int(model.num_timesteps)
    t0 = time.time()
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cbs, reset_num_timesteps=False)
    state["sessions"] += 1
    state["train_time_s"] += time.time() - t0
    state["timesteps"] = int(model.num_timesteps)
    state_path.write_text(json.dumps(state, indent=1))
    if budget_cb.out_of_time and model.num_timesteps < total:
        venv.close()
        print(f"[train] time budget reached at {model.num_timesteps:,} / {total:,} steps; "
              f"checkpoint saved, re-run the same command to continue")
        return None

    if out.get("save_model", True):
        model.save(run_dir / "model.zip")
    summary, episodes = full_evaluation(model, cfg)
    summary["train_time_s"] = state["train_time_s"]
    summary["timesteps"] = int(model.num_timesteps)
    summary["sessions"] = state["sessions"]
    summary["resumed_at"] = state.get("resumed_at", [])  # empty = uninterrupted, bit-reproducible run
    save_results(run_dir / "eval.json", summary)
    save_episode_arrays(run_dir / "episodes.npz", episodes)
    if ckpt.exists():
        ckpt.unlink()
    venv.close()
    return summary
