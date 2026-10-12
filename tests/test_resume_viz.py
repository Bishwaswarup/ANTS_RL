"""Checkpoint/resume with a time budget, subprocess envs, snapshots, and drawing."""

import json

import numpy as np

from ants_rl.baselines import BASELINES
from ants_rl.config import load_config
from ants_rl.evaluation import heldout_episodes
from ants_rl.training import load_model, train
from ants_rl.viz import draw_frame, run_ants

TINY = ["ppo.total_timesteps=1024", "ppo.n_envs=2", "ppo.n_steps=128", "ppo.batch_size=64",
        "ppo.n_epochs=1", "eval.n_episodes=4", "eval.n_train_episodes=2", "eval.curve_every=512",
        "eval.curve_episodes=2", "eval.transfer_sizes=[]", "output.checkpoint_every=256",
        "output.snapshots=[0,512]"]


def test_time_budget_pauses_then_resumes(tmp_path):
    cfg = load_config(overrides=TINY)
    assert train(cfg, tmp_path, max_hours=1e-9) is None          # stops at once
    assert (tmp_path / "checkpoint.zip").exists()
    assert not (tmp_path / "eval.json").exists()
    summary = train(cfg, tmp_path)                                # continues
    assert summary is not None and summary["sessions"] == 2
    assert summary["timesteps"] >= 1024
    assert not (tmp_path / "checkpoint.zip").exists()
    steps = [int(float(l.split(",")[0])) for l in (tmp_path / "curves.csv").read_text().splitlines()[1:]]
    assert steps == sorted(steps)                                 # curve stays monotone
    assert json.loads((tmp_path / "eval.json").read_text())["sessions"] == 2


def test_subproc_envs_and_snapshots(tmp_path):
    cfg = load_config(overrides=TINY + ["ppo.vec_env=subproc"])
    train(cfg, tmp_path)
    snaps = sorted(p.name for p in (tmp_path / "snapshots").glob("step_*.zip"))
    assert snaps[0] == "step_00000000.zip" and len(snaps) == 2
    model = load_model(tmp_path, cfg, path=tmp_path / "snapshots" / snaps[1])
    spec = heldout_episodes(1)[0]
    ro = run_ants(dict(cfg["env"]), spec, n_ants=3, model=model, cfg=cfg)
    assert len(ro.paths) == 3 and all(p[0] == ro.nest for p in ro.paths)
    img = draw_frame(ro, ro.length - 1, cell=10)
    assert img.shape == (cfg["env"]["height"] * 10, cfg["env"]["width"] * 10, 3)


def test_baseline_swarm_wall_follower_all_succeed():
    spec = heldout_episodes(1)[0]
    ro = run_ants({}, spec, n_ants=4, baseline_cls=BASELINES["wall_follower"])
    assert all(ro.success)
    assert np.all(draw_frame(ro, 0).shape[:2])
