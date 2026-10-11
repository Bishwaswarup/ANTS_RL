"""End-to-end: tiny training run for every memory type, then evaluation."""

import json

import pytest

from ants_rl.config import load_config, parse_overrides
from ants_rl.training import load_model, train
from ants_rl.evaluation import evaluate_model, heldout_episodes


@pytest.mark.parametrize("memory", ["none", "stack", "lstm"])
def test_train_and_evaluate(tmp_path, memory):
    cfg = load_config(overrides=[
        f"agent.memory={memory}", "ppo.total_timesteps=1024", "ppo.n_envs=2",
        "ppo.n_steps=128", "ppo.batch_size=64", "ppo.n_epochs=1",
        "eval.n_episodes=6", "eval.n_train_episodes=4", "eval.curve_every=512",
        "eval.curve_episodes=4", "eval.transfer_sizes=[13]", "env.n_train_mazes=4",
    ])
    summary = train(cfg, tmp_path)
    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "curves.csv").read_text().count("\n") >= 3
    data = json.loads((tmp_path / "eval.json").read_text())
    for mode in ("deterministic", "stochastic"):
        assert 0.0 <= data["heldout"][mode]["success_rate"] <= 1.0
        assert data["heldout"][mode]["n_episodes"] == 6
    assert "13" in data["transfer"]
    model = load_model(tmp_path, cfg)
    res = evaluate_model(model, cfg["env"], heldout_episodes(3), memory=memory, n_envs=2)
    assert len(res) == 3 and all(r is not None for r in res)


def test_overrides():
    o = parse_overrides(["ppo.total_timesteps=2e5", "agent.memory=lstm", "eval.modes=[stochastic]"])
    assert o == {"ppo": {"total_timesteps": 200000}, "agent": {"memory": "lstm"},
                 "eval": {"modes": ["stochastic"]}}
