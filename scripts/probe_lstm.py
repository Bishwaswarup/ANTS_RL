#!/usr/bin/env python
"""What did the LSTM learn? Linear probes on its hidden state.

For a trained RecurrentPPO run we roll the policy out on held-out episodes and
record the LSTM state at every step, together with ground-truth quantities the
agent never observes directly. For each quantity we fit a linear probe
(ridge regression / softmax regression) on some episodes and score it on
disjoint episodes.

Probed quantities
-----------------
heading       direction of the last successful move (4-way; what the wall follower stores)
last_action   last action taken (4-way)
visited_nbr   for each of the 4 neighbours: is it a free cell the ant has already visited?
nest_dx/dy    displacement from the nest (path integration)
n_unique      number of distinct cells visited so far
food_dir      direction of the food (4-way), only on steps where the food was seen
              earlier but is currently outside the view
food_dist     geodesic distance to the food, same steps

Controls (same trajectories, same probe)
----------------------------------------
obs        probe from the current observation alone (anything above this is memory)
random_h   hidden state of an *untrained* LSTM of the same size (rules out
           "random recurrent features already contain it")

Behavioural test
----------------
rhr_agreement: fraction of steps where the policy's action equals the
right-hand-rule action for the true heading. 25% is chance; 100% means the
policy *is* the wall follower.

Example
-------
    python scripts/probe_lstm.py --run runs/main/memory=lstm__n_train_mazes=0/seed* \
        --out results/probe.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ants_rl.baselines import WallFollowerPolicy  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.env import AntMazeEnv  # noqa: E402
from ants_rl.evaluation import heldout_episodes  # noqa: E402
from ants_rl.maze import ACTION_DELTAS, FREE  # noqa: E402
from ants_rl.training import build_model, build_vec_env, load_model  # noqa: E402

CLASS_TARGETS = ["heading", "last_action", "visited_up", "visited_right", "visited_down",
                 "visited_left", "food_dir"]
REG_TARGETS = ["nest_dx", "nest_dy", "n_unique", "food_dist"]


# ----------------------------------------------------------------- rollouts
def rhr_action(env, heading: int):
    for turn in (1, 0, -1, 2):
        a = (heading + turn) % 4
        if WallFollowerPolicy._free(env, a):
            return a
    return heading


def hidden_trace(model, obs_seq: np.ndarray) -> np.ndarray:
    """LSTM (h, c) after consuming each observation: array (T, 2H)."""
    state, out = None, []
    for t, o in enumerate(obs_seq):
        _, state = model.predict(o[None], state=state, episode_start=np.array([t == 0]),
                                 deterministic=True)
        out.append(np.concatenate([state[0][0, 0], state[1][0, 0]]))
    return np.array(out)


def collect(model, cfg, n_episodes: int, max_steps: int, deterministic: bool):
    env_cfg = dict(cfg["env"], n_train_mazes=0)
    env = AntMazeEnv(gamma=cfg["ppo"]["gamma"], **env_cfg)
    eps = heldout_episodes(n_episodes, width=env_cfg["width"], height=env_cfg["height"],
                           algorithm=env_cfg["maze_algorithm"], braid=env_cfg.get("braid", 0.0))
    rows, rhr_hits, rhr_n = [], 0, 0
    for ep, (ms, ps) in enumerate(eps):
        obs, _ = env.reset(options={"maze_seed": ms, "placement_seed": ps})
        state, first = None, True
        visited = {env.agent_pos}
        heading, last_action, food_seen = -1, -1, False
        obs_seq, hs, tgt = [], [], []
        for t in range(min(env.max_steps, max_steps)):
            action, state = model.predict(obs[None], state=state, episode_start=np.array([first]),
                                          deterministic=deterministic)
            first = False
            a = int(action[0])
            x, y = env.agent_pos
            food_in_view = bool(obs[1].any())
            food_seen = food_seen or food_in_view
            nbr_visited = [int(env.grid[x + dx, y + dy] == FREE and (x + dx, y + dy) in visited)
                           for dx, dy in ACTION_DELTAS]
            fx, fy = env.food_pos
            rel = (fx - x, fy - y)
            if abs(rel[0]) >= abs(rel[1]):
                fdir = 1 if rel[0] > 0 else 3
            else:
                fdir = 2 if rel[1] > 0 else 0
            hmask = food_seen and not food_in_view
            tgt.append(dict(
                heading=heading, last_action=last_action,
                visited_up=nbr_visited[0], visited_right=nbr_visited[1],
                visited_down=nbr_visited[2], visited_left=nbr_visited[3],
                food_dir=fdir if hmask else -1,
                nest_dx=x - env.nest_pos[0], nest_dy=y - env.nest_pos[1],
                n_unique=len(visited),
                food_dist=float(env.dist_to_food[x, y]) if hmask else np.nan,
                episode=ep,
            ))
            if heading >= 0:
                rhr_n += 1
                rhr_hits += int(a == rhr_action(env, heading))
            obs_seq.append(obs)
            hs.append(np.concatenate([state[0][0, 0], state[1][0, 0]]))
            before = env.agent_pos
            obs, _, term, trunc, _ = env.step(a)
            last_action = a
            if env.agent_pos != before:
                heading = a
                visited.add(env.agent_pos)
            if term or trunc:
                break
        for r, o, h in zip(tgt, obs_seq, hs):
            r["_obs"], r["_h"] = o.reshape(-1), h
        rows.extend(tgt)
    return rows, (rhr_hits / max(rhr_n, 1))


# ------------------------------------------------------------------- probes
def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_r2(Xtr, ytr, Xte, yte, lam=10.0):
    Xtr, Xte = _standardise(Xtr, Xte)
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    pred = np.c_[Xte, np.ones(len(Xte))] @ w
    ss_res, ss_tot = ((yte - pred) ** 2).sum(), ((yte - yte.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def softmax_acc(Xtr, ytr, Xte, yte, wd=1e-3, iters=100):
    """Multinomial logistic regression (LBFGS). Returns (accuracy, majority-class accuracy)."""
    classes = np.unique(ytr)
    if len(classes) < 2:
        return float("nan"), float("nan")
    Xtr, Xte = _standardise(Xtr, Xte)
    ci = {c: i for i, c in enumerate(classes)}
    ytr_i = torch.tensor([ci[c] for c in ytr])
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    lin = torch.nn.Linear(Xt.shape[1], len(classes))
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xt), ytr_i) + wd * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = lin(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    pred_c = classes[pred]
    maj = classes[np.bincount([ci[c] for c in ytr]).argmax()]
    return float((pred_c == yte).mean()), float((yte == maj).mean())


def run_probes(feats: dict[str, np.ndarray], rows: list[dict], train_frac=0.6) -> dict:
    ep = np.array([r["episode"] for r in rows])
    train = ep < int(ep.max() * train_frac)
    out: dict = {}
    for name, X in feats.items():
        out[name] = {}
        for t in CLASS_TARGETS:
            y = np.array([r[t] for r in rows])
            m = y >= 0
            tr, te = m & train, m & ~train
            if tr.sum() < 50 or te.sum() < 50:
                out[name][t] = None
                continue
            acc, maj = softmax_acc(X[tr], y[tr], X[te], y[te])
            out[name][t] = {"acc": acc, "majority": maj, "n_test": int(te.sum())}
        for t in REG_TARGETS:
            y = np.array([r[t] for r in rows], dtype=float)
            m = ~np.isnan(y)
            tr, te = m & train, m & ~train
            if tr.sum() < 50 or te.sum() < 50:
                out[name][t] = None
                continue
            out[name][t] = {"r2": ridge_r2(X[tr], y[tr], X[te], y[te]), "n_test": int(te.sum())}
    return out


def random_model_like(cfg, seed):
    c = json.loads(json.dumps(cfg))
    c["ppo"]["n_envs"] = 1
    return build_model(c, build_vec_env(c, seed), seed)


def probe_run(run: Path, n_episodes, max_steps, deterministic) -> dict:
    cfg = load_config(overrides=load_yaml(run / "config.yaml"))
    if cfg["agent"]["memory"] != "lstm":
        raise SystemExit(f"{run}: probes need an LSTM run (memory={cfg['agent']['memory']})")
    model = load_model(run, cfg)
    rows, rhr = collect(model, cfg, n_episodes, max_steps, deterministic)
    H = cfg["agent"]["lstm_hidden"]
    h = np.array([r["_h"] for r in rows])
    obs = np.array([r["_obs"] for r in rows])
    # Untrained LSTM of the same architecture, fed the *same* observations.
    rnd = random_model_like(cfg, int(cfg["seed"]) + 999)
    rh, i = [], 0
    for ep in sorted({r["episode"] for r in rows}):
        n = sum(1 for r in rows if r["episode"] == ep)
        rh.append(hidden_trace(rnd, obs.reshape(len(rows), *model.observation_space.shape)[i:i + n]))
        i += n
    rh = np.concatenate(rh)
    feats = {"lstm_h": h[:, :H], "lstm_hc": h, "random_h": rh[:, :H], "obs": obs}
    res = run_probes(feats, rows)
    return {"run": str(run), "seed": cfg["seed"], "lstm_hidden": H, "n_steps": len(rows),
            "rhr_agreement": rhr, "probes": res}


def print_table(r: dict):
    print(f"\n== {r['run']}  (H={r['lstm_hidden']}, {r['n_steps']} steps, "
          f"right-hand-rule agreement {r['rhr_agreement']:.2f}, chance 0.25)")
    srcs = list(r["probes"])
    print(f"{'target':14s}" + "".join(f"{s:>12s}" for s in srcs) + "   (acc | majority) or R2")
    for t in CLASS_TARGETS + REG_TARGETS:
        cells = []
        for s in srcs:
            v = r["probes"][s].get(t)
            if v is None:
                cells.append("n/a")
            elif "acc" in v:
                cells.append(f"{v['acc']:.2f}|{v['majority']:.2f}")
            else:
                cells.append(f"R2={v['r2']:.2f}")
        print(f"{t:14s}" + "".join(f"{c:>12s}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="LSTM run dirs or glob patterns")
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=150, help="cap per episode (stuck agents otherwise dominate)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default="results/probe.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    runs = sorted({Path(p) for pat in args.run for p in glob.glob(pat)})
    results = []
    for rd in runs:
        r = probe_run(rd, args.n_episodes, args.max_steps, args.deterministic)
        print_table(r)
        results.append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
