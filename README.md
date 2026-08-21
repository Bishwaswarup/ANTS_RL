# 🐜 ANTS — Emergent Ant Colony Simulation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-black?logo=openaigym&logoColor=white)](https://gymnasium.farama.org)
[![Stable Baselines3](https://img.shields.io/badge/SB3-2.0+-blue)](https://stable-baselines3.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e)]()
[![Phase](https://img.shields.io/badge/Phase-1%20Complete%20%7C%202%20In%20Progress-orange)]()
[![IISc](https://img.shields.io/badge/Lab-OUSUMS%20%40%20IISc-blue)]()

**A deep reinforcement learning research project exploring emergent navigation and collective intelligence in procedurally generated mazes.**

[Overview](#overview) · [Demo](#demo) · [Architecture](#architecture) · [Quickstart](#quickstart) · [Results](#results) · [Roadmap](#roadmap) · [Research Context](#research-context)

</div>

---

## Overview

ANTS (Emergent Ant Colony Simulation) models foraging behaviour of real ant colonies using deep RL. A single ant — equipped with only a **partial local view** of its environment — must navigate procedurally generated mazes, find food, and return to the nest. In later phases, multiple agents coordinate through learned pheromone-like communication.

The project is structured as a three-phase research pipeline:

```
Phase 1 ✅          Phase 2 🔨            Phase 3 📋
─────────────       ──────────────────    ──────────────────────
Single-agent        Multi-agent MAPPO     Emergent specialisation
CNN + Frame-Stack   + Pheromone trails    Scout / Forager roles
PPO baseline        + Learned comms       Colony-level objectives
```

> **Key result:** Trained agent successfully navigates 11×11 randomly generated mazes in as few as **16 steps** using only a 7×7 local observation window — demonstrating genuine spatial reasoning under partial observability.

---

## Demo

<table>
<tr>
<td align="center"><b>Live Evaluation (Pygame)</b></td>
<td align="center"><b>Visitation Heatmap</b></td>
</tr>
<tr>
<td>Agent 🔴 navigates from Nest 🟫 to Food 🟢. Pink trail shows visited cells.</td>
<td>Warmer colours = more visits. Reveals the agent's exploration strategy.</td>
</tr>
</table>

```bash
python evaluate.py               # watch the trained agent
python evaluate.py --episodes 5  # run 5 episodes back-to-back
```

---

## Architecture

### Observation Space

The agent receives a **3-channel 7×7 grid** centred on its current position. It never sees the full maze — only what's within 3 cells in each direction.

```
Channel 0 — Wall mask      ██ ██ ██ ██
Channel 1 — Food signal       🟢
Channel 2 — Nest signal          🟫
```

**Frame stacking** concatenates the last 4 observations as extra channels, giving the agent implicit motion memory without recurrence:

```
Single obs: (3, 7, 7)
Stacked:   (12, 7, 7)   ← 4 frames × 3 channels
```

### CNN Feature Extractor (`agents/policy.py`)

```
Input  (12, 7, 7)
  → Conv2d(12→16, k=3, pad=1) + ReLU     (16, 7, 7)
  → Conv2d(16→32, k=3, pad=1) + ReLU     (32, 7, 7)
  → AdaptiveAvgPool2d(3, 3)               (32, 3, 3)
  → Flatten + Linear(288→64) + ReLU      (64,)
  → PPO policy/value heads
```

Spatial structure is preserved throughout — the model knows *where* the food is relative to the agent, not just whether it exists.

### Reward Function

| Component | Formula | Purpose |
|---|---|---|
| Step penalty | −0.01 | Efficiency pressure |
| Wall collision | −0.10 | Discourage bumping |
| Distance shaping | +0.15 × Δ(Manhattan) | Potential-based guidance |
| Revisitation | −0.05 × min(visits−1, 5) | Break oscillation loops |
| Food reward | +10.0 (terminal) | Primary objective |

The distance shaping term satisfies the **potential-based shaping theorem** (Ng et al., 1999): adding any reward of the form `F(s,a,s') = γΦ(s') − Φ(s)` preserves the set of optimal policies of the original MDP. This guarantees that shaping accelerates learning without changing what the agent ultimately learns.

### Training Strategy

| Parameter | Value | Why |
|---|---|---|
| Algorithm | PPO | Stable, sample-efficient policy gradient |
| Memory | Frame stacking (N=4) | Robust vs. LSTM — no gradient-through-time issues |
| Parallel envs | 8 | 8× faster data collection |
| `n_steps` | 512 | Long enough rollouts to observe full corridors |
| `ent_coef` | 0.05 | Maintain exploration pressure |
| Total timesteps | 500k / run | Stacks across runs via checkpoint resumption |

---

## Project Structure

```
ANTS/
├── envs/
│   ├── maze_env.py          # Custom Gymnasium env — reward shaping, partial obs
│   ├── maze_generator.py    # Randomised DFS perfect maze generation
│   └── pheromone.py         # Pheromone diffusion physics (Phase 2)
│
├── agents/
│   ├── policy.py            # AntCNNExtractor + AntAttentionExtractor (SB3-compatible)
│   ├── memory.py            # Spatial working memory + episodic curriculum controller
│   └── comms.py             # Message channel, comm buffer, centralised critic aggregator
│
├── training/
│   ├── reward_shaping.py    # Composable reward shaper classes
│   ├── rollout_buffer.py    # Episode statistics tracker (path efficiency, wall hits)
│   └── mappo.py             # Multi-Agent PPO skeleton (Phase 2)
│
├── analysis/
│   ├── metrics.py           # Episode metrics collection + JSON serialisation
│   ├── visualize.py         # Heatmaps, training curves, path overlays (matplotlib)
│   └── message_probe.py     # Linear probing for learned agent representations
│
├── configs/
│   └── default.yaml         # All hyperparameters — single source of truth
│
├── train.py                 # Training entrypoint (auto-resumes from checkpoint)
├── evaluate.py              # Pygame live visualisation
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/your-username/ANTS.git
cd ANTS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (auto-resumes from checkpoint if one exists)
python train.py

# 3. Watch the trained agent
python evaluate.py

# 4. Run multiple episodes
python evaluate.py --episodes 10

# 5. Monitor training live
tensorboard --logdir ppo_ant_tensorboard/
```

---

## Results

### Phase 1 — Single Agent Baseline

| Metric | Value |
|---|---|
| Algorithm | PPO + Frame Stacking |
| Maze size | 11 × 11 (procedurally generated) |
| Observation | 7 × 7 local window (partial) |
| Best episode | **16 steps** to food |
| Reward at convergence | ~9.6 |
| Training steps | 500k (1 run) |

The agent demonstrates **genuine spatial reasoning**: navigating maze corridors, identifying junctions, and reaching the food source — all from a local view that covers less than 15% of the maze at any moment.

---

## Research Context

ANTS sits at the intersection of several active research areas:

**Multi-Agent Reinforcement Learning (MARL)** — Phase 2 uses CTDE (Centralised Training, Decentralised Execution) via MAPPO. Each agent acts on its own local observation at test time, but training uses a shared centralised critic.

**Emergent Communication** — Do agents develop interpretable messaging protocols? Tested via linear probing in `analysis/message_probe.py`. A logistic regression trained on agent messages can reveal whether they encode food direction, carrying state, etc.

**Stigmergy** — Coordination through environmental modification (pheromones), as studied in swarm intelligence. The `envs/pheromone.py` module implements physics-accurate diffusion: `P[t+1] = (P[t] × (1−ε)) ⊛ K` where ε is evaporation rate and K is a normalised diffusion kernel.

**Curriculum Learning** — `agents/memory.py` implements `EpisodicMemory` for automatic difficulty scaling: increase `min_food_dist` when the rolling success rate exceeds 75%, decrease it if performance drops below 25%.

**Potential-Based Reward Shaping** — The distance shaping component uses Manhattan distance as a potential function, satisfying the Ng et al. (1999) theorem that guarantees policy-invariance of the shaping. See `docs/theory.pdf` for the full mathematical treatment.

---

## Roadmap

- [x] Phase 1: Single-agent CNN + frame-stack PPO baseline
- [x] Procedural DFS maze generation
- [x] Composable reward shaping framework
- [x] Custom CNN feature extractor (SB3-compatible)
- [x] Pheromone diffusion physics engine
- [x] Inter-agent message passing channel
- [x] Spatial + episodic memory modules
- [x] Analysis + linear probing tools
- [ ] Phase 2: Multi-agent environment with pheromone observations
- [ ] MAPPO training loop
- [ ] Curriculum learning integration
- [ ] Phase 3: Emergent role specialisation (scouts vs. foragers)

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | ≥0.28 | RL environment API |
| `stable-baselines3` | ≥2.0 | PPO implementation |
| `sb3-contrib` | ≥2.0 | RecurrentPPO (LSTM alternative) |
| `torch` | ≥2.0 | Neural network backend |
| `numpy` | ≥1.24 | Array operations |
| `scipy` | any | Pheromone convolution |
| `pygame` | ≥2.5 | Live visualisation |
| `matplotlib` | any | Analysis plots |
| `tensorboard` | ≥2.13 | Training monitoring |

---

## Author

**Priya** — OUSUMS Lab, Department of Physics, IISc Bangalore
*Exploring emergent collective behaviour through the lens of reinforcement learning.*

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ants2025,
  author    = {Priya},
  title     = {ANTS: Emergent Ant Colony Simulation via Deep Reinforcement Learning},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/your-username/ANTS}
}
```

---

## License

MIT — free to use, modify, and build upon.
