# 🐜 ANTS — Emergent Ant Colony Simulation

> A reinforcement learning research project exploring **emergent navigation, memory, and collective intelligence** in procedurally generated mazes.  
> Built with Python · PyTorch · Stable-Baselines3 · Gymnasium · Pygame

---

## Overview

ANTS is a multi-phase RL research project modelled after the foraging behaviour of real ant colonies. Ants navigate mazes with only **partial local visibility**, must remember where they've been, and — in later phases — coordinate through learned pheromone-like communication.

The project is structured as an evolving research pipeline, progressing from a single-agent LSTM baseline toward a full multi-agent system with emergent collective intelligence.

```
Phase 1 (complete)  →  Phase 2 (in progress)  →  Phase 3 (roadmap)
Single-agent LSTM      Multi-agent MAPPO +         Emergent specialisation
Maze foraging          pheromone trails            & colony-level tasks
```

---

## Demo

| Pygame Visualisation (Phase 1) | Visitation Heatmap |
|---|---|
| Agent (🔴) navigates an 11×11 maze from Nest (🟫) to Food (🟢) | Warmer = more visits; reveals the agent's exploration strategy |

*Run `python evaluate.py` to launch the live visualisation.*

---

## Key Features

**Environment**
- Procedurally generated mazes using Randomised Depth-First Search — every episode is a unique puzzle
- 3-channel local observation window (walls / food / nest) — the agent never sees the full maze
- Configurable size, observation radius, and food distance via `configs/default.yaml`

**Agent & Training (Phase 1)**
- Recurrent PPO with an LSTM core — maintains memory across time steps to handle partial observability
- Custom CNN feature extractor (`agents/policy.py`) — preserves spatial structure of the grid observation instead of naively flattening it
- Modular, composable reward shaping (`training/reward_shaping.py`):
  - Step penalty — time pressure
  - Potential-based distance shaping — guides without cheating
  - Revisitation penalty — breaks oscillation loops
  - Exploration bonus (optional) — count-based novelty signal

**Analysis & Interpretability**
- Per-episode metrics: success rate, path efficiency, unique cells visited, wall hit rate
- Visitation heatmaps — reveal which parts of the maze the agent explores most
- Linear probing framework (`analysis/message_probe.py`) — quantifies what agent representations encode

**Phase 2 Architecture (scaffolded)**
- Pheromone diffusion grid with evaporation and diffusion (`envs/pheromone.py`) — physics-based stigmergic communication
- Inter-agent message passing channel with configurable aggregation (`agents/comms.py`)
- Spatial and episodic working memory (`agents/memory.py`) — explicit visit maps + curriculum controller
- MAPPO training loop skeleton (`training/mappo.py`)

---

## Project Structure

```
ANTS/
├── envs/
│   ├── maze_env.py          # Custom Gymnasium environment (Phase 1)
│   ├── maze_generator.py    # Randomised DFS maze generation
│   ├── pheromone.py         # Pheromone diffusion physics (Phase 2)
│   └── tasks/               # Task definitions for multi-agent scenarios
│
├── agents/
│   ├── policy.py            # CNN + attention feature extractors (SB3-compatible)
│   ├── memory.py            # Spatial working memory + episodic curriculum memory
│   └── comms.py             # Message channel, comm buffer, centralised critic aggregator
│
├── training/
│   ├── reward_shaping.py    # Composable reward shaping components
│   ├── rollout_buffer.py    # Episode statistics tracker for custom training loops
│   └── mappo.py             # Multi-Agent PPO (Phase 2)
│
├── analysis/
│   ├── metrics.py           # Episode metrics collection and JSON serialisation
│   ├── visualize.py         # Heatmaps, training curves, path overlays (matplotlib)
│   └── message_probe.py     # Linear probing for learned representations
│
├── configs/
│   └── default.yaml         # All hyperparameters — one file to rule them all
│
├── train.py                 # Phase 1 training entrypoint
├── evaluate.py              # Pygame live visualisation + evaluation
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

# 2. Train Phase 1 (LSTM baseline)
python train.py
# → Saves: ppo_ant_lstm_baseline.zip
# → Logs:  ppo_ant_tensorboard/

# 3. Watch the trained agent navigate
python evaluate.py

# 4. Monitor training in TensorBoard
tensorboard --logdir ppo_ant_tensorboard/
```

---

## Architecture Deep-Dive

### Observation Space

The agent receives a **3-channel 7×7 grid** centred on its current position:

```
Channel 0: Wall mask      ██ ██ ██
Channel 1: Food indicator      🟢
Channel 2: Nest indicator   🟫
```

This local window creates partial observability — the agent cannot see the full maze and must remember where it has been. The LSTM hidden state serves as implicit memory across timesteps.

### Reward Shaping

Reward shaping is handled by composable components in `training/reward_shaping.py`:

| Component | Formula | Purpose |
|---|---|---|
| `StepPenalty` | −0.01 / step | Encourage efficiency |
| `WallPenalty` | −0.10 on collision | Discourage wall-bumping |
| `DistanceShaper` | +0.15 × Δ(Manhattan dist) | Potential-based shaping |
| `RevisitationPenalty` | −0.03 × extra_visits | Break oscillation loops |
| `ExplorationBonus` | +0.05 / √visit_count | Optional novelty signal |

By the **potential-based shaping theorem** (Ng et al., 1999), the `DistanceShaper` preserves the optimal policy of the original MDP while significantly accelerating convergence.

### LSTM + CNN Policy

```
Observation (3, 7, 7)
        ↓
AntCNNExtractor
  Conv2d(3→16) + ReLU
  Conv2d(16→32) + ReLU
  AdaptiveAvgPool(3,3)
  Linear(288 → 64)
        ↓ (64,)
  LSTM hidden state
        ↓
  Policy head → action logits
  Value head  → V(s)
```

The CNN preserves the spatial geometry of the observation (which direction is the food?) before passing it to the LSTM, giving the agent a structured spatial inductive bias.

### Phase 2: Pheromone Communication

In Phase 2, multiple agents share a `PheromoneGrid` — a 2-channel diffusion grid with evaporation physics:

```
P[t+1] = (P[t] × (1 − ε) ⊛ kernel).clip(0, P_max)
```

Where ε is the evaporation rate and ⊛ is a convolution with a normalised diffusion kernel. Agents deposit pheromone at their position, and the gradient of the pheromone field guides other agents — the same mechanism used by real ants.

---

## Training Results (Phase 1)

| Metric | Value |
|---|---|
| Training timesteps | 80,000 |
| Final success rate | ~85% |
| Mean steps (successful) | ~120 |
| Maze size | 11 × 11 |
| Algorithm | RecurrentPPO (LSTM) |

*Results vary due to random maze generation. Run `python evaluate.py` to reproduce.*

---

## Roadmap

- [x] Phase 1: Single-agent LSTM baseline
- [x] Procedural maze generation (DFS)
- [x] Reward shaping framework
- [x] CNN feature extractor
- [x] Pheromone physics engine
- [x] Inter-agent communication channel
- [ ] Phase 2: Multi-agent environment with pheromone observation
- [ ] MAPPO training loop
- [ ] Curriculum learning (automatic difficulty scaling)
- [ ] Phase 3: Emergent role specialisation (scouts vs. foragers)
- [ ] Web-based visualisation dashboard

---

## Research Context

This project sits at the intersection of several active research areas:

- **Multi-agent reinforcement learning (MARL)** — specifically CTDE (Centralised Training, Decentralised Execution) via MAPPO
- **Emergent communication** — do agents develop interpretable protocols? Tested via linear probing in `analysis/message_probe.py`
- **Stigmergy** — indirect coordination through environmental modification (pheromones), as studied in swarm intelligence
- **Curriculum learning** — automatic difficulty scaling via `agents/memory.py`'s `EpisodicMemory`

---

## Dependencies

| Package | Purpose |
|---|---|
| `gymnasium` | RL environment API |
| `stable-baselines3` | PPO implementation |
| `sb3-contrib` | RecurrentPPO (LSTM) |
| `torch` | Neural network backend |
| `numpy` | Array operations |
| `scipy` | Pheromone convolution |
| `pygame` | Live visualisation |
| `matplotlib` | Analysis plots |
| `tensorboard` | Training monitoring |

---

## Author

**Priya** · OUSUMS Lab, Physics Department, IISc Bangalore  
*Exploring emergent collective behaviour through the lens of reinforcement learning.*

---

## License

MIT — free to use, modify, and build upon.
