"""
agents/comms.py
===============
Inter-agent communication module for EAC Phase 2 (multi-agent).

Background
----------
Real ant colonies communicate primarily through pheromones (handled in
envs/pheromone.py), but they also make direct body contact and produce
acoustic signals.  In multi-agent RL, learned communication allows agents
to share compact latent messages — richer than pheromones, but limited in
range.

This module provides:

  1. **MessageChannel** — a simple broadcast channel where agents write
     fixed-size messages and read the aggregated signal.  Think of it as
     a shared whiteboard with bounded capacity.

  2. **CommBuffer** — per-agent message history for diagnostics and probing.

  3. **CommAggregator** — combines N agent messages into a single context
     vector (mean / attention-weighted) for the centralised critic.

Phase 2 integration sketch
---------------------------
::

    channel = MessageChannel(n_agents=4, msg_dim=16)

    # Each agent writes its message (a learned vector from its policy head)
    for i, agent in enumerate(agents):
        msg = agent.policy.communicate(obs[i])  # shape (msg_dim,)
        channel.write(agent_id=i, message=msg)

    # Each agent reads the aggregated broadcast
    for i, agent in enumerate(agents):
        context = channel.read(agent_id=i)  # shape (msg_dim,)
        # Concatenate context to obs before action selection

    # Advance the channel (clears messages for next step)
    channel.step()

Note
----
This module contains no PyTorch — message tensors are plain NumPy arrays.
The learned communication head lives in agents/policy.py (future extension).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Message Channel
# ---------------------------------------------------------------------------

class MessageChannel:
    """
    Broadcast communication channel between agents.

    Each step, every agent can write one message vector.  All agents then
    read the aggregated signal (mean of all other agents' messages).

    Parameters
    ----------
    n_agents : int
        Number of agents sharing the channel.
    msg_dim  : int
        Dimensionality of each message vector.
    aggregation : str
        How to combine messages: "mean" | "sum" | "max"
    exclude_self : bool
        If True, each agent's read excludes its own write (default True).
        Prevents the agent from simply ignoring others.
    """

    AGGREGATIONS = {"mean", "sum", "max"}

    def __init__(
        self,
        n_agents: int,
        msg_dim: int = 16,
        aggregation: str = "mean",
        exclude_self: bool = True,
    ) -> None:
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.AGGREGATIONS}")

        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.aggregation = aggregation
        self.exclude_self = exclude_self

        # Shape: (n_agents, msg_dim)  — zeros = silence
        self._messages = np.zeros((n_agents, msg_dim), dtype=np.float32)
        self._written = np.zeros(n_agents, dtype=bool)

    def reset(self) -> None:
        """Clear all messages (call at episode start)."""
        self._messages[:] = 0.0
        self._written[:] = False

    def write(self, agent_id: int, message: np.ndarray) -> None:
        """
        Agent `agent_id` writes a message to the channel.

        Parameters
        ----------
        agent_id : int         Index of the writing agent.
        message  : np.ndarray  Shape (msg_dim,).
        """
        if message.shape != (self.msg_dim,):
            raise ValueError(
                f"Expected message shape ({self.msg_dim},), got {message.shape}"
            )
        self._messages[agent_id] = message.astype(np.float32)
        self._written[agent_id] = True

    def read(self, agent_id: Optional[int] = None) -> np.ndarray:
        """
        Read the aggregated message signal.

        Parameters
        ----------
        agent_id : int, optional
            If provided and exclude_self=True, this agent's own message is
            excluded from the aggregate.  Pass None to aggregate all.

        Returns
        -------
        np.ndarray  Shape (msg_dim,).
        """
        if agent_id is not None and self.exclude_self:
            mask = np.arange(self.n_agents) != agent_id
            msgs = self._messages[mask]
        else:
            msgs = self._messages

        if len(msgs) == 0:
            return np.zeros(self.msg_dim, dtype=np.float32)

        if self.aggregation == "mean":
            return msgs.mean(axis=0)
        elif self.aggregation == "sum":
            return msgs.sum(axis=0)
        elif self.aggregation == "max":
            return msgs.max(axis=0)

    def read_all(self) -> np.ndarray:
        """
        Return the full message matrix — shape (n_agents, msg_dim).

        Used by the centralised critic to condition on all agent messages.
        """
        return self._messages.copy()

    def step(self) -> None:
        """Advance one timestep — clears the channel for the next write."""
        self._messages[:] = 0.0
        self._written[:] = False

    @property
    def n_active(self) -> int:
        """Number of agents that have written this step."""
        return int(self._written.sum())

    def __repr__(self) -> str:
        return (
            f"MessageChannel(n_agents={self.n_agents}, "
            f"msg_dim={self.msg_dim}, "
            f"agg={self.aggregation})"
        )


# ---------------------------------------------------------------------------
# Communication diagnostics buffer
# ---------------------------------------------------------------------------

class CommBuffer:
    """
    Stores recent messages per agent for offline analysis and probing.

    Use `analysis/message_probe.py` to analyse what the agents have learned
    to communicate (e.g. do messages encode food direction?).

    Parameters
    ----------
    n_agents : int
    msg_dim  : int
    capacity : int   Max timesteps of history to keep (per agent).
    """

    def __init__(self, n_agents: int, msg_dim: int, capacity: int = 1000) -> None:
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.capacity = capacity
        self._buffers: list[deque] = [
            deque(maxlen=capacity) for _ in range(n_agents)
        ]

    def record(self, agent_id: int, message: np.ndarray) -> None:
        """Store a message for later analysis."""
        self._buffers[agent_id].append(message.copy())

    def get_history(self, agent_id: int) -> np.ndarray:
        """
        Return all stored messages for agent `agent_id`.

        Returns
        -------
        np.ndarray  Shape (T, msg_dim) where T ≤ capacity.
        """
        msgs = list(self._buffers[agent_id])
        if not msgs:
            return np.zeros((0, self.msg_dim), dtype=np.float32)
        return np.stack(msgs, axis=0)

    def reset(self) -> None:
        for buf in self._buffers:
            buf.clear()

    def __repr__(self) -> str:
        sizes = [len(b) for b in self._buffers]
        return f"CommBuffer(n_agents={self.n_agents}, stored={sizes})"


# ---------------------------------------------------------------------------
# Centralised critic aggregator
# ---------------------------------------------------------------------------

class CommAggregator:
    """
    Aggregates N agent observations + messages into a single context vector
    for the centralised critic in MAPPO.

    This is a simple concatenation + mean-pooling approach.  Replace with an
    attention mechanism for richer cross-agent conditioning.

    Parameters
    ----------
    n_agents     : int
    obs_dim      : int   Flattened observation dimension per agent.
    msg_dim      : int   Message dimension per agent.
    """

    def __init__(self, n_agents: int, obs_dim: int, msg_dim: int) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.output_dim = obs_dim + msg_dim   # per-agent joint feature

    def aggregate(
        self,
        obs_list: list[np.ndarray],
        msg_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Combine observations and messages into a centralised state vector.

        Parameters
        ----------
        obs_list   : list of (obs_dim,) arrays, one per agent.
        msg_matrix : (n_agents, msg_dim) — from MessageChannel.read_all().

        Returns
        -------
        np.ndarray  Shape (n_agents * (obs_dim + msg_dim),)  — flat joint state.
        """
        joint = np.concatenate(
            [np.concatenate([obs_list[i], msg_matrix[i]]) for i in range(self.n_agents)]
        )
        return joint.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"CommAggregator(n={self.n_agents}, "
            f"obs={self.obs_dim}, msg={self.msg_dim}, "
            f"out={self.n_agents * self.output_dim})"
        )
