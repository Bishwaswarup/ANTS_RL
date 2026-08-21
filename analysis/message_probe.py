"""
analysis/message_probe.py
=========================
Probing tools for analysing what EAC agents have learned to communicate.

Probing is the technique of training a simple linear classifier on top of
frozen agent representations to ask: "does the latent space linearly encode
concept X?"

In the EAC context, the key questions are:
  - Do agent messages encode food direction?
  - Do LSTM hidden states track whether the agent is carrying food?
  - Does the visual feature extractor separate wall-adjacent cells from open ones?

Classes
-------
  LinearProbe       — train a logistic regression on top of frozen features
  DirectionalProbe  — specialised probe: does a vector encode direction to target?
  MessageAnalyser   — high-level tool that runs standard probes on CommBuffer data

Usage
-----
::

    from agents.comms import CommBuffer
    from analysis.message_probe import MessageAnalyser

    # After a multi-agent eval run:
    analyser = MessageAnalyser(comm_buffer, n_agents=4)
    results = analyser.run_all_probes(labels)   # labels = dict of concept→array
    print(results)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Linear probe (logistic regression with gradient descent)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Results from a single probing experiment."""
    probe_name: str
    train_acc: float
    val_acc: float
    baseline_acc: float   # majority-class accuracy
    improvement: float    # val_acc - baseline_acc

    @property
    def is_significant(self) -> bool:
        """True if probe accuracy is ≥5pp above the majority baseline."""
        return self.improvement >= 0.05

    def __repr__(self) -> str:
        sig = "✓" if self.is_significant else "✗"
        return (
            f"ProbeResult[{self.probe_name}] "
            f"val={self.val_acc:.1%}  base={self.baseline_acc:.1%}  "
            f"Δ={self.improvement:+.1%}  {sig}"
        )


class LinearProbe:
    """
    Lightweight logistic regression probe trained with mini-batch SGD.

    Parameters
    ----------
    input_dim     : int   Dimension of the feature vector (e.g. msg_dim).
    n_classes     : int   Number of target classes (2 = binary).
    lr            : float Learning rate.
    n_epochs      : int   Training epochs.
    val_split     : float Fraction of data held out for validation (0–1).
    random_seed   : int
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        lr: float = 0.01,
        n_epochs: int = 200,
        val_split: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.rng = np.random.default_rng(random_seed)

        # Weight matrix: (n_classes, input_dim)
        self.W = self.rng.standard_normal((n_classes, input_dim)).astype(np.float32) * 0.01
        self.b = np.zeros(n_classes, dtype=np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W.T + self.b)

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self._forward(X).argmax(axis=1)
        return float((preds == y).mean())

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Train the probe.

        Parameters
        ----------
        X : (N, input_dim) float array — feature vectors.
        y : (N,) int array — class labels.

        Returns
        -------
        (train_accuracy, val_accuracy)
        """
        N = len(X)
        n_val = max(1, int(N * self.val_split))
        idx = self.rng.permutation(N)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # One-hot encode
        Y_train = np.eye(self.n_classes)[y_train]

        for _ in range(self.n_epochs):
            probs = self._forward(X_train)          # (N_train, n_classes)
            delta = probs - Y_train                  # (N_train, n_classes)
            dW = (delta.T @ X_train) / len(X_train) # (n_classes, input_dim)
            db = delta.mean(axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

        train_acc = self._accuracy(X_train, y_train)
        val_acc   = self._accuracy(X_val, y_val)
        return train_acc, val_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# Directional probe
# ---------------------------------------------------------------------------

class DirectionalProbe:
    """
    Ask: does a vector space linearly encode direction to a target?

    Converts the continuous direction angle to one of 4 or 8 discrete classes,
    then trains a LinearProbe to recover the class from the feature vector.

    Parameters
    ----------
    n_directions : int  4 = N/E/S/W, 8 = adds diagonals.
    """

    def __init__(self, n_directions: int = 4, **probe_kwargs) -> None:
        self.n_directions = n_directions
        self.probe = LinearProbe(
            input_dim=probe_kwargs.pop("input_dim", 16),
            n_classes=n_directions,
            **probe_kwargs,
        )

    @staticmethod
    def direction_labels(
        positions: np.ndarray,
        target: tuple,
        n_directions: int = 4,
    ) -> np.ndarray:
        """
        Compute discrete direction labels (0..n_directions-1) from positions → target.

        Parameters
        ----------
        positions  : (N, 2) array of (x, y) agent positions.
        target     : (tx, ty) target position.
        n_directions : int

        Returns
        -------
        np.ndarray  (N,) int array.
        """
        dx = target[0] - positions[:, 0]
        dy = target[1] - positions[:, 1]
        angles = np.arctan2(dy, dx)                     # [-π, π]
        angles = (angles + np.pi) / (2 * np.pi)         # [0, 1)
        labels = (angles * n_directions).astype(int) % n_directions
        return labels

    def fit(
        self,
        features: np.ndarray,
        positions: np.ndarray,
        target: tuple,
    ) -> ProbeResult:
        """
        Train the directional probe.

        Parameters
        ----------
        features  : (N, D) feature vectors (e.g. agent messages).
        positions : (N, 2) agent (x, y) positions at those timesteps.
        target    : (tx, ty) food or nest position.

        Returns
        -------
        ProbeResult
        """
        labels = self.direction_labels(positions, target, self.n_directions)
        baseline = float(np.bincount(labels).max()) / len(labels)

        train_acc, val_acc = self.probe.fit(features, labels)

        return ProbeResult(
            probe_name=f"DirectionalProbe({self.n_directions}-way)",
            train_acc=train_acc,
            val_acc=val_acc,
            baseline_acc=baseline,
            improvement=val_acc - baseline,
        )


# ---------------------------------------------------------------------------
# High-level message analyser
# ---------------------------------------------------------------------------

class MessageAnalyser:
    """
    Runs a battery of standard probes over a CommBuffer's recorded messages.

    Parameters
    ----------
    comm_buffer : CommBuffer  Filled by a multi-agent evaluation run.
    n_agents    : int
    """

    def __init__(self, comm_buffer, n_agents: int) -> None:
        self.comm_buffer = comm_buffer
        self.n_agents = n_agents

    def probe_food_direction(
        self,
        agent_id: int,
        positions: np.ndarray,
        food_pos: tuple,
        n_directions: int = 4,
    ) -> ProbeResult:
        """
        Does agent `agent_id`'s message encode the direction to food?

        Parameters
        ----------
        positions : (N, 2) int array — agent positions at each recorded step.
        food_pos  : (fx, fy) fixed food location.
        """
        messages = self.comm_buffer.get_history(agent_id)   # (T, msg_dim)
        T = min(len(messages), len(positions))
        messages, positions = messages[:T], positions[:T]

        probe = DirectionalProbe(
            n_directions=n_directions,
            input_dim=messages.shape[1],
        )
        return probe.fit(messages, positions, food_pos)

    def probe_has_food(
        self,
        agent_id: int,
        has_food_labels: np.ndarray,
    ) -> ProbeResult:
        """
        Does the message encode whether the agent is carrying food?

        Parameters
        ----------
        has_food_labels : (N,) bool/int array — 1 if carrying food, 0 otherwise.
        """
        messages = self.comm_buffer.get_history(agent_id)
        T = min(len(messages), len(has_food_labels))
        messages = messages[:T]
        labels = has_food_labels[:T].astype(int)

        baseline = float(np.bincount(labels).max()) / len(labels)
        probe = LinearProbe(input_dim=messages.shape[1], n_classes=2)
        train_acc, val_acc = probe.fit(messages, labels)

        return ProbeResult(
            probe_name=f"HasFoodProbe(agent={agent_id})",
            train_acc=train_acc,
            val_acc=val_acc,
            baseline_acc=baseline,
            improvement=val_acc - baseline,
        )

    def run_standard_battery(
        self,
        agent_id: int,
        positions: np.ndarray,
        food_pos: tuple,
        has_food_labels: np.ndarray,
    ) -> list[ProbeResult]:
        """
        Run the standard probe battery for one agent.

        Returns a list of ProbeResult objects sorted by improvement (best first).
        """
        results = [
            self.probe_food_direction(agent_id, positions, food_pos, n_directions=4),
            self.probe_food_direction(agent_id, positions, food_pos, n_directions=8),
            self.probe_has_food(agent_id, has_food_labels),
        ]
        return sorted(results, key=lambda r: r.improvement, reverse=True)
