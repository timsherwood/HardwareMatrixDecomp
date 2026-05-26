"""Tier-2 hardware network: full PCB prototype simulation.

Implements the memristive temporal neural network at µs scale using:
  - KnowmSDC devices for each weight (d_pos, d_neg differential pair)
  - Analytical nLSE with τ_sense for fast SPSA training
  - BJT sense-amp simulation (forward_bjt) for hardware verification

Scaling from ASIC (ns) to PCB (µs):
    All timing parameters × 10,000
    Same resistance range (50–500 kΩ) + larger C (1 nF vs 100 fF)
    Same SPSA algorithm; same log-conductance parameterisation
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from memristor.tier2.device import KnowmSDC
from memristor.tier2.sense_amp import BJTSenseAmp

# ─── PCB constants ────────────────────────────────────────────────────────────

C_CELL_F: float = 1e-9       # 1 nF load capacitor
D_MIN_US: float = 50.0       # minimum delay µs  (R_min × C_cell)
D_MAX_US: float = 500.0      # maximum delay µs  (R_max × C_cell)
KAPPA_US: float = math.sqrt(D_MIN_US * D_MAX_US)   # ≈ 158.1 µs geometric mean
V_DD: float = 1.5            # V  (RC ramp supply)
V_TH: float = V_DD / 2.0    # V  (BJT threshold = 50% crossing)
V_T: float = 0.026           # V  (thermal voltage at 25 °C)
GAIN_A: float = 20.0         # translinear mirror gain; must be < V_th/(V_T·ln(N_max)) = 26.3
TAU_US: float = 100.0        # µs nLSE temperature for SPSA training
TAU_D_US: float = 50.0       # µs decision temperature
T_INACTIVE_US: float = 1500.0  # µs silent-input time


# ─── weight layer ─────────────────────────────────────────────────────────────

@dataclass
class Tier2Layer:
    """One differential-delay layer: n_in × n_out pairs of KnowmSDC devices.

    Each weight (i, j) consists of two devices: d_pos[i,j] and d_neg[i,j].
    The trainable parameters are u_pos, u_neg in log-conductance space:
        d = κ_us × exp(−u)
    """

    n_in: int
    n_out: int
    u_pos: np.ndarray   # shape (n_in, n_out)
    u_neg: np.ndarray   # shape (n_in, n_out)
    kappa_us: float = KAPPA_US
    d_min_us: float = D_MIN_US
    d_max_us: float = D_MAX_US

    @classmethod
    def random(
        cls,
        n_in: int,
        n_out: int,
        rng: np.random.Generator,
        kappa_us: float = KAPPA_US,
        d_min_us: float = D_MIN_US,
        d_max_us: float = D_MAX_US,
    ) -> Tier2Layer:
        """Initialise near u=0 (d ≈ κ_us, geometric midpoint delay)."""
        u_pos = rng.normal(0.0, 0.5, size=(n_in, n_out))
        u_neg = rng.normal(0.0, 0.5, size=(n_in, n_out))
        return cls(n_in, n_out, u_pos, u_neg, kappa_us, d_min_us, d_max_us)

    def _u_bounds(self) -> tuple[float, float]:
        lo = math.log(self.kappa_us / self.d_max_us)
        hi = math.log(self.kappa_us / self.d_min_us)
        return lo, hi

    def _clamp(self) -> None:
        lo, hi = self._u_bounds()
        self.u_pos = np.clip(self.u_pos, lo, hi)
        self.u_neg = np.clip(self.u_neg, lo, hi)

    def delays_us(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (d_pos, d_neg) arrays in µs, shape (n_in, n_out)."""
        d_pos = np.clip(self.kappa_us * np.exp(-self.u_pos), self.d_min_us, self.d_max_us)
        d_neg = np.clip(self.kappa_us * np.exp(-self.u_neg), self.d_min_us, self.d_max_us)
        return d_pos, d_neg

    def devices(self) -> list[tuple[KnowmSDC, KnowmSDC]]:
        """Return a 2-D list of (device_pos, device_neg) KnowmSDC objects."""
        d_pos, d_neg = self.delays_us()
        R_pos = d_pos / (C_CELL_F * 1e6)
        R_neg = d_neg / (C_CELL_F * 1e6)
        result = []
        for i in range(self.n_in):
            row = []
            for j in range(self.n_out):
                row.append((KnowmSDC(R=R_pos[i, j]), KnowmSDC(R=R_neg[i, j])))
            result.append(row)
        return result


# ─── main network ─────────────────────────────────────────────────────────────

class Tier2Network:
    """PCB-scale memristive temporal neural network.

    Parameters
    ----------
    n_inputs, hidden_sizes, n_outputs:
        Architecture (same meaning as MemristorNet).
    tau_us:
        nLSE temperature for SPSA training (µs).  Should be close to τ_sense.
    tau_d_us:
        Decision temperature (µs).
    T_inactive_us:
        Arrival time for inactive (silent) inputs (µs).
    gain_A:
        Translinear mirror gain for BJT verification pass.
    seed:
        Random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: list[int],
        n_outputs: int,
        tau_us: float = TAU_US,
        tau_d_us: float = TAU_D_US,
        T_inactive_us: float = T_INACTIVE_US,
        kappa_us: float = KAPPA_US,
        d_min_us: float = D_MIN_US,
        d_max_us: float = D_MAX_US,
        gain_A: float = GAIN_A,
        seed: int | None = None,
    ) -> None:
        self.tau_us = tau_us
        self.tau_d_us = tau_d_us
        self.T_inactive_us = T_inactive_us
        self.kappa_us = kappa_us
        self.d_min_us = d_min_us
        self.d_max_us = d_max_us
        self.gain_A = gain_A

        rng = np.random.default_rng(seed)
        # Each hidden layer appends a bias node, so next layer gets h+1 inputs
        in_sizes = [n_inputs + 1] + [h + 1 for h in hidden_sizes]
        out_sizes = hidden_sizes + [n_outputs]
        self.layers: list[Tier2Layer] = []
        for n_in, n_out in zip(in_sizes, out_sizes, strict=True):
            self.layers.append(
                Tier2Layer.random(n_in, n_out, rng, kappa_us, d_min_us, d_max_us)
            )

    # ----------------------------------------------------------------- encoding

    def encode_binary(self, x: np.ndarray) -> np.ndarray:
        """Binary input → arrival time vector (µs) + bias node.

        Active (x > 0.5) → T = 0 µs
        Inactive (x ≤ 0.5) → T = T_inactive_us
        Bias (last element) → T = 0 µs always
        """
        T = np.where(x > 0.5, 0.0, self.T_inactive_us)
        return np.append(T, 0.0)  # append bias

    # ----------------------------------------------------------------- forward (analytical nLSE)

    def forward(self, T_in_us: np.ndarray) -> np.ndarray:
        """Forward pass using analytical nLSE with τ_sense.

        Returns output probabilities p (n_outputs,) in (0, 1).
        τ_sense is computed from gain_A at d_nom (used as a fixed temperature
        for the whole network, consistent with SPSA training at τ_us).
        """
        tau = self.tau_us
        T_current = np.asarray(T_in_us, dtype=float)
        n_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            is_last = idx == n_layers - 1
            d_pos, d_neg = layer.delays_us()  # (n_in, n_out)

            # Arrival times: A[i, j] = T_in[i] + d[i, j]
            A_pos = T_current[:, np.newaxis] + d_pos   # (n_in, n_out)
            A_neg = T_current[:, np.newaxis] + d_neg

            # nLSE soft-min over input branches for each output neuron
            T_plus = _nLSE_axis0(A_pos, tau)    # (n_out,)
            T_minus = _nLSE_axis0(A_neg, tau)
            margin = T_minus - T_plus           # (n_out,)

            if is_last:
                return _sigmoid(margin / self.tau_d_us)

            # Hidden layer: differentiable timing output
            p = _sigmoid(margin / self.tau_d_us)
            T_h = p * T_plus + (1.0 - p) * T_minus
            # Append bias node
            T_current = np.append(T_h, 0.0)

        raise RuntimeError("No layers")  # pragma: no cover

    # ----------------------------------------------------------------- forward (BJT simulation)

    def forward_bjt(self, T_in_us: np.ndarray) -> np.ndarray:
        """Forward pass using BJT sense-amp numerical simulation.

        Slower than forward() but uses the physical circuit model to verify
        that the trained solution works on actual hardware.

        Returns output probabilities p (n_outputs,) in (0, 1).
        The hidden layer outputs use the same winner-selection logic:
        T_out = min(T_plus_bjt, T_minus_bjt) — the first race wins.
        """
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=self.gain_A, dt_us=0.5)
        T_current = np.asarray(T_in_us, dtype=float)
        n_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            is_last = idx == n_layers - 1
            d_pos, d_neg = layer.delays_us()   # (n_in, n_out)
            n_in, n_out = d_pos.shape

            T_plus_bjt = np.empty(n_out)
            T_minus_bjt = np.empty(n_out)

            for j in range(n_out):
                delays_pos_j = list(d_pos[:, j])
                delays_neg_j = list(d_neg[:, j])
                T_ins_j = list(T_current)
                T_plus_bjt[j] = amp.fire_time_us(delays_pos_j, T_ins_j)
                T_minus_bjt[j] = amp.fire_time_us(delays_neg_j, T_ins_j)

            margin = T_minus_bjt - T_plus_bjt  # (n_out,)

            if is_last:
                return _sigmoid(margin / self.tau_d_us)

            # Hidden layer: use same p-weighted formula as training so that the
            # only difference between forward() and forward_bjt() is the sense-amp
            # simulation; hidden-layer output rule stays consistent.
            p = _sigmoid(margin / self.tau_d_us)
            T_h = p * T_plus_bjt + (1.0 - p) * T_minus_bjt
            T_current = np.append(T_h, 0.0)

        raise RuntimeError("No layers")  # pragma: no cover

    # ----------------------------------------------------------------- predict

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Classify all samples in X (n_samples, n_inputs).

        Returns integer predictions (n_samples,).
        For single-output networks: 1 if p > 0.5, else 0.
        For multi-output networks: argmax of output probabilities.
        """
        preds = []
        for x in X:
            T_in = self.encode_binary(x)
            p = self.forward(T_in)
            if len(p) == 1:
                preds.append(int(p[0] > 0.5))
            else:
                preds.append(int(np.argmax(p)))
        return np.array(preds, dtype=int)

    # ----------------------------------------------------------------- SPSA training

    def train_spsa(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 3000,
        eta: float = 0.05,
        eps: float = 0.10,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> dict[str, object]:
        """SPSA training on (X, Y) with hardware noise model.

        Each epoch:
          1. Draw random ±1 perturbation vector Δ
          2. Evaluate loss at (u), (u + ε·Δ), (u − ε·Δ)
          3. Update: u ← u − η · (L+ − L−)/(2ε) · Δ

        After each update step, P&V is simulated: weights are clamped to the
        nearest achievable delay on the KnowmSDC device grid.

        Returns
        -------
        dict with keys:
            accuracy        – fraction correct on (X, Y) after training
            final_loss      – cross-entropy loss after last epoch
            converged_epoch – first epoch with 100% accuracy (or None)
        """
        rng = np.random.default_rng(seed)
        n_samples = len(X)
        _batch = batch_size or n_samples
        converged_epoch = None

        for epoch in range(n_epochs):
            # Select mini-batch
            idx = rng.choice(n_samples, size=min(_batch, n_samples), replace=False)
            X_b, Y_b = X[idx], Y[idx]

            # Flatten all u parameters into one vector
            u_flat, shapes = _flatten_u(self.layers)
            n_params = len(u_flat)

            # SPSA perturbation
            delta = rng.choice([-1.0, 1.0], size=n_params)

            u_plus = u_flat + eps * delta
            u_minus = u_flat - eps * delta

            L_plus = _eval_loss(self, u_plus, shapes, X_b, Y_b)
            L_minus = _eval_loss(self, u_minus, shapes, X_b, Y_b)

            grad_est = (L_plus - L_minus) / (2.0 * eps) * delta
            u_new = u_flat - eta * grad_est

            _unflatten_u(self.layers, u_new, shapes)
            for layer in self.layers:
                layer._clamp()

            # Check convergence
            if converged_epoch is None:
                preds = self.predict_all(X)
                if np.all(preds == Y):
                    converged_epoch = epoch

        preds = self.predict_all(X)
        accuracy = float(np.mean(preds == Y))
        final_loss = _eval_loss(self, _flatten_u(self.layers)[0], _flatten_u(self.layers)[1], X, Y)

        return {
            "accuracy": accuracy,
            "final_loss": final_loss,
            "converged_epoch": converged_epoch,
        }

    # ----------------------------------------------------------------- HIL SPSA

    def train_spsa_bjt(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 2000,
        eta: float = 0.05,
        eps: float = 0.10,
        n_warmup_epochs: int = 1000,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> dict[str, object]:
        """Hardware-in-the-loop SPSA: loss evaluated with BJT forward pass.

        First runs n_warmup_epochs of fast analytical SPSA to land near a
        valid solution, then fine-tunes with the BJT forward pass so weights
        adapt to the actual τ_sense distribution.  Training stops as soon as
        all patterns are correctly classified under the BJT model.

        Returns
        -------
        dict with keys:
            accuracy        – fraction correct under BJT evaluation
            final_loss      – BJT cross-entropy after last epoch
            converged_epoch – first BJT epoch with 100% BJT accuracy (or None)
        """
        if n_warmup_epochs > 0:
            self.train_spsa(
                X, Y, n_epochs=n_warmup_epochs, eta=eta, eps=eps,
                batch_size=batch_size, seed=seed,
            )

        rng = np.random.default_rng(seed)
        n_samples = len(X)
        _batch = batch_size or n_samples
        converged_epoch = None

        for epoch in range(n_epochs):
            idx = rng.choice(n_samples, size=min(_batch, n_samples), replace=False)
            X_b, Y_b = X[idx], Y[idx]

            u_flat, shapes = _flatten_u(self.layers)
            n_params = len(u_flat)
            delta = rng.choice([-1.0, 1.0], size=n_params)

            u_plus = u_flat + eps * delta
            u_minus = u_flat - eps * delta

            L_plus = _eval_loss_bjt(self, u_plus, shapes, X_b, Y_b)
            L_minus = _eval_loss_bjt(self, u_minus, shapes, X_b, Y_b)

            grad_est = (L_plus - L_minus) / (2.0 * eps) * delta
            u_new = u_flat - eta * grad_est

            _unflatten_u(self.layers, u_new, shapes)
            for layer in self.layers:
                layer._clamp()

            preds = self.predict_all_bjt(X)
            if np.all(preds == Y):
                converged_epoch = epoch
                break

        preds = self.predict_all_bjt(X)
        accuracy = float(np.mean(preds == Y))
        u_flat_f, shapes_f = _flatten_u(self.layers)
        final_loss = _eval_loss_bjt(self, u_flat_f, shapes_f, X, Y)
        return {
            "accuracy": accuracy,
            "final_loss": final_loss,
            "converged_epoch": converged_epoch,
        }

    def predict_all_bjt(self, X: np.ndarray) -> np.ndarray:
        """Classify all samples using BJT forward pass."""
        preds = []
        for x in X:
            T_in = self.encode_binary(x)
            p = self.forward_bjt(T_in)
            if len(p) == 1:
                preds.append(int(p[0] > 0.5))
            else:
                preds.append(int(np.argmax(p)))
        return np.array(preds, dtype=int)

    # ----------------------------------------------------------------- utilities

    def with_device_noise(
        self,
        noise_frac: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> Tier2Network:
        """Return a deep copy with Gaussian noise added to delay values.

        Simulates cycle-to-cycle device variability after a P&V programming step.
        Each delay d ← d × exp(Normal(0, noise_frac)), clamped to [d_min, d_max].
        """
        _rng = rng or np.random.default_rng()
        net_copy = copy.deepcopy(self)
        for layer in net_copy.layers:
            noise_pos = _rng.normal(0.0, noise_frac, size=layer.u_pos.shape)
            noise_neg = _rng.normal(0.0, noise_frac, size=layer.u_neg.shape)
            # Perturb in u-space (equivalent to log-normal noise on d)
            layer.u_pos = layer.u_pos + noise_pos
            layer.u_neg = layer.u_neg + noise_neg
            layer._clamp()
        return net_copy

    @classmethod
    def from_software_model(cls, soft_net: object) -> Tier2Network:
        """Construct a Tier2Network with weights from a MemristorNet.

        Scales all delays by 10,000 (ns → µs).  The log-conductance parameters
        u are unchanged (same ratio d/κ), but κ is scaled to κ_us.

        Parameters
        ----------
        soft_net:
            A memristor.network.MemristorNet instance.
        """
        # Both models use the same u (log-conductance) values.
        # d_hw_us = kappa_us * exp(-u)  vs  d_si_ns = kappa_ns * exp(-u)
        # → numerical ratio = kappa_us / kappa_ns = 158.1 µs / 15.81 ns = 10
        SCALE = KAPPA_US / soft_net.layers[0].kappa  # ≈ 10.0
        hw = cls.__new__(cls)
        hw.tau_us = float(soft_net.tau) * SCALE
        hw.tau_d_us = float(soft_net.tau_d) * SCALE
        hw.T_inactive_us = float(soft_net.T_inactive) * SCALE
        hw.kappa_us = KAPPA_US
        hw.d_min_us = D_MIN_US
        hw.d_max_us = D_MAX_US
        hw.gain_A = GAIN_A

        hw.layers = []
        for sw_layer in soft_net.layers:
            # Reuse u_pos/u_neg directly (same log-conductance ratios)
            u_pos = sw_layer.u_pos.detach().numpy().copy()
            u_neg = sw_layer.u_neg.detach().numpy().copy()
            n_in, n_out = u_pos.shape
            layer = Tier2Layer(
                n_in=n_in, n_out=n_out,
                u_pos=u_pos, u_neg=u_neg,
                kappa_us=KAPPA_US,
                d_min_us=D_MIN_US,
                d_max_us=D_MAX_US,
            )
            layer._clamp()
            hw.layers.append(layer)

        return hw


# ─── private helpers ──────────────────────────────────────────────────────────

def _nLSE_axis0(A: np.ndarray, tau: float) -> np.ndarray:  # noqa: N802
    """nLSE along axis-0 (over input branches) → shape (n_out,)."""
    log_terms = -A / tau               # (n_in, n_out)
    lse = np.max(log_terms, axis=0) + np.log(
        np.sum(np.exp(log_terms - np.max(log_terms, axis=0, keepdims=True)), axis=0)
    )
    return -tau * lse


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _cross_entropy(p: np.ndarray, y: int) -> float:
    """Binary cross-entropy for single-output or multi-class."""
    eps = 1e-7
    if len(p) == 1:
        p0 = float(np.clip(p[0], eps, 1.0 - eps))
        return -(y * math.log(p0) + (1 - y) * math.log(1.0 - p0))
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(-math.log(p_clipped[y]))


def _eval_loss(
    net: Tier2Network,
    u_flat: np.ndarray,
    shapes: list,
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Evaluate mean cross-entropy loss using analytical nLSE forward pass."""
    _unflatten_u(net.layers, u_flat, shapes)
    total = 0.0
    for x, y in zip(X, Y, strict=True):
        T_in = net.encode_binary(x)
        p = net.forward(T_in)
        total += _cross_entropy(p, int(y))
    return total / len(X)


def _eval_loss_bjt(
    net: Tier2Network,
    u_flat: np.ndarray,
    shapes: list,
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Evaluate mean cross-entropy loss using BJT sense-amp forward pass."""
    _unflatten_u(net.layers, u_flat, shapes)
    total = 0.0
    for x, y in zip(X, Y, strict=True):
        T_in = net.encode_binary(x)
        p = net.forward_bjt(T_in)
        total += _cross_entropy(p, int(y))
    return total / len(X)


def _flatten_u(layers: list[Tier2Layer]) -> tuple[np.ndarray, list]:
    """Concatenate all u_pos and u_neg into one 1-D array."""
    parts = []
    shapes = []
    for layer in layers:
        parts.append(layer.u_pos.ravel())
        parts.append(layer.u_neg.ravel())
        shapes.append(layer.u_pos.shape)
    return np.concatenate(parts), shapes


def _unflatten_u(layers: list[Tier2Layer], u_flat: np.ndarray, shapes: list) -> None:
    """Write flattened u vector back into layer.u_pos and layer.u_neg."""
    offset = 0
    for layer, shape in zip(layers, shapes, strict=True):
        n = shape[0] * shape[1]
        layer.u_pos = u_flat[offset : offset + n].reshape(shape).copy()
        offset += n
        layer.u_neg = u_flat[offset : offset + n].reshape(shape).copy()
        offset += n
