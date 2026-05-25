"""Hardware energy model for the memristive timing network.

Physical assumptions
-------------------
Each weight is a memristive delay cell: an RRAM device in series with a
fixed load capacitor C_cell.  The delay is d = R_mem × C_cell = κ·exp(-u),
so κ = R_nom × C_cell where R_nom is the resistance at u=0 (log-conductance).

With default values κ = 15.81 ns:
  C_cell = 100 fF  →  R_nom = 158 kΩ  (RRAM LRS range: 10–500 kΩ  ✓)
  d_min  = 5 ns    →  R_min = 50 kΩ   (high conductance state)
  d_max  = 50 ns   →  R_max = 500 kΩ  (low conductance state)
  Ratio  = 10×     →  one decade of conductance swing  ✓

Energy sources
--------------
1. Delay-cell traversal (inference):
   E_cell = 0.5 × C_cell × V_dd²
   Only "active" cells dissipate energy — those whose input pulse actually
   arrives within the relevant timing window.

2. Race detection per output neuron (inference):
   E_race = n_out × E_comparator
   A clocked sense amplifier or current-race flip-flop that fires when the
   first input pulse arrives.  E_comparator ~ 1–10 fJ in 28 nm.

3. Input pulse drivers (inference):
   E_driver ≈ E_cell  (sized to drive C_cell)
   Only active (firing) inputs contribute.

4. TDC readout (training only):
   E_tdc = n_readings × E_tdc_sample
   A flash TDC with 150 ns range and 1 ns resolution needs ~150 comparators.
   Typical energy: 100 fJ–1 pJ per conversion at ns resolution.

5. P&V programming (training only):
   E_pv = n_params × n_pv_mean × E_pv_pulse
   Each SET/RESET pulse on the RRAM device: E_pulse = I_prog × V_prog × t_pulse.
   Typical RRAM values: 100 μA × 1.5 V × 10 ns = 1.5 fJ  (optimistic)
   to 1 mA × 2 V × 100 ns = 200 pJ  (conservative).
   We use 10 fJ as the nominal value (recent scaled RRAM, ~28 nm node).

Active fraction
---------------
Not all branches fire per inference.  From gradient analysis, the active
fraction (branches with meaningful nLSE weight) is 10–40% depending on:
  - Input sparsity (MNIST: ~50% pixels active per digit)
  - Timing window relative to τ: active ≈ branches where T_in + d ∈ [T_min, T_min + 3τ]

We model active_fraction_input as the fraction of inputs that actually fire
(x > threshold for binary; x > eps for time-encoded), and active_fraction_branch
as the fraction of branches within the nLSE window (default 0.25).

SPSA training energy
--------------------
Each SPSA epoch over N samples with batch size B:
  n_batches = N / B
  Per batch:
    - 3 forward passes (L0, L+, L-)  [3 × inference energy]
    - 3 rounds of P&V to program all weights to 3 states
  Per epoch:  n_batches × (3 × E_inference_batch + 3 × E_pv_all)

Digital comparison
------------------
Reference: 8-bit integer MAC at 28 nm ≈ 1.0 pJ  (based on Horowitz 2014 scaling).
The temporal network replaces MACCs with delay-cell traversals and race detection.
Effective energy per "equivalent MAC" = E_inference / n_mac_equiv where
n_mac_equiv = total weight × input multiply-adds.
"""

from __future__ import annotations

from dataclasses import dataclass

from memristor.network import MemristorNet


@dataclass
class HardwareSpec:
    """Physical parameters of the memristive timing circuit.

    All energies in fJ (femtojoules), times in ns, voltages in V,
    resistances in kΩ, capacitances in fF.
    """

    # ---- Delay cell -------------------------------------------------------
    C_cell_fF: float = 100.0
    """Load capacitance per delay cell (fF). Sets κ = R_nom × C_cell."""

    V_dd_V: float = 0.8
    """Supply voltage (V). Determines switching energy."""

    # ---- Race detection ---------------------------------------------------
    E_comparator_fJ: float = 5.0
    """Energy per race-detection event per output neuron (fJ).
    Clocked sense-amplifier or current-race flip-flop in 28 nm."""

    # ---- Input drivers ----------------------------------------------------
    E_driver_fJ: float | None = None
    """Energy per active input pulse driver (fJ).
    None → use E_cell (driver sized to match delay cell)."""

    # ---- TDC (training) ---------------------------------------------------
    E_tdc_fJ: float = 500.0
    """Energy per TDC conversion (fJ). Flash TDC, 150 ns range, 1 ns resolution.
    Conservative 28 nm estimate; SAR-TDC can achieve ~100 fJ."""

    # ---- P&V programming (training) ---------------------------------------
    E_pv_pulse_fJ: float = 10.0
    """Energy per single SET/RESET pulse on the RRAM device (fJ).
    Nominal: I=100 μA, V=1.5 V, t_pulse=10 ns → 150 fJ (conservative scaled).
    Aggressive: I=10 μA, V=1 V, t_pulse=1 ns → 10 fJ."""

    n_pv_pulses_mean: float = 20.0
    """Mean number of P&V pulses to program one cell to target delay.
    From programming.py simulation with nominal noise parameters."""

    # ---- Active fraction --------------------------------------------------
    active_fraction_input: float = 0.5
    """Fraction of inputs that actually fire (send a pulse) per inference.
    Binary inputs: ~50% active. MNIST time-encoded: ~40% (bright pixels)."""

    active_fraction_branch: float = 0.25
    """Fraction of branches in the nLSE active timing window per neuron.
    Branches with T_in + d outside [T_fastest, T_fastest + 3τ] contribute ~0."""

    # ---- Digital reference ------------------------------------------------
    E_mac_digital_fJ: float = 1000.0
    """Energy per 8-bit integer MAC in 28 nm (fJ). Horowitz 2014 scaling."""

    @property
    def E_cell_fJ(self) -> float:  # noqa: N802
        """Energy per pulse traversal through one delay cell (fJ)."""
        return 0.5 * self.C_cell_fF * self.V_dd_V**2

    @property
    def E_driver_effective_fJ(self) -> float:  # noqa: N802
        """Energy per active input driver (fJ)."""
        return self.E_driver_fJ if self.E_driver_fJ is not None else self.E_cell_fJ

    @property
    def E_pv_cell_fJ(self) -> float:  # noqa: N802
        """Total P&V energy to program one delay cell to its target (fJ)."""
        return self.n_pv_pulses_mean * self.E_pv_pulse_fJ


@dataclass
class InferenceEnergy:
    """Breakdown of energy for one forward pass on one sample."""

    E_input_fJ: float
    """Input pulse driver energy."""

    E_delay_fJ: float
    """Delay-cell traversal energy (only active cells)."""

    E_race_fJ: float
    """Race-detection (comparator) energy."""

    E_total_fJ: float
    """Total inference energy."""

    n_active_cells: float
    """Effective number of active delay cells contributing energy."""

    @property
    def E_total_pJ(self) -> float:  # noqa: N802
        return self.E_total_fJ / 1000.0

    def __str__(self) -> str:
        return (
            f"Inference energy breakdown (per sample):\n"
            f"  Input drivers:    {self.E_input_fJ:8.1f} fJ\n"
            f"  Delay cells:      {self.E_delay_fJ:8.1f} fJ   ({self.n_active_cells:.0f} active)\n"
            f"  Race detection:   {self.E_race_fJ:8.1f} fJ\n"
            f"  ─────────────────────────────────\n"
            f"  Total:            {self.E_total_fJ:8.1f} fJ  ({self.E_total_pJ:.3f} pJ)"
        )


@dataclass
class TrainingEnergy:
    """Energy breakdown for one training epoch (SPSA)."""

    E_inference_fJ: float
    """Energy for all 3 forward passes across all batches."""

    E_pv_fJ: float
    """Energy for all P&V programming rounds (3 per batch)."""

    E_tdc_fJ: float
    """Energy for TDC readout (one per output per sample, 3 passes)."""

    E_total_fJ: float
    """Total training epoch energy."""

    n_batches: int
    """Number of mini-batches per epoch."""

    @property
    def E_total_nJ(self) -> float:  # noqa: N802
        return self.E_total_fJ / 1e6

    def __str__(self) -> str:
        return (
            f"Training energy breakdown (per epoch, SPSA):\n"
            f"  3× forward passes: {self.E_inference_fJ/1e6:8.3f} nJ\n"
            f"  3× P&V rounds:     {self.E_pv_fJ/1e6:8.3f} nJ   ({self.n_batches} batches)\n"
            f"  TDC readout:       {self.E_tdc_fJ/1e6:8.3f} nJ\n"
            f"  ─────────────────────────────────\n"
            f"  Total:             {self.E_total_nJ:8.3f} nJ/epoch"
        )


def estimate_inference_energy(
    net: MemristorNet,
    spec: HardwareSpec,
    n_inputs_active: int | None = None,
) -> InferenceEnergy:
    """Estimate energy for one forward pass on one sample.

    Parameters
    ----------
    net:
        The MemristorNet to evaluate.
    spec:
        Physical hardware parameters.
    n_inputs_active:
        Number of input pulses that actually fire (x > 0 inputs).
        None → use spec.active_fraction_input × net.n_inputs.
    """
    if n_inputs_active is None:
        n_inputs_active = spec.active_fraction_input * net.n_inputs

    # Input driver energy: active inputs + bias (always fires)
    n_driving = n_inputs_active + 1  # +1 for bias
    E_input = n_driving * spec.E_driver_effective_fJ

    # Delay cell energy: propagate through layers
    # At each layer, only active_fraction_branch of cells see an active pulse
    E_delay = 0.0
    n_active_cells_total = 0.0
    current_active = n_driving  # active signals entering this layer

    for layer in net.layers:
        # Branches that are in the nLSE window (soft-min active region)
        n_active = current_active * layer.n_out * spec.active_fraction_branch
        E_delay += n_active * spec.E_cell_fJ
        n_active_cells_total += n_active
        # Hidden layer: n_out neurons fire, all feed next layer
        current_active = layer.n_out + 1  # +1 for bias at next layer

    # Race detection energy: one comparator event per output neuron per layer
    n_neurons_total = sum(layer.n_out for layer in net.layers)
    E_race = n_neurons_total * spec.E_comparator_fJ

    E_total = E_input + E_delay + E_race

    return InferenceEnergy(
        E_input_fJ=E_input,
        E_delay_fJ=E_delay,
        E_race_fJ=E_race,
        E_total_fJ=E_total,
        n_active_cells=n_active_cells_total,
    )


def estimate_training_energy(
    net: MemristorNet,
    spec: HardwareSpec,
    n_epochs: int,
    n_samples: int,
    batch_size: int = 128,
    n_inputs_active: int | None = None,
) -> TrainingEnergy:
    """Estimate total training energy for SPSA over n_epochs.

    Each epoch: for each mini-batch, 3 forward passes + 3 P&V rounds.

    Parameters
    ----------
    net:
        Network to train.
    spec:
        Physical hardware parameters.
    n_epochs:
        Number of training epochs.
    n_samples:
        Training set size.
    batch_size:
        Mini-batch size for SPSA.
    n_inputs_active:
        Active input count per sample. None → fraction estimate.
    """
    n_batches_per_epoch = max(1, n_samples // batch_size)
    n_batches_total = n_epochs * n_batches_per_epoch

    infer = estimate_inference_energy(net, spec, n_inputs_active)

    # 3 forward passes per batch (L0, L+, L-)
    E_inference = 3 * n_batches_total * batch_size * infer.E_total_fJ

    # 3 P&V rounds per batch: program ALL parameters to 3 states
    n_cells = net.n_delay_cells  # physical cells (2× branch pairs for standard)
    E_pv = 3 * n_batches_total * n_cells * spec.E_pv_cell_fJ

    # TDC: one reading per output neuron, 3 passes, all samples in batch
    n_outputs = net.layers[-1].n_out
    E_tdc = 3 * n_batches_total * batch_size * n_outputs * spec.E_tdc_fJ

    E_total = E_inference + E_pv + E_tdc

    return TrainingEnergy(
        E_inference_fJ=E_inference,
        E_pv_fJ=E_pv,
        E_tdc_fJ=E_tdc,
        E_total_fJ=E_total,
        n_batches=n_batches_per_epoch,
    )


def compare_to_digital(net: MemristorNet, spec: HardwareSpec) -> dict[str, float]:
    """Compare inference energy to a digital MAC baseline.

    The digital baseline performs n_mac multiply-accumulates per inference
    (one MAC per weight per active input), all at spec.E_mac_digital_fJ each.

    Returns a dict with:
      n_mac:          equivalent MAC operations
      E_digital_fJ:   digital energy (fJ)
      E_ours_fJ:      our timing network energy (fJ)
      speedup:        E_digital / E_ours  (>1 means we win)
    """
    infer = estimate_inference_energy(net, spec)

    # MACs: for each layer, n_in × n_out weights × n_active_inputs multiply-adds
    n_mac = 0.0
    n_active = spec.active_fraction_input * net.n_inputs + 1
    for layer in net.layers:
        n_mac += n_active * layer.n_out
        n_active = layer.n_out + 1

    E_digital = n_mac * spec.E_mac_digital_fJ

    return {
        "n_mac": n_mac,
        "E_digital_fJ": E_digital,
        "E_ours_fJ": infer.E_total_fJ,
        "speedup": E_digital / infer.E_total_fJ if infer.E_total_fJ > 0 else float("inf"),
        "E_per_mac_ours_fJ": infer.E_total_fJ / n_mac if n_mac > 0 else 0.0,
    }


def sensitivity_analysis(net: MemristorNet) -> None:
    """Print inference energy over a range of C_cell and E_pv_pulse assumptions."""
    print("Inference energy sensitivity to C_cell and V_dd")
    print("=" * 62)
    print(f"{'C_cell':>10}  {'V_dd':>6}  {'E_cell':>8}  {'E_infer':>10}  {'vs 8b MAC':>10}")
    print("-" * 62)

    for C_fF in (10.0, 50.0, 100.0, 500.0):
        for V in (0.5, 0.8, 1.2):
            spec = HardwareSpec(C_cell_fF=C_fF, V_dd_V=V)
            infer = estimate_inference_energy(net, spec)
            cmp = compare_to_digital(net, spec)
            e_cell = spec.E_cell_fJ
            e_inf = infer.E_total_fJ
            speedup = cmp["speedup"]
            print(
                f"{C_fF:>8.0f}fF  {V:>4.1f}V  {e_cell:>6.1f}fJ  "
                f"{e_inf:>8.0f}fJ  {speedup:>8.1f}×"
            )

    print()
    print("P&V energy per epoch (SPSA, MNIST, 200 epochs, n=6000, bs=128)")
    print("=" * 62)
    print(f"{'E_pv_pulse':>12}  {'n_pv_pulses':>12}  {'E_pv/epoch':>12}  {'E_total/epoch':>15}")
    print("-" * 62)

    for e_pv in (1.0, 10.0, 100.0, 1000.0):
        for n_pv in (10, 20, 50):
            spec = HardwareSpec(E_pv_pulse_fJ=e_pv, n_pv_pulses_mean=float(n_pv))
            tr = estimate_training_energy(net, spec, n_epochs=1, n_samples=6000, batch_size=128)
            print(
                f"{e_pv:>10.0f}fJ  {n_pv:>12d}  "
                f"{tr.E_pv_fJ/1e6:>10.3f}nJ  {tr.E_total_nJ:>13.3f}nJ"
            )
