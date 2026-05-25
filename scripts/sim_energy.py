"""Energy model simulation for the memristive temporal neural network.

Shows inference and training energy for XOR and MNIST architectures,
sensitivity analysis over physical parameters, and comparison to digital MACs.

Usage:
    uv run python scripts/sim_energy.py
"""

from __future__ import annotations

import numpy as np
import torch

from memristor.energy import (
    HardwareSpec,
    compare_to_digital,
    estimate_inference_energy,
    estimate_training_energy,
    sensitivity_analysis,
)
from memristor.network import MemristorNet


def _make_net(n_inputs: int, hidden: list[int], n_out: int) -> MemristorNet:
    torch.manual_seed(0)
    return MemristorNet(n_inputs=n_inputs, hidden_sizes=hidden, n_outputs=n_out)


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    spec = HardwareSpec()

    print_section("HARDWARE SPECIFICATION (nominal)")
    print(f"  Delay cell:     C_cell = {spec.C_cell_fF:.0f} fF,  V_dd = {spec.V_dd_V} V")
    print(f"  E_cell:         {spec.E_cell_fJ:.1f} fJ  (½CV²)")
    kappa_nom_ns = 15.81  # R_nom × C_cell at nominal parameters
    # ns/fF = 1e-9/1e-15 Ω = 1e6 Ω = 1e3 kΩ
    r_nom_kOhm = kappa_nom_ns / spec.C_cell_fF * 1e3
    print(f"  R_nom:          {r_nom_kOhm:.0f} kΩ  (κ = R_nom×C_cell = {kappa_nom_ns:.2f} ns)")
    print(f"  E_comparator:   {spec.E_comparator_fJ:.0f} fJ  (sense amplifier / race flip-flop)")
    print(f"  E_tdc:          {spec.E_tdc_fJ:.0f} fJ  (flash TDC, 150 ns range, 1 ns res)")
    print(f"  E_pv_pulse:     {spec.E_pv_pulse_fJ:.0f} fJ  (RRAM SET/RESET pulse)")
    print(f"  n_pv_pulses:    {spec.n_pv_pulses_mean:.0f}  (mean pulses per cell)")
    print(f"  E_pv_cell:      {spec.E_pv_cell_fJ:.0f} fJ  (total programming per cell)")
    print(
        f"  Active frac:    {spec.active_fraction_input:.0%} input, "
        f"{spec.active_fraction_branch:.0%} branch"
    )
    print(f"  Digital MAC:    {spec.E_mac_digital_fJ:.0f} fJ  (8-bit, 28 nm, Horowitz 2014)")

    # -------------------------------------------------------------------------
    # XOR network
    # -------------------------------------------------------------------------
    print_section("XOR NETWORK  (n_inputs=2, hidden=[2], n_out=1)")
    xor_net = _make_net(2, [2], 1)
    print(f"  Delay cells:    {xor_net.n_delay_cells}")
    xor_infer = estimate_inference_energy(xor_net, spec)
    print(xor_infer)
    xor_cmp = compare_to_digital(xor_net, spec)
    print(
        f"\n  Digital MAC baseline:  {xor_cmp['n_mac']:.0f} MACs × {spec.E_mac_digital_fJ:.0f} fJ"
        f" = {xor_cmp['E_digital_fJ']:.0f} fJ"
    )
    print(f"  Speedup vs digital:    {xor_cmp['speedup']:.1f}×")
    print(f"  Energy per MAC equiv:  {xor_cmp['E_per_mac_ours_fJ']:.1f} fJ")

    xor_train = estimate_training_energy(
        xor_net, spec, n_epochs=1000, n_samples=4, batch_size=4
    )
    print("\n  Training (1000 epochs, all 4 samples, SPSA):")
    print(xor_train)

    # -------------------------------------------------------------------------
    # MNIST network
    # -------------------------------------------------------------------------
    print_section("MNIST NETWORK  (n_inputs=64, hidden=[32], n_out=10)")
    mnist_net = _make_net(64, [32], 10)
    print(f"  Delay cells:    {mnist_net.n_delay_cells}")
    mnist_infer = estimate_inference_energy(mnist_net, spec)
    print(mnist_infer)
    mnist_cmp = compare_to_digital(mnist_net, spec)
    print(
        f"\n  Digital MAC baseline:  {mnist_cmp['n_mac']:.0f} MACs × {spec.E_mac_digital_fJ:.0f} fJ"
        f" = {mnist_cmp['E_digital_fJ']/1e3:.0f} pJ"
    )
    print(f"  Speedup vs digital:    {mnist_cmp['speedup']:.1f}×")
    print(f"  Energy per MAC equiv:  {mnist_cmp['E_per_mac_ours_fJ']:.2f} fJ")

    print("\n  Training (200 epochs, 6000 samples, batch=128, SPSA):")
    mnist_train = estimate_training_energy(
        mnist_net, spec, n_epochs=200, n_samples=6000, batch_size=128
    )
    print(mnist_train)
    print("\n  Training energy breakdown:")
    pct_infer = 100 * mnist_train.E_inference_fJ / mnist_train.E_total_fJ
    pct_pv    = 100 * mnist_train.E_pv_fJ / mnist_train.E_total_fJ
    pct_tdc   = 100 * mnist_train.E_tdc_fJ / mnist_train.E_total_fJ
    print(f"    Forward passes:  {pct_infer:.1f}%")
    print(f"    P&V programming: {pct_pv:.1f}%")
    print(f"    TDC readout:     {pct_tdc:.1f}%")

    # -------------------------------------------------------------------------
    # Inference latency estimate
    # -------------------------------------------------------------------------
    print_section("INFERENCE LATENCY ESTIMATE (MNIST)")
    kappa_ns = 15.81  # R_nom × C_cell
    tau_ns = 10.0
    T_inactive_ns = 150.0
    print(f"  κ (R_nom×C_cell):   {kappa_ns:.2f} ns")
    d_min = kappa_ns * np.exp(-np.log(10))
    print(f"  Delay range:        {d_min:.1f} – {kappa_ns:.1f} ns  (10× conductance ratio)")
    print(f"  nLSE window (3τ):   {3*tau_ns:.0f} ns  (τ = {tau_ns:.0f} ns)")
    print(f"  T_inactive:         {T_inactive_ns:.0f} ns  (silent input time)")
    n_layers = len(mnist_net.layers)
    latency_ns = T_inactive_ns + n_layers * (kappa_ns + 3 * tau_ns)
    print(f"  Estimated latency:  {latency_ns:.0f} ns  ({n_layers} layers, worst-case pipeline)")
    throughput_MHz = 1e3 / latency_ns  # 1 ns = 1e-9 s; MHz = 1e6/s
    print(f"  Throughput:         {throughput_MHz:.1f} MHz  (samples/s in pipelined mode)")

    # -------------------------------------------------------------------------
    # Sensitivity analysis
    # -------------------------------------------------------------------------
    print_section("SENSITIVITY ANALYSIS")
    sensitivity_analysis(mnist_net)


if __name__ == "__main__":
    main()
