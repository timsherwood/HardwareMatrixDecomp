# Memristive Temporal Neural Network — Hardware Specification

**For: Engineering Team**
**Status: Prototype design, simulation-validated**

---

## 1. Concept Summary

This is a neural network accelerator that replaces multiply-accumulate operations with **pulse-arrival-time races through memristive delay cells**. Each weight is a programmable delay rather than a stored number. Classification decisions are made by detecting which signal wins a race — no digital arithmetic required during inference.

The key claims, all simulation-validated:

| Metric | Value |
|---|---|
| Inference energy (MNIST 8×8, 64→32→10) | **12.4 pJ/sample** |
| Versus 8-bit integer MAC baseline (28 nm) | **112× lower energy** |
| Inference latency (MNIST, 2 layers) | **~240 ns** |
| Throughput (pipelined) | **~4 MHz** |
| Jitter tolerance | **σ_j < 1 ns** (robust at σ_j < τ/10 = 1 ns) |
| Training method | **SPSA** — only 3 forward passes per weight update, no backprop |

---

## 2. Physical Device: RRAM Delay Cell

Each weight is implemented as a **single-port RRAM (Resistive RAM) device** in series with a fixed load capacitor.

```
  V_pulse ──┤ RRAM ├──┬── delayed_output
            (R_mem)   │
                    C_cell
                      │
                     GND
```

**Device parameters (28 nm node):**

| Parameter | Value | Notes |
|---|---|---|
| C_cell | 100 fF | Fixed load capacitor |
| V_dd | 0.8 V | Supply voltage |
| R_nom (at u=0) | 158 kΩ | Within RRAM LRS range: 10–500 kΩ ✓ |
| R_min (d_min) | 50 kΩ | High-conductance state |
| R_max (d_max) | 500 kΩ | Low-conductance state |
| Conductance ratio | 10× | One decade swing — achievable with standard RRAM |
| E_cell (½CV²) | 32 fJ | Energy per pulse traversal |

**Delay equation:**

```
d = κ · exp(−u)

where:
  d   = propagation delay (ns)
  κ   = R_nom × C_cell = 15.81 ns  (fabrication calibration constant)
  u   = ln(G / G_ref) = log-conductance (the trained weight)
```

This gives d_min = 5 ns at high conductance, d_max = 50 ns at low conductance — a 10:1 range using one decade of RRAM conductance swing.

---

## 3. Differential Branch Pair (One Weight)

Each scalar weight is implemented as a **differential pair of two delay cells**: one positive branch and one negative branch.

```
  T_in ──┬──[ d_pos ]──► A_pos = T_in + d_pos  ─┐
         │                                        ├── margin = T_minus − T_plus
         └──[ d_neg ]──► A_neg = T_in + d_neg  ─┘
```

- `d_pos < d_neg` → pulse arrives early on positive side → positive weight contribution
- `d_pos > d_neg` → pulse arrives early on negative side → negative weight contribution
- `d_pos ≈ d_neg` → near-zero weight

This gives **bidirectional weight** without needing signed arithmetic. Each weight requires two RRAM cells.

**Complementary variant (half the devices):** `d_neg = (d_min + d_max) − d_pos`. This hardwires the constraint that both branches sum to a constant, cutting device count in half. Works well; simulation shows equivalent convergence.

---

## 4. Layer Architecture

A layer maps N_in arrival-time inputs to N_out output neurons. Each output neuron runs a **race detection** event.

### 4.1 Positive and Negative Races

For output neuron j receiving inputs T_in[0..N_in-1]:

```
A_pos[i,j] = T_in[i] + d_pos[i,j]     for each input i

T_plus[j]  = −τ · log Σ_i exp(−A_pos[i,j] / τ)   ← nLSE soft-min
T_minus[j] = −τ · log Σ_i exp(−A_neg[i,j] / τ)   ← nLSE soft-min

margin[j] = T_minus[j] − T_plus[j]
p[j] = sigmoid(margin[j] / τ_d)                   ← output probability
```

**nLSE (negative log-sum-exp)** is the differentiable approximation to a minimum — it selects approximately the first arriving pulse, with softness controlled by temperature τ.

### 4.2 Circuit Implementation

**The nLSE is naturally implemented by a current-summing sense amplifier:**

- Each positive branch pulse drives a small current onto a shared wire
- The first sufficiently large current deflection triggers the sense amplifier
- This is electrically equivalent to the soft-min of arrival times
- Race detection energy: ~5 fJ per output neuron (clocked sense amp / current-race flip-flop at 28 nm)

### 4.3 Hidden Layer Output

Hidden neurons forward a timing signal to the next layer:

```
T_out[j] = p[j] · T_plus[j] + (1 − p[j]) · T_minus[j]
```

This interpolates between the two races, keeping the signal differentiable for training purposes. Physically it is a voltage interpolation between two tapped delay lines.

---

## 5. Network Architecture (MNIST Example)

```
Input layer:     64 pixels  →  64 arrival times T_in[0..63]
                              + 1 bias node (always fires at t=0)
                              = 65-dimensional arrival-time vector

Layer 0: DelayLayer(65 inputs → 32 hidden neurons)
         65 × 32 = 2,080 differential branch pairs
         = 4,160 individual RRAM delay cells

Layer 1: DelayLayer(33 inputs → 10 output neurons)   [32 hidden + 1 bias]
         33 × 10 = 330 differential branch pairs
         = 660 individual RRAM delay cells

TOTAL:   2,410 differential pairs = 4,820 RRAM cells
         10 output race detectors (one per class)
```

---

## 6. Input Encoding

### Binary inputs

```
x[i] > 0.5  →  T[i] = 0 ns   (pulse fires immediately)
x[i] ≤ 0.5  →  T[i] = 150 ns  (T_inactive — outside the timing window)
```

### Continuous / analog inputs (MNIST pixels)

```
T[i] = clip(−50 · ln(x[i]), 0, T_inactive)

Bright pixel (x→1) → T≈0 ns    (fires early)
Dark pixel   (x→0) → T→150 ns  (fires late / silent)
```

The encoding is performed by a **time-to-digital ring oscillator driver**: input voltage controls gating of a ring oscillator, producing a pulse whose delay is proportional to `−ln(x)`.

---

## 7. Timing Budget

| Parameter | Value | Notes |
|---|---|---|
| κ (RC product) | 15.81 ns | R_nom × C_cell; fabrication calibrated |
| d_min | 5 ns | Fastest delay (max conductance) |
| d_max | 50 ns | Slowest delay (min conductance) |
| τ (nLSE temperature) | 10 ns | Soft-min integration window |
| τ_d (decision temp) | 5 ns | Sigmoid sharpness |
| nLSE active window | ~30 ns | ≈ 3τ around the fastest arrival |
| T_inactive (silent) | 150 ns | Time after which a "0" input is ignored |
| T_inactive / d_max | 3× | Silent inputs stay clear of active window |
| Per-layer latency | ~60–80 ns | d_max + 3τ |
| 2-layer total latency | ~240 ns | Plus T_inactive for first input |
| Throughput (pipelined) | ~4 MHz | New sample every ~240 ns |

**Jitter tolerance:** Simulation confirms robust operation at σ_j ≤ 1 ns (σ_j / τ ≤ 0.1). Timing noise corrupts results when σ_j approaches τ = 10 ns.

---

## 8. Energy Model

### Inference (per sample, MNIST 64→32→10)

| Component | Energy | Notes |
|---|---|---|
| Input pulse drivers | 1,056 fJ | 33 active inputs × 32 fJ/driver |
| Delay cell traversal | 11,088 fJ | ~346 active cells × 32 fJ/cell |
| Race detection | 210 fJ | 42 neurons × 5 fJ/comparator |
| **Total** | **12,354 fJ (12.4 pJ)** | |

**Active fraction** (~25% of branches fire per inference): inputs with T_in outside the [T_fastest, T_fastest + 3τ] window contribute negligible energy to the nLSE operation.

Digital baseline (8-bit MAC, 28 nm, Horowitz 2014): ~1,386 pJ → **112× better energy efficiency.**

### Training (SPSA, per epoch, MNIST)

| Component | Energy | % of total |
|---|---|---|
| 3× forward passes | ~44 µJ | 50% |
| 3× P&V programming | ~27 µJ | 30% |
| TDC readout | ~18 µJ | 20% |
| **Total per epoch** | **~88 µJ** | |

Full MNIST training (200 epochs): ~17.6 mJ.

---

## 9. Training: SPSA + Pulse-and-Verify (P&V)

Training requires **no backpropagation** and is compatible with direct hardware execution.

### 9.1 Simultaneous Perturbation Stochastic Approximation (SPSA)

Each mini-batch update requires exactly **3 forward passes**:

```
1. Sample random perturbation vector Δ (each element ∈ {+1, −1})
2. L0 = loss(weights)           [baseline forward pass]
3. L+ = loss(weights + ε·Δ)    [perturbed + forward pass]
4. L− = loss(weights − ε·Δ)    [perturbed − forward pass]
5. Update: u[i] -= η · (L+ − L−) / (2ε) · Δ[i]   for all i
```

Gradient estimation noise is tamed by using mini-batches of 128 samples per pass. The 3-pass cost is **independent of the number of weights** — unlike backprop which scales with depth.

### 9.2 Pulse-and-Verify (P&V) RRAM Programming

Each weight update is applied to the RRAM device using iterative P&V:

```
1. Apply SET or RESET pulse (E_pulse ≈ 10 fJ each)
2. Read current resistance via TDC (500 fJ per read)
3. Compare to target delay
4. If |d_current − d_target| > tolerance: repeat
5. Typical cycles: ~20 pulses per cell to converge
```

**P&V energy per cell:** 20 pulses × 10 fJ = 200 fJ (programmable in ~200 ns).

### 9.3 TDC (Time-to-Digital Converter)

Used during training to read back actual delay values for P&V verification and loss computation.

- **Type:** Flash TDC with 150 ns range, 1 ns resolution
- **Comparator count:** ~150 per readout channel
- **Energy per conversion:** ~500 fJ (conservative 28 nm estimate)
- **Usage:** One reading per output neuron per forward pass; not needed during inference

---

## 10. Key Physical Constraints and Tolerances

| Constraint | Requirement | Basis |
|---|---|---|
| σ_j (timing jitter) | < 1 ns | Simulation: robust at σ_j/τ < 0.1 |
| RRAM conductance ratio | ≥ 10× | Required for full d_min/d_max range |
| RRAM cycle endurance | ≥ 10⁶ cycles | ~20 P&V pulses × 50,000 weight updates |
| RRAM retention | ≥ 10 years | Standard RRAM spec at 85°C |
| κ calibration tolerance | ±20% | nLSE is robust to uniform κ drift |
| C_cell matching | ±5% | Mismatch absorbed by weight recalibration |
| d_min/d_max ratio | ≥ 10× | One decade conductance swing required |

---

## 11. Sensitivity to Physical Parameters

Inference energy scales with C_cell and V_dd²:

| C_cell | V_dd | E_inference | vs 8-bit MAC |
|---|---|---|---|
| 10 fF | 0.5 V | 684 fJ | 2025× |
| 10 fF | 0.8 V | 1,424 fJ | 973× |
| 100 fF | 0.8 V | 12,354 fJ | **112×** (nominal) |
| 100 fF | 1.2 V | 27,534 fJ | 50× |
| 500 fF | 1.2 V | 136,830 fJ | 10× |

The design retains significant advantage over digital MACs across the full technology range. At C_cell = 10 fF (sub-5 nm node), the advantage exceeds 1000×.

---

## 12. Block Diagram Summary

```
                     ┌─────────────────────────────────────────────┐
  Input              │  LAYER 0 (65 × 32 branch pairs)             │
  pixels ──encode──► │                                             │
  (64 + 1 bias)      │  For each of 32 neurons:                    │
                     │    T_in[i] ─── d_pos[i,j] ──► ─┐           │
                     │    T_in[i] ─── d_neg[i,j] ──►  ├─ race    │
                     │    (×65 pairs)                   └─ detect  │
                     │                                   T_out[j]  │
                     └────────────────── 32+1 ──────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  LAYER 1 (33 × 10 branch pairs)             │
                     │                                             │
                     │    T_h[i] ─── d_pos[i,j] ──► ─┐            │
                     │    T_h[i] ─── d_neg[i,j] ──►  ├─ race     │
                     │    (×33 pairs)                   └─ detect  │
                     │                               margin[j]     │
                     └─────────────────────────────────────────────┘
                                              │
                                    softmax(margin / τ_d)
                                              │
                                        class prediction
```

---

## 13. Fabrication Notes

- **Process node:** 28 nm or sub-28 nm CMOS with embedded RRAM back-end
- **RRAM back-end:** 1T1R cell structure (one transistor, one resistor) for access control
- **Delay cells:** Fixed C_cell implemented as MOM capacitor in metal routing layers
- **Race detectors:** Clocked sense amplifiers, one per output neuron per layer
- **TDC:** Flash TDC, shared across output channels (time-multiplexed during training)
- **P&V controller:** On-chip state machine; no off-chip communication during training
- **Digital control:** Minimal — only P&V sequencer and SPSA perturbation LFSR

The design is intentionally free of large digital datapaths. The only digital logic is a pseudorandom number generator (for SPSA Δ vectors) and a simple counter-based P&V controller.
