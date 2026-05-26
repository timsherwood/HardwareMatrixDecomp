# Tier-2 PCB Prototype — Engineering Build Guide

**Status:** Simulation-validated, ready for fabrication  
**Purpose:** Bench-level proof of the memristive delay-cell neural network using real Knowm SDC
memristors, 1 nF RC delay cells, and a BJT exponential-transconductance sense amplifier  
**Demo task:** XOR classification (2-input, 1-output) — 18 memristors total  
**Reference simulation:** `memristor/tier2/` — every number in this document is drawn
directly from that code

---

## 1. What This Prototype Proves

The ASIC concept in `HARDWARE_SPEC.md` uses 100 fF capacitors and 5–50 ns delays, which
are not measurable on a bench. The Tier-2 prototype scales timing by 10,000× using 1 nF
capacitors, giving 50–500 µs delays and µs-resolution timing that any oscilloscope or
microcontroller can capture.

The architecture is identical. Every result generalises directly to the ASIC:

| | ASIC | Tier-2 PCB |
|---|---|---|
| C_cell | 100 fF | **1 nF** |
| Delay range | 5–50 ns | **50–500 µs** |
| R range | 50–500 kΩ | **50–500 kΩ** (same) |
| κ (geometric mean) | 15.81 ns | **158.1 µs** |
| V_DD (RC ramp) | 0.8 V | **1.5 V** |
| V_th (threshold) | — | **0.75 V = V_DD/2** |
| τ_sense (sense amp) | ~1 ns | **~110 µs** |
| Training algorithm | SPSA | **SPSA** (same) |
| Log-conductance weights u | same | **same** |

**Scaling law:** d_PCB = d_ASIC × 10,000. Weight matrices (u_pos, u_neg) are identical
between the two; only κ changes.

---

## 2. Two-Phase Build Plan

**Phase 1 — Digital-assisted prototype (this document)**  
A Teensy 4.1 microcontroller replaces the analog sense amplifier. The µC measures RC
threshold crossings in hardware, computes the nLSE soft-min in firmware, and drives
downstream cells. This approach validates memristor P&V, delay-cell timing, weight
parameterisation, and end-to-end XOR classification with zero analog design risk.

**Phase 2 — Analog BJT sense amplifier (add-on board)**  
A separate small board plugs into the Phase 1 board and replaces the µC comparator path
with a physical BJT exponential-transconductance circuit. The simulation in
`memristor/tier2/sense_amp.py` defines the exact target behaviour. Section 9 documents
the circuit requirements.

Build and verify Phase 1 first. Phase 2 is a straightforward add-on once Phase 1
classification is confirmed.

---

## 3. XOR Network — Device Count

```
Architecture:  2 inputs  →  [2 hidden neurons]  →  1 output

Layer 0 (input → hidden):
  Input vector:  x0, x1, bias_0  (3 signals)
  Each hidden neuron races:
    Positive race  =  3 delay cells  (d_pos[0..2, j])
    Negative race  =  3 delay cells  (d_neg[0..2, j])
  Two hidden neurons  →  2 × 6 = 12 delay cells

Layer 1 (hidden → output):
  Input vector:  h0, h1, bias_1  (3 signals)
  One output neuron:
    Positive race  =  3 delay cells
    Negative race  =  3 delay cells
  → 6 delay cells

Total:  18 delay cells = 18 Knowm SDC memristors + 18 × 1 nF capacitors
```

**Weight matrix layout** (for PCB labelling):

```
Layer 0 — d_pos[i,j]:      Layer 0 — d_neg[i,j]:
  i=0 (x0):   D00P, D01P     i=0 (x0):   D00N, D01N
  i=1 (x1):   D10P, D11P     i=1 (x1):   D10N, D11N
  i=2 (bias): D20P, D21P     i=2 (bias): D20N, D21N

Layer 1 — d_pos[i,0]:      Layer 1 — d_neg[i,0]:
  i=0 (h0):   E00P            i=0 (h0):   E00N
  i=1 (h1):   E10P            i=1 (h1):   E10N
  i=2 (bias): E20P            i=2 (bias): E20N
```

Each cell Dxy{P,N} consists of one Knowm SDC memristor + one 1 nF cap.

---

## 4. Delay Cell Circuit

```
3.3 V ──────────────────────────────┐
                                    │
                               M_reset (BSS138)
                                    │  ← GPIO_RESET_n (active-LOW to discharge)
         Knowm SDC                  │
3.3 V ──[  R_mem  ]──────┬──────────┘
                         │
                       C_cell (1 nF, C0G)
                         │
                        GND

Node voltage:  V_RC(t) = V_DD × (1 − exp(−(t − T_in) / d))    for t > T_in
                       = 0                                       reset state

Threshold crossing time (at V_th = 1.65 V = V_DD/2):
  t_cross = T_in + d × ln(2)     where  d = R_mem × C_cell
```

**Equations implemented in `memristor/tier2/delay_cell.py`:**
- `delay_us = R_ohm × C_cell_nF × 1e-3`  (line 57)
- `threshold_crossing_us = T_in + delay_us × ln(V_DD / (V_DD − V_th))`  (line 67)

**Reset sequence:** Pull GPIO_RESET_n HIGH → turns on M_reset → discharges C_cell to 0 V.
Timing starts when GPIO_RESET_n goes LOW.

**Delay range:**
- R_min = 50 kΩ → d_min = 50 kΩ × 1 nF = 50 µs
- R_max = 500 kΩ → d_max = 500 kΩ × 1 nF = 500 µs
- κ = √(50 × 500) = 158.1 µs → initial device R ≈ 158 kΩ

---

## 5. Comparator Circuit (Phase 1)

```
V_RC ────┤+                  ├──── COMP_OUT (to Teensy GPIO, 3.3 V logic)
         │  LM339 (1 of 4)   │
V_th ────┤−                  │
(1.65 V)  └──────────────────┘
               ↕ 10 kΩ pull-up to 3.3 V (open-drain output)
```

Reference voltage: 1.65 V from a resistor divider (2× 10 kΩ from 3.3 V to GND) with a
100 nF bypass cap. For better stability, use a dedicated 1.65 V reference such as the
LM4040 (see BOM).

Each comparator output goes directly to a Teensy digital input. The Teensy captures the
rising edge timestamp with a hardware interrupt (1 µs or better resolution using its
600 MHz internal timers).

---

## 6. Firmware Architecture (Teensy 4.1)

The firmware maps exactly to the Python simulation in `memristor/tier2/network.py`.

```
Setup:
  - 18 GPIO outputs for reset switches (GPIO_RESET_00 … GPIO_RESET_17)
  - 18 GPIO inputs with interrupt-on-rising-edge for comparator outputs
  - USB serial for communication with host PC

Inference pass (one XOR input vector [x0, x1]):
  1. Encode inputs (delay_cell.py line 159):
       T_in[x0]    = (x0 > 0.5) ? 0 µs : 1500 µs
       T_in[x1]    = (x1 > 0.5) ? 0 µs : 1500 µs
       T_in[bias]  = 0 µs   (always)

  2. Release delay cells at their encoded times:
       t=0:     release all cells with T_in=0 µs (active inputs + bias)
       t=1500µs: release cells with T_in=1500 µs (inactive inputs)
       Use TeensyTimerTool for µs-accurate delayed release

  3. Record threshold crossing times via interrupt handlers:
       When COMP_OUT_ij rises, record t_cross[i][j]

  4. Compute nLSE soft-min for each race (network.py line 504):
       T_plus[j]  = nLSE(t_cross_pos[:, j], tau_us=100.0)
       T_minus[j] = nLSE(t_cross_neg[:, j], tau_us=100.0)
       margin[j]  = T_minus[j] - T_plus[j]

  5. Hidden layer output timing (network.py line 199):
       p[j]       = sigmoid(margin[j] / tau_d_us=50.0)
       T_hidden[j] = p[j]*T_plus[j] + (1-p[j])*T_minus[j]

  6. Repeat steps 1-4 for Layer 1 using T_hidden as input

  7. Output: p_out = sigmoid(margin_out / 50.0)
     Prediction: (p_out > 0.5) ? 1 : 0

nLSE formula (numerically stable, network.py line 504-510):
  log_terms = -t_cross / tau
  lse = max(log_terms) + log(sum(exp(log_terms - max(log_terms))))
  return -tau * lse

P&V programming (device.py lines 69-114):
  Host PC sends target delay d_target_us over USB serial.
  Teensy applies SET (low-voltage forward pulse) or RESET (reverse pulse)
  to the target memristor, measures resulting d = R×C via an RC timing
  measurement, and repeats until |d - d_target| ≤ 5 µs or 100 pulses.

SPSA training loop (network.py train_spsa_bjt, line 347):
  Runs on host PC in Python using the simulation as oracle,
  with hardware measurements substituted for simulated BJT calls
  once Phase 2 sense amp is attached.
```

Reference implementation: `memristor/tier2/network.py` is the exact algorithmic
specification. Port each function directly to Teensy C++.

---

## 7. Bill of Materials

### 7.1 Memristors

| Ref | Description | Qty | Supplier | Part / Notes |
|---|---|---|---|---|
| M1–M18 | Knowm SDC memristor, 50–500 kΩ | 18 | Knowm Inc | **W2x2 SDC** — order from [knowm.org/inc](https://knowm.org/inc). Request through-hole or SMT-mountable form. Confirm current SKU with Knowm sales (they sell in kits and bulk). Resistance range 50–500 kΩ is a hard requirement — verify with datasheet. Order 25+ for yield margin. |

**Knowm SDC device model** is implemented in `memristor/tier2/device.py`.
Key parameters from that file:
- R_min = 50,000 Ω, R_max = 500,000 Ω (lines 41-42)
- step_mean_ln = 0.10 (~10% ΔlnR per pulse, line 43)
- noise_frac = 0.15 (cycle-to-cycle variability, line 44)
- P&V success rate ≥ 90% within 100 pulses (verified in `test_tier2_device.py`)

### 7.2 Load Capacitors (Delay Cell)

| Ref | Description | Qty | MPN | Supplier |
|---|---|---|---|---|
| C1–C18 | 1 nF, 50 V, **C0G/NP0**, 0402 | 18 + 10 spare | **GRM1555C1H102JA01D** | Murata / DigiKey [490-1504-1-ND] or Mouser [81-GRM1555C1H102JA1D] |

**Critical:** Must be C0G (NP0) dielectric. X7R and Y5V have voltage and temperature
coefficients that will cause delay drift and measurement error. C0G is stable to ±30 ppm/°C.

Ordering note: GRM1555 is 0402 metric (1.0 × 0.5 mm). If hand-soldering, use
GRM2165C1H102JA01D (0805, same spec) instead — easier to place.

### 7.3 Reset Switches (one per delay cell)

| Ref | Description | Qty | MPN | Supplier |
|---|---|---|---|---|
| Q1–Q18 | N-ch MOSFET, SOT-23, V_GS(th) ≤ 1.5 V | 18 + 5 spare | **BSS138** | Nexperia / DigiKey [BSS138CT-ND] or Mouser [771-BSS138-T/R] |

BSS138 turns on fully at 3.3 V gate (V_GS(th) max = 1.5 V at 1 mA). Sufficient to
discharge 1 nF in ≪ 1 µs with R_DS(on) ≈ 1 Ω.

### 7.4 Threshold Comparators (Phase 1)

| Ref | Description | Qty | MPN | Supplier |
|---|---|---|---|---|
| U1–U5 | Quad comparator, SOIC-14, open-drain | 5 | **LM339DR** | TI / DigiKey [296-1393-1-ND] or Mouser [595-LM339DR] |
| R_pull | 10 kΩ, 1%, 0402 pull-up resistors | 20 | RC0402FR-0710KL | Yageo / any distributor |

5× LM339DR gives 20 comparators; 18 used, 2 spare. Each output is open-drain — the 10 kΩ
pull-up to 3.3 V is required to get a logic-level signal to the Teensy.

LM339 input range: 0 V to V_CC − 1.5 V. At V_CC = 3.3 V, valid input up to 1.8 V.
This is fine: V_RC_max in use is ~1.65 V (V_th). Above V_th the comparator fires;
V_RC continues charging past 1.65 V but the comparator output is already latched.

Alternative if faster response is needed: **TLV3201DBVR** (7 ns prop delay, SOT-23-5,
single, TI, DigiKey 296-12504-1-ND) — use 18 of them instead of the quad LM339.

### 7.5 Reference Voltage

| Ref | Description | Qty | MPN | Supplier |
|---|---|---|---|---|
| U6 | 1.65 V precision voltage reference, SOT-23 | 1 | **LM4040C16IDBZR** | TI / DigiKey 296-12811-1-ND |
| C_bypass | 100 nF, 16 V, X5R, 0402 | 2 | GRM155R61C104KA88D | Murata |

The LM4040C16 provides a stable 1.638 V (≈ 1.65 V, 0.5% accuracy) reference for all 18
comparators. Connect CATHODE to the V_th rail; connect ANODE to GND.
Bypass cap on V_th rail: 100 nF to GND.

### 7.6 Microcontroller

| Ref | Description | Qty | Supplier | Notes |
|---|---|---|---|---|
| MCU1 | **Teensy 4.1** | 1 | PJRC ([pjrc.com/teensy41](https://www.pjrc.com/store/teensy41.html)) | 600 MHz ARM Cortex-M7, 55 digital I/O, 3.3 V logic, USB serial. ~$30 |

The Teensy 4.1 has sufficient pins for 18 reset GPIOs + 18 comparator input GPIOs + USB
serial. Its hardware interrupt latency is < 100 ns; more than adequate for µs-resolution
crossing time measurement. Install with the standard Arduino + Teensyduino toolchain.

### 7.7 Power Supply and Passives

| Ref | Description | Qty | MPN | Supplier |
|---|---|---|---|---|
| U7 | 3.3 V, 500 mA LDO, SOT-23-5 | 1 | **MIC5205-3.3YM5-TR** | Microchip / DigiKey MIC5205-3.3YM5TRCT-ND |
| C_in | 1 µF, 10 V, X5R, 0402 input cap | 1 | GRM155R61A105KE15D | Murata |
| C_out | 1 µF, 10 V, X5R, 0402 output cap | 1 | GRM155R61A105KE15D | Murata |
| J1 | USB Micro-B connector (or use Teensy's built-in USB) | — | — | Power from PC USB |

**Supply rails needed:**
- 3.3 V: all logic, reset FETs, comparators, RC ramp V_DD, reference
- GND: single ground plane

The board can be powered entirely from PC USB (5 V in → 3.3 V LDO). The Teensy 4.1
has its own onboard 3.3 V regulator; you may be able to power the external circuits
from the Teensy's 3V3 pin (250 mA limit — check total current load).

### 7.8 Programming Pulse Interface (Memristor P&V)

Knowm SDC devices require bipolar voltage pulses for SET (forward) and RESET (reverse).
Typical requirements (confirm with Knowm datasheet):
- SET pulse: +1.5 V to +2 V, 1–10 ms duration
- RESET pulse: −1.5 V to −2 V, 1–10 ms duration

The Teensy cannot generate negative voltages. Use one of:

**Option A (simplest):** Texas Instruments **DRV8871** H-bridge motor driver (SOIC-8,
$1.60, DigiKey 296-42316-1-ND). Two GPIO pins select direction (SET vs RESET); one
PWM pin controls pulse duration. One DRV8871 can drive all memristors in sequence if
they share a bus with individual address enables.

**Option B:** Dedicated ±5 V supply (e.g. Texas Instruments **TPS65131** ±5 V boost/
inverter DCDC, $3.50) + SPDT analog mux (SN74HC4066) for each device to steer the
polarity. More complex but gives cleaner pulse edges.

For initial bring-up, Option A is recommended.

---

## 8. PCB Layout Recommendations

**Form factor:** Single 4-layer PCB, roughly 100 × 80 mm, USB-powered.

**Layer stack:**
1. Component / signal
2. GND (solid pour)
3. Power (3.3 V pour)
4. Component / signal (bottom)

**Critical layout rules for the delay cells:**

1. Place each memristor (M_i) directly adjacent to its load capacitor (C_i). Keep the
   trace from memristor drain to capacitor top plate ≤ 3 mm to minimise parasitic
   capacitance (target < 100 fF stray; the 1 nF load dominates at < 10% error).

2. Route all capacitor bottom plates (and M_reset source pins) directly to the GND
   pour with short vias. Shared ground impedance across delay cells would cause
   crosstalk.

3. The V_th reference rail (1.65 V) should be routed as a low-impedance trace
   (≥ 0.5 mm wide) from the LM4040 to all comparator V− pins. Bypass at each
   comparator pin with 10 nF (GND pour via close to the pin).

4. Keep comparator outputs short and away from V_RC traces to avoid noise injection
   into the RC network.

5. Place Teensy near the edge of the board. Route the 36 GPIO traces (18 reset + 18
   comparator) as a short bus; matched length is not required but keep traces ≤ 50 mm.

6. Decoupling: 100 nF 0402 C0G cap at each LM339 V_CC pin and each BSS138 gate. A
   10 µF bulk cap (electrolytic or polymer tantalum) at the 3.3 V entry point.

---

## 9. Phase 2 — Analog BJT Sense Amplifier Circuit

Once Phase 1 validates the delay cells and P&V, replace the comparator + firmware nLSE
path with this physical circuit. The simulation `memristor/tier2/sense_amp.py` is the
exact specification.

### Physical circuit (one sense amplifier = one race for one output neuron)

```
                        V_CC = 1.5 V
                             │
                          R_load
                             │
                             ├──────────────────────── T_fire output
                             │                         (to comparator or latch)
                  ┌──────────┤
                  │          │          │           │
               I_col[0]   I_col[1]   I_col[2]     ...
                  │          │          │
               [Q0 NPN]  [Q1 NPN]  [Q2 NPN]       (N_in branches)
                  │          │          │
               V_RC[0]    V_RC[1]    V_RC[2]       (RC delay cell outputs)
                  │          │          │
                 GND        GND        GND
```

All collector currents sum at the R_load node. When Σ I_col reaches a threshold set
by R_load, the node voltage drops and a comparator fires. The firmware timestamp
from Phase 1 is replaced by this physical firing event.

### gain_A implementation (gain_A = 20 in simulation)

The simulation parameter `gain_A = 20` implements BJT exponential-transconductance
with effective thermal voltage (gain_A × V_T = 0.52 V). This stretches the firing
window from ~26 mV to ~520 mV, giving τ_sense ≈ 110 µs at nominal delay (see
`sense_amp.py` line 96):

```python
tau_sense = gain_A * V_T * d / (V_DD - V_th)
          = 20 * 0.026 * 158e-6 / 0.75  ≈  110 µs
```

The simplest circuit implementation is a **voltage-to-voltage scaling stage** before
each BJT base:

```
V_RC ──[19R]──┬──[1R]──GND
              │
              └──── V_BE = V_RC × 1/20

V_BE_eff = V_RC / 20   ⟹   I_C ∝ exp(V_RC / (20 × V_T))
```

Using a resistor divider (19:1 ratio, e.g. 190 kΩ + 10 kΩ) scales V_RC down by 20
before the BJT base. Choose BJTs with low V_BE(on): the maximum V_BE applied is
1.5 V / 20 = 75 mV, so the transistors operate in deep subthreshold (< V_BE(on)).

For deep subthreshold operation, a **MOSFET is more suitable** than a BJT at these
voltages. Use the **ALD1106** (N-ch enhancement MOSFET in SOIC-8, Advanced Linear
Devices, available at DigiKey) — these are zero-threshold MOSFETs explicitly designed
for subthreshold circuits with V_th(on) < 0.3 V. One package contains 4 matched FETs.

**Practical component for Phase 2 sense amp:**

| Ref | Description | Qty | MPN | Notes |
|---|---|---|---|---|
| QA1–QA6 | Quad matched zero-V_th MOSFET, SOIC-8 | 6 | **ALD1106** | Advanced Linear Devices, DigiKey |
| R_div_hi | 190 kΩ, 1%, 0402 (×18) | 18 | CRCW0402190KFKED | Vishay |
| R_div_lo | 10 kΩ, 1%, 0402 (×18) | 18 | CRCW040210K0FKED | Vishay |
| R_load | 100 kΩ, 1%, 0402 (×6) | 6 | CRCW0402100KFKED | Vishay |
| U_comp | Fast comparator for T_fire detection (same TLV3201 as Phase 1) | 6 | TLV3201DBVR | TI |

**Gain_A constraint (from simulation, sense_amp.py comment line 34):**
```
gain_A < V_th / (V_T × ln(N_max))
gain_A < 0.75 / (0.026 × ln(3)) = 26.3
```
gain_A = 20 is chosen to satisfy this with comfortable margin. Do not increase gain_A
above 26 for a 3-input neuron without re-running the BJT training simulation.

**Validation test for the Phase 2 sense amp** (from `test_tier2_sense_amp.py`):
- Single branch: fire time within ±2 × τ_sense of analytic t_cross = T_in + d × ln(2)
- Two simultaneous identical branches: fire τ_sense × ln(2) ≈ 76 µs earlier than one
- Three simultaneous: fire τ_sense × ln(3) ≈ 121 µs earlier than one
- (Exact formula: advance = d × ln(1 + (τ_sense/d) × ln(N)) — see test line 119)

---

## 10. Bring-Up and Test Procedure

### Step 1 — Power-on check

1. Connect USB. Verify 3.3 V on the power rail. Check LM4040 output = 1.638 V ± 10 mV.
2. Load a Teensy sketch that drives all 18 reset GPIOs HIGH (hold all caps at 0 V),
   then reads all 18 comparator inputs. All should read LOW (comparator V+ < V_th).
3. Set all reset GPIOs LOW (release all caps). All 18 V_RC nodes begin charging.
   Within ~500 ms (≈ 3 × d_max = 3 × 500 µs × wait-for-full-charge) all comparators
   should go HIGH. Verify on an oscilloscope: V_RC rises from 0 V to 3.3 V
   asymptotically with time constant d = R_mem × 1 nF.

### Step 2 — Single delay cell characterisation

For each of the 18 cells (automate via Teensy + Python serial):

1. Hold cell in reset (GPIO HIGH).
2. Release reset → record comparator rising edge timestamp t_cross.
3. Check: t_cross = d × ln(2) + T_in. With T_in = 0 and d = R_mem × 1 nF:
   - At R_nom = 158 kΩ: t_cross ≈ 158 µs × 0.693 ≈ 109.5 µs
   - Range: 50 µs × 0.693 = 34.7 µs (R_min) to 500 µs × 0.693 = 346.6 µs (R_max)
4. Derive measured R = t_cross / (C × ln(2)) = t_cross / (1e-9 × 0.693).
5. Log R for all 18 cells — initial values should scatter around R_nom.

Expected variation: ±15% (noise_frac = 0.15 in the device model).

### Step 3 — P&V programming test

For each cell, run the P&V algorithm from `device.py` (ported to Teensy firmware):

```
Target: d_target = 158 µs (κ), tolerance ±5 µs
Expected: convergence in < 50 pulses, ≥ 90% success rate across 18 cells
```

SET pulse: drive the memristor forward (+V polarity) with ≈ 1 V amplitude for 5 ms.
RESET pulse: reverse polarity, same amplitude and duration. Adjust pulse amplitude and
duration based on Knowm's programming guide for the SDC device.

After P&V: all 18 cells should read d = 158 ± 5 µs. Verify with the single-cell
characterisation method in Step 2.

### Step 4 — XOR classification (all-software path first)

Before running on real hardware, confirm the Python simulation matches expectations:

```bash
cd /path/to/HardwareMatrixDecomp
uv run python scripts/sim_tier2.py
```

Expected output: XOR convergence in ≤ 3000 epochs (analytical SPSA), all 4 patterns
correct. See `test_tier2_network.py::TestTier2NetworkSPSA` for pass criteria.

### Step 5 — XOR on hardware

1. Run SPSA training in Python (`Tier2Network.train_spsa`, `network.py` line 272).
   This computes target weights (u_pos, u_neg) for all 18 cells.

2. Convert weights to target delays:
   `d_target = kappa_us × exp(−u)` — all targets will be in [50, 500] µs.

3. Program each cell via P&V to its target delay (Step 3 procedure).

4. Run inference: apply each of the 4 XOR patterns ([0,0], [0,1], [1,0], [1,1]),
   record crossing times, compute nLSE and margin in firmware, report prediction.

5. Expected: all 4 patterns classified correctly. Reference test:
   `test_tier2_network.py::TestTier2NetworkSPSA::test_xor_all_four_patterns_correct`.

---

## 11. Expected Results Summary

All numbers are direct outputs of the simulation. Running `uv run pytest
memristor/tests/test_tier2_*.py -v` before assembly confirms everything the hardware
should reproduce.

| Test | File | Expected result |
|---|---|---|
| d = R × C | `test_tier2_delay_cell.py::test_delay_formula_nominal` | d = 158.1 µs at R_nom |
| Crossing time | `test_tier2_delay_cell.py::test_crossing_at_half_vdd_is_d_ln2` | t_cross = d × 0.693 |
| P&V success rate | `test_tier2_device.py::test_pv_success_rate_high` | ≥ 90% in 100 pulses |
| P&V polarity | `test_tier2_device.py::test_pv_polarity_set_when_delay_too_long` | SET → shorter delay |
| XOR convergence | `test_tier2_network.py::test_spsa_converges_on_xor` | 100% in ≤ 3000 epochs |
| XOR patterns | `test_tier2_network.py::test_xor_all_four_patterns_correct` | [0,1,1,0] |
| 5% noise robustness | `test_tier2_network.py::test_trained_network_robust_to_small_device_noise` | ≥ 80% trials correct |
| BJT τ_sense formula | `test_tier2_sense_amp.py::test_tau_sense_formula` | 110.1 µs at d = κ |
| BJT N-branch advance | `test_tier2_sense_amp.py::test_multiplicity_advance_approx_tau_ln_n` | within ±1 µs of exact |

---

## 12. Condensed BOM (Order List)

| Item | MPN | Qty | Approx unit cost | Supplier |
|---|---|---|---|---|
| **Knowm SDC memristor** | W2x2 SDC (contact Knowm) | 25 | ~$5–15 | knowm.org/inc |
| Cap 1 nF C0G 0402 | GRM1555C1H102JA01D | 30 | $0.10 | DigiKey / Mouser |
| MOSFET reset switch BSS138 | BSS138 SOT-23 | 25 | $0.10 | DigiKey / Mouser |
| Quad comparator LM339 | LM339DR SOIC-14 | 5 | $0.35 | DigiKey / Mouser |
| Pull-up 10 kΩ 0402 | RC0402FR-0710KL | 30 | $0.01 | any |
| V_ref 1.65 V | LM4040C16IDBZR SOT-23 | 2 | $0.90 | DigiKey |
| Bypass cap 100 nF 0402 | GRM155R61C104KA88D | 20 | $0.05 | Murata / DigiKey |
| H-bridge (P&V pulses) | DRV8871DDAR SOIC-8 | 2 | $1.60 | DigiKey |
| **Teensy 4.1** | Teensy 4.1 | 1 | $29.85 | pjrc.com |
| 3.3 V LDO | MIC5205-3.3YM5-TR SOT-23-5 | 2 | $0.55 | DigiKey |
| PCB fabrication (100×80 mm, 4L) | — | 5 | ~$5 | JLCPCB / OSHPark |
| **Phase 2 only:** ALD1106 zero-Vth MOSFET | ALD1106SBLF SOIC-8 | 6 | $3.00 | Advanced Linear Devices / DigiKey |
| **Phase 2 only:** 190 kΩ 1% 0402 | CRCW0402190KFKED | 20 | $0.05 | Vishay / DigiKey |
| **Phase 2 only:** 10 kΩ 1% 0402 | CRCW040210K0FKED | 20 | $0.05 | Vishay / DigiKey |
| **Phase 2 only:** 100 kΩ 1% 0402 | CRCW0402100KFKED | 10 | $0.05 | Vishay / DigiKey |
| **Phase 2 only:** Fast comparator | TLV3201DBVR SOT-23-5 | 10 | $0.70 | TI / DigiKey |

**Estimated total (Phase 1):** ≈ $65 in parts + PCB + Teensy  
**Estimated total (Phase 1 + 2):** ≈ $100

---

## 13. Key Simulation Files — Quick Reference

| File | What it models | Key functions |
|---|---|---|
| `memristor/tier2/delay_cell.py` | RC delay cell physics | `delay_us`, `threshold_crossing_us`, `waveform` |
| `memristor/tier2/device.py` | Knowm SDC stochastic switching, P&V | `set_pulse`, `reset_pulse`, `program_to_delay` |
| `memristor/tier2/sense_amp.py` | BJT exponential-transconductance sense amp | `fire_time_us`, `tau_sense_us`, `nLSE_us` |
| `memristor/tier2/network.py` | Full XOR network, SPSA training | `forward`, `forward_bjt`, `train_spsa_bjt` |
| `scripts/sim_tier2.py` | End-to-end demo script | `main()` — run for a full system walkthrough |
| `memristor/tests/test_tier2_*.py` | Acceptance tests for each block | Run with `uv run pytest memristor/tests/test_tier2_*` |

To run the complete simulation before building:
```bash
uv run python scripts/sim_tier2.py        # full system demo
uv run pytest memristor/tests/test_tier2_network.py -v   # all 17 network tests
```
