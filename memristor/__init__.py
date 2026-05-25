"""Memristor-calibrated log-delay network for temporal neural computation.

Physical model:  d = kappa / G = kappa * exp(-u)

where G is memristor conductance, u = ln(G) is log-conductance, and kappa
is a per-cell calibration constant encoding capacitance, threshold, supply
voltage, and layout parasitics.

Training rule (spec Section 4):
    d_target = d * exp(-eta * lambda * d)
    u_new    = ln(kappa / d_target)

where lambda = dL/dd.  This is equivalent to gradient descent on u
(since d ln(G)/dd = -1/d, so grad_u = -lambda*d), meaning standard
backprop through the temporal forward pass with SGD on u implements
the spec's local rule exactly.

Milestones (spec Section 22):
    1 – Behavioral XOR model (this module, Simulation 1)
    2 – Quantized + jitter + programming noise model (Simulation 2–4)
    3+– Hardware-in-the-loop and MNIST target
"""

from memristor.delay_cell import DelayCell
from memristor.energy import (
    HardwareSpec,
    InferenceEnergy,
    TrainingEnergy,
    compare_to_digital,
    estimate_inference_energy,
    estimate_training_energy,
)
from memristor.gradient_analysis import (
    extract_delay_gradients,
    gradient_active_fraction,
    gradient_summary,
)
from memristor.hil_training import HILTrainer
from memristor.network import ComplementaryDelayLayer, DelayLayer, MemristorNet
from memristor.noise import NoisyMemristorNet
from memristor.quantization import make_quantized_net, quantize_complementary, quantize_delays
from memristor.training import MemristorTrainer

__all__ = [
    "ComplementaryDelayLayer",
    "DelayCell",
    "DelayLayer",
    "HILTrainer",
    "HardwareSpec",
    "InferenceEnergy",
    "MemristorNet",
    "MemristorTrainer",
    "NoisyMemristorNet",
    "TrainingEnergy",
    "compare_to_digital",
    "estimate_inference_energy",
    "estimate_training_energy",
    "extract_delay_gradients",
    "gradient_active_fraction",
    "gradient_summary",
    "make_quantized_net",
    "quantize_complementary",
    "quantize_delays",
]
