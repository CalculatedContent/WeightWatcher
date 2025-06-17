# WeightWatcher QFT Module Documentation

## Overview

The Quantum Field Theory (QFT) module in WeightWatcher implements theoretical foundations from quantum field theory and renormalization group theory to analyze neural network weight matrices. This module is based on the theory that neural networks approach a critical point during training, which can be described as a kind of fractal where the free energy satisfies scale invariance, according to Wilson's exact renormalization group theory.

## Theoretical Background

### Critical Points in Neural Networks

Neural networks exhibit behavior analogous to physical systems near critical points. At these critical points:

1. **Scale Invariance**: Statistical properties remain unchanged across different scales
2. **Power Law Distributions**: Eigenvalue/singular value distributions follow power laws
3. **Long-Range Correlations**: Correlations decay as power laws rather than exponentially
4. **Fractal Structure**: Self-similarity emerges across different scales

### Wilson's Renormalization Group Theory

The module implements concepts from Wilson's exact renormalization group theory:

1. **Renormalization Flow**: Tracking how weight distributions evolve during training
2. **Fixed Points**: Identifying stable, unstable, and critical fixed points in parameter space
3. **Critical Exponents**: Measuring universal quantities that characterize phase transitions
4. **Free Energy Landscape**: Mapping the thermodynamic properties of weight matrices

## Key Features

### 1. Critical Point Analysis

```python
from weightwatcher import RGAnalyzer

# Initialize the analyzer
rg_analyzer = RGAnalyzer()

# Analyze a weight matrix
results = rg_analyzer.analyze_critical_point(W)
```

This method analyzes how close a weight matrix is to a critical point by:
- Estimating power law exponents from eigenvalue distributions
- Measuring scale invariance properties
- Determining if the matrix is near a critical point

### 2. Fractal Dimension Analysis

```python
# Compute fractal dimension
fractal_metrics = rg_analyzer.compute_fractal_dimension(W)
```

This method:
- Estimates the fractal dimension using a box-counting approach
- Quantifies self-similarity across scales
- Provides metrics on the fractal properties of weight matrices

### 3. Free Energy Landscape Mapping

```python
# Map free energy landscape
energy_metrics = rg_analyzer.map_free_energy_landscape(W)
```

This method:
- Calculates free energy based on eigenvalue spectrum
- Computes entropy and energy
- Identifies critical points in the free energy landscape
- Measures specific heat capacity to detect phase transitions

### 4. Renormalization Group Flow Tracking

```python
# Track RG flow across training epochs
for epoch, weights in enumerate(weight_history):
    metrics = rg_analyzer.track_rg_flow(weights, epoch=epoch)

# Visualize the flow
fig = rg_analyzer.visualize_rg_flow()
```

This functionality:
- Tracks how weight matrices evolve during training
- Visualizes the approach to criticality
- Identifies when networks enter or leave critical regions

### 5. Phase Transition Detection

```python
# Detect phase transitions between epochs
transition = rg_analyzer.detect_phase_transition(W_before, W_after)
```

This method:
- Detects significant changes in network behavior
- Classifies transitions (to critical, from critical, non-critical)
- Quantifies the magnitude of transitions

### 6. Correlation Length Analysis

```python
# Analyze correlation length
corr_metrics = rg_analyzer.analyze_correlation_length(W)
```

This method:
- Estimates correlation length, which diverges at critical points
- Calculates correlation decay exponents
- Provides another measure of criticality

### 7. Universality Class Classification

```python
# Classify into universality classes
univ_class = rg_analyzer.compute_universality_class(W)
```

This method:
- Classifies networks into known universality classes from statistical physics
- Provides confidence scores for classifications
- Helps understand universal properties of neural networks

## Practical Applications

### 1. Architecture Design

The QFT module can guide architecture design by:
- Identifying optimal initialization strategies that place networks near criticality
- Suggesting layer sizes and connectivity patterns that promote scale invariance
- Recommending regularization techniques that maintain critical behavior

### 2. Training Optimization

The module helps optimize training by:
- Detecting when networks move away from critical regions
- Identifying phase transitions that might indicate training issues
- Suggesting learning rate adjustments based on RG flow

### 3. Generalization Analysis

The module provides insights into generalization by:
- Correlating critical behavior with generalization performance
- Identifying universality classes that tend to generalize better
- Suggesting modifications to improve generalization based on QFT principles

## Integration with Traditional WeightWatcher

The QFT module complements traditional WeightWatcher metrics:

| Traditional Metric | QFT Metric | Relationship |
|-------------------|------------|--------------|
| Alpha (power law) | Power Law Exponent | Directly related, but QFT provides theoretical foundation |
| Stable Rank | Scale Invariance | Both measure effective dimensionality, but from different perspectives |
| MP Fit | Free Energy | Both measure deviation from random matrices |
| Layer Norm | Correlation Length | Both relate to the conditioning of weight matrices |

## Example Usage

```python
import numpy as np
from weightwatcher import RGAnalyzer

# Initialize the analyzer
rg_analyzer = RGAnalyzer(temperature=1.0)

# Generate a synthetic weight matrix
np.random.seed(42)
W = np.random.normal(0, 1, (1000, 500))

# Comprehensive analysis
results = rg_analyzer.track_rg_flow(W)

# Check if the matrix is near a critical point
if results['is_critical']:
    print("The weight matrix is near a critical point")
else:
    print("The weight matrix is far from criticality")
    
# Print key metrics
print(f"Power Law Exponent: {results['power_law_exponent']:.4f}")
print(f"Scale Invariance: {results['scale_invariance']:.4f}")
print(f"Fractal Dimension: {results['fractal_dimension']:.4f}")
print(f"Free Energy: {results['free_energy']:.4f}")
```

## References

1. Wilson, K.G. (1971). "Renormalization Group and Critical Phenomena"
2. Martin, C.H. & Mahoney, M.W. (2019). "Traditional and Heavy-Tailed Self Regularization in Neural Network Models"
3. Sornette, D. (2006). "Critical Phenomena in Natural Sciences"
4. Mehta, P. & Schwab, D.J. (2014). "An exact mapping between the Variational Renormalization Group and Deep Learning"
5. Roberts, D.A., Yaida, S., & Hanin, B. (2021). "The Principles of Deep Learning Theory"
