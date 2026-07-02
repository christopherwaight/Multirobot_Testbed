# Research Findings and Analysis

Research findings and experimental results **excluded from current papers** but valuable for future work and design context.

---

## 1. Dual Jacobian Analysis

**Status:** EXCLUDED from Paper_Draft_6.tex  
**Reason:** 4-robot dual Jacobian performs worse than simpler 3-robot configuration in most conditions.

### Key Findings

**The 3-Robot Configuration Wins:**
| Config                       | Error @ r=0.1m | Notes |
|------------------------------|----------------|-------|
| 3-robot (α=0)                | 0.031m | Stable, no saturation |
| 4-robot dual (constrained)   | 0.012m | 100% velocity saturated |
| 4-robot dual (unconstrained) | 0.109m | Unstable |

The 3-robot approach is simpler, more stable, doesn't require velocity saturation, and handles real-world disturbances better.

**Momentum Filtering Hurts Performance:**
| Config         | With Momentum (α=0.7) | No Momentum (α=0) |
|----------------|----------------------|-------------------|
| 3-robot        | 0.108m | 0.031m (71% better) |
| 4-robot square | 0.108m | 0.031m (71% better) |
| 4-robot dual   | 0.012m | 0.007m (40% better) |

Setting α=0 consistently improves performance across all configurations.

**The Dual Jacobian Instability:**
The algorithm manages three competing objectives (convergence, formation control, orientation) that create destructive interference. Velocity saturation accidentally stabilizes it by acting as implicit hierarchical control—but when given freedom, it oscillates.

**Why 3-Robot and 4-Robot Square Perform Identically:**
Both use the same orbital control algorithm. The 4th robot in a symmetric square adds no information.

### Paper Decision

Paper focuses on 3-robot methods. The real contributions from this analysis:
- Momentum filtering harms performance
- Multi-objective control conflicts cause instability  
- Constraints can stabilize unstable algorithms
- Simplicity beats complexity in distributed control

---

## 2. Scalar Field Findings

### Formation Rotation is Essential

| Metric                    | With Rotation (kr=0.3) | Without |
|---------------------------|------|---------|
| Success rate              | 100% | 60% |
| Path length (favorable)   | ~14m | ~3m |
| Path length (unfavorable) | ~14m | Diverges |

**Critical Discovery—The Gain Dead Zone:**

| Gain Range     | Success Rate |
|----------------|--------------|
| kr = 0.0       | 60% (no control) |
| kr = 0.005–0.1 | 0–40% (**worse than none!**) |
| kr = 0.2–0.5   | 100% (robust) |
| kr = 0.3       | Optimal |

Low gains (0.02–0.1) actually degrade performance below having no rotation control at all.

**Value Proposition:** Rotation control is "insurance"—sacrifices efficiency on easy cases to enable convergence from otherwise impossible configurations.

---

## 3. Sim2Real Comparison

### Agreement Summary
- **Convergence:** Simulation predicts within 5mm; hardware achieves 4mm accuracy (excellent agreement)
- **Orbital:** Simulation predicts ±8%; hardware achieves ±6% (good agreement)

### Key Discrepancies
| Issue               | Simulation | Hardware | Cause |
|---------------------|------------|----------|-------|
| Center drift        | Stable     | 1–2cm over 30s | Sensor calibration accumulation |
| Formation shape.    | Perfect    | ±2cm variation | Individual tracking errors |
| Velocity saturation | Rare       | 15–20% of time | Conservative safety limits |

**Important:** Simulation is conservative—hardware often performs better than predicted.

---

## 4. Experimental Protocol Summary

### 3-Robot Vortex (278 experiments)
- 100% convergence success, 96% orbital success
- Average final error: 4mm
- Average convergence time: 12s

### 3-Robot Saddle (148 experiments)  
- 100% success rate
- Average final error: 6mm
- Average convergence time: 18s (harder due to unstable equilibria)

### 4-Robot Tests
Limited scope; did not show significant advantages over 3-robot.

---

## 5. Future Work Considerations

### Algorithmic Extensions
| Direction                  | Current State   | Future Potential |
|----------------------------|-----------------|------------------|
| Time-varying fields        | Memoryless (instantaneous) | Prediction models, oscillating/drifting fields |
| 3D extension               | 2D planar       | ≥10 robots for 3×3 Hessian, tetrahedral formations |
| Multi-cluster coordination | Single cluster  | Multiple clusters, task allocation, distributed mapping |
| Adaptive formations        | Fixed shapes    | Dynamic sizing based on field curvature |
| Energy awareness           | None            | Battery-aware planning, optimal formation shapes |
| Uncertainty quantification | Point estimates | Confidence bounds, risk-aware navigation |

### Hardware Improvements
- **Sensors:** Real flow sensors (anemometers, current meters), thermal, chemical, multi-modal
- **Control rate:** 10 Hz → 50–100 Hz for faster response
- **Workspace:** 1.6×1.6m arena → outdoor/ocean/atmospheric scales

### Algorithmic Improvements
- Robust Jacobian estimation with outlier rejection
- Automatic field type classification from eigenvalues
- Coarse-to-fine multi-scale estimation

---

**Last Updated:** December 17, 2025