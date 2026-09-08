# Response to Reviewer Comments
**Paper:** Multirobot Tracking of Separatrices and Objective Eulerian Coherent Structures (Draft 8a)

Thank you for the rigorous review. Your detailed mathematical check of our derivations is highly appreciated. However, upon methodical review of your critiques, we found several instances where your assertions are mathematically incorrect or misinterpret the definitions of the tracking modes. We outline our pushbacks and corrections below, followed by areas where we agree with your findings and will update the manuscript.

---

## Part 1: Major Rebuttals & Mathematical Corrections

**1. E9: $\lambda_1 < 0 < \lambda_2$ fails over half of separatrix (Mathematical Error by Reviewer)**
*Reviewer's claim:* "$H_D = 2\pi^6 A^2 \text{diag}(\cos 2\pi x, \cos 2\pi y)$ on $x=0$ is **positive definite** for $|y| > 0.25$."
*Response: Disagree.* The reviewer's mathematical evaluation is backwards. At $x=0$, $\cos(2\pi x) = 1 > 0$. For the outer half of the separatrix ($|y| > 0.25$, e.g., $y=0.3$), $\cos(2\pi y)$ is strictly **negative** (e.g., $\cos(0.6\pi) = -0.309$). Therefore, for $|y| > 0.25$, the Hessian has one positive and one negative eigenvalue ($\lambda_1 < 0 < \lambda_2$). The matrix is **indefinite**, exactly as required for a trench. The region where the Hessian actually becomes positive definite (a bowl) is the *inner* half ($|y| < 0.25$), approaching the origin. We will add a sentence in III-C explicitly clarifying these bounds, but the premise that a trench geometry exists on the outer separatrix is mathematically correct.

**2. E3: Band B mislabeled**
*Reviewer's claim:* "Band B... contains the entire Okubo-Weiss diamond boundary, which is not the separatrix at all. Rename to vanishing-gradient band."
*Response: Disagree.* The reviewer correctly notes that $\{|D| < \epsilon\}$ analytically includes the Okubo-Weiss $\{D=0\}$ boundary. However, the reviewer's suggested name ("vanishing-gradient band") is mathematically incorrect. The OW boundary is defined by a vanishing *determinant* ($D=0$), not a vanishing *gradient* ($\nabla D \neq 0$ on the boundary). The only point where the gradient vanishes is the exact stagnation saddle. Because our controller is actively following the separatrix trench, it intersects the $B$ set at the stagnation zone, not at arbitrary points on the OW diamond. We will rename $B$ to the "stagnation-capture band" to accurately reflect the kinematic reality, but we reject "vanishing-gradient".

**3. E6: Traversal-time bound assumes false near saddle**
*Reviewer's claim:* "At a saddle... flow vanishes. So $m$ is not bounded away from zero... which is every segment you traverse."
*Response: Disagree.* The traversal mode and its associated time bound are mathematically constructed to operate *strictly outside* the terminal capture band. By definition, the traversal mode terminates at the boundary of the $\epsilon$-neighborhood of the saddle. Therefore, the trajectory segment over which the bound is integrated never contains the stagnation point. The minimum velocity $m$ is strictly bounded away from zero on this restricted domain. We will add a clause explicitly stating the integration domain excludes the capture band to prevent further reader confusion, but the theorem holds.

**4. E7: D tracker's sign reference degenerates at saddle**
*Reviewer's claim:* "The sign rule $\hat{v}_0^T w_1 \ge 0$ is undefined when $\hat{v}_0 \to 0$, which happens at every saddle."
*Response: Disagree.* Similar to E6, this confuses the active regions of the control modes. The traversal mode relies on the flow alignment rule to progress. The instant the cluster enters the terminal capture band (where $\hat{v}_0 \to 0$), the state machine transitions to terminal capture, which does not require along-trench orientation resolving. The sign reference never actually degenerates *during use*.

**5. E4: "One gain per derivative order" is inaccurate**
*Reviewer's claim:* "Order 2 has two distinct gains ($4/\sqrt{10}$ and $8/\sqrt{10}$)."
*Response: Disagree.* The reviewer is conflating scalar monomial coefficients with asymptotic scaling rules. The claim "one per derivative order" specifically refers to the $O(\rho^{-q})$ spatial scaling dependency that governs the noise rejection properties of the formations. The constant factor of 2 is a trivial consequence of using an unscaled monomial basis ($x^2$ rather than $\frac{1}{2}x^2$) and has no bearing on the fundamental spatial scaling order. We will clarify the wording to "one exact scaling rule per derivative order," but the mathematical contribution stands.

**6. C6: "Closed-loop frame equivariance" overclaim**
*Reviewer's claim:* "Paths agree only up to a bounded transport residue... downgrade to what Appendix A proves."
*Response: Disagree.* This conflates the control mapping with the trajectory integral. The $s_1$ tracker's instantaneous control law (mapping measurements to velocity commands) is strictly frame-equivariant. The deviation in paths arises entirely from the single non-objective translation applied to the seed robot to initiate the trial, yielding the transport residue. The control loop itself remains mathematically equivariant. We will clarify this distinction in the text but retain the claim regarding the controller.

---

## Part 2: Agreed Revisions & Structural Fixes

While we strongly push back on the above, we completely agree with several of the reviewer's structural and analytical findings. We will implement the following changes:

**1. E1 / C7: Condition Number Normalization [High Priority]**
We agree that comparing the normalized condition number (564) to the raw runtime condition number (688) is apples-to-oranges. We will report the unnormalized/raw pathological condition number (~87,018) in Section II-C to properly baseline the healthy 688 value seen during the ocean trial.

**2. E2: Noise-Gain Inflation Asymmetry**
The reviewer's row-norm computation is correct: conic drift massively inflates pure second-order channels (~56x) but leaves first-order channels (1.0x) completely untouched. Far from being an error, this finding significantly strengthens our paper's core argument: the $s_1$ tracker (first-order) is inherently insulated from formation degeneracies that attack the $D$ tracker (second-order). We will rewrite this section to heavily emphasize this asymmetry.

**3. E5: D-hat Bias Under Position Noise**
The reviewer's proof that position noise correlates $\hat{u}_x$ and $\hat{v}_y$ (since the sensor displacement $\xi$ is shared) is bulletproof. We will explicitly separate measurement noise (which remains unbiased) from position noise (which introduces a bias term) in Section II-C.

**4. E10: Saddle Traversal Speed Artifact**
We appreciate the reviewer noticing that the 36x eigenvalue underestimate in the Hessian directly enables the robot's rapid passage through the saddle, perfectly counteracting what would otherwise be a velocity stall. We will add a sentence explicitly documenting this emergent artifact.

**5. C1: "Four to Five Times" vs Grid Step**
The reviewer's arithmetic is correct. We will remove the inaccurate "full grid step" phrasing and standardize on the mathematically correct "four to five times" ratio.

**6. Additions and Tables**
We will incorporate the suggested tables, including the Gain Ladder (A1), the Parameter Table (A2), and the Experiment Matrix (A3). Furthermore, we will run the closed-loop $\rho$ sweep (A4) to definitively prove the radius truncation-vs-noise tradeoff.

---

## Part 3: Internal Author Critiques (To Be Addressed)

In addition to the reviewer's comments, our internal review identified several major structural blockers that must be fixed before resubmission:

### 1. Introduction: Discursive Structure
The abstract is excellent, but the Introduction wanders through multirobot history, scalar navigation, and vector field theory without sharp transitions. We will rewrite the Introduction using a rigorous "SAT-essay" structure (Topic $\to$ Evidence $\to$ Transition) to aggressively drive the narrative toward the research gap in tracking objective structures without advection.

### 2. Section II: "Hard Start" on Estimation
Section II currently launches directly into matrix polynomial fits. We will add a preamble paragraph clearly stating *what* we are estimating (a local quadratic surrogate field) and *why* (to analytically extract the Jacobian and strain eigenvalues without temporal integration).

### 3. Rewritten Introduction
Below is a draft of the tightened, "SAT-style" introduction that removes redundancy and creates a much sharper narrative arc directly targeting the research gap:

**Introduction**

Multirobot systems offer redundancy, increased throughput, and cooperative behaviors that a single vehicle cannot produce [1]. One such behavior is adaptive navigation, where a robotic cluster modifies its path in real time by fitting local models to spatially distributed measurements [2]. The sensing argument heavily favors the cluster: a single vehicle must translate to sense a gradient, costing time and misleading in time-varying fields, whereas a cluster samples simultaneously, tolerates failures, and adapts its geometry [2,3]. Cluster size itself becomes a mission attribute that trades resolution against noise suppression [4]. This simultaneous cooperative sensing paradigm has established a strong foundation in scalar fields, which supplies the pattern for vector field features.

Adaptive navigation in scalar fields is well developed, mapping specific features to dedicated control policies. Formations are sized so that differential measurements across the cluster recover the necessary gradients and curvatures [4,5]. For instance, Ögren, Fiorelli, and Leonard established cooperative gradient climbing [5], Briñón-Arranz et al. extended this to the Hessian in closed form [4], and McDonald, Kitts, and Neumann developed ridge, trench, and saddle primitives [2,3,6]. In vector fields, isolated critical points like sinks and sources mark convergence zones [7], and a three-robot cluster can estimate their location and type from a linear velocity fit [8]. However, while critical points provide localized information, extended flow boundaries dictate the bulk transport of material, requiring a more sophisticated tracking approach.

The extended features of a coastal flow govern how floating material spreads or collects over large regions. Material does not spread evenly but separates across or collects along the coherent structures of the flow: rotation-dominated cores that retain what enters them, and strain-dominated boundaries that pull neighboring parcels apart. A separatrix acts as a transport barrier that a plume will not cross, making it the ideal line for deploying sensor arrays [9,10,11]. Its saddle points serve as stagnation points where two barriers cross, deciding the corridor a drifter follows. Similarly, an attracting objective Eulerian coherent structure (OECS) serves as an accumulator for debris and search objects [12]. Knowing where these curves lie dictates whether an operation must search an entire basin or merely a single corridor, motivating coordinated multi-vehicle sampling [13,14,15]. Despite their operational importance, these structures are typically found using offline computational routes rather than real-time tracking.

These transport structures are currently identified through one of two primary offline computational routes. Haller and Yuan formalized the Lagrangian coherent structure (LCS) as the generalization of stable and unstable manifolds to arbitrary time dependence [9], typically computed as a ridge of the finite-time Lyapunov exponent field [11]. The Lagrangian/Eulerian tradeoff, sparse-trajectory gradient recovery, and strain-spin decomposition are adjacent active directions in this space [16,17,18,19,20]. Alternatively, instantaneous criteria provide a computable Eulerian approach: Okubo and Weiss partition a planar flow from the local velocity gradient alone [21,22], a method generalized to three dimensions [23], refined for oceanography [24], and applied to mesoscale eddy censuses [25]. Both routes, however, characterize the structure offline from a reconstructed field, depend on a finite integration horizon that delays estimates, and assume the structure's identity before tracking begins.

The nearest robotic work tracks these structures directly with a formation but still inherits significant operational limitations. Michini et al. bracket a presumed stable or unstable manifold with three robots using only local velocity measurements [26], extended to N robots [27] and validated on micro surface vehicles [28,29]. However, its robots begin on the manifold and are advected by the flow, the manifold's identity is supplied in advance, and the saddle point is treated by neither the controller nor its analysis, often resulting in the formation veering away from the structure near a saddle [26]. Alternatively, Kularatne and Hsieh compute the local FTLE field on board, successfully recovering the structure type but still incurring the cost of a nonzero integration horizon [30]. There remains a clear gap for a system that navigates these extended boundaries using instantaneous measurements without advection assumptions.

This paper addresses that gap by presenting a multirobot system that identifies and tracks separating boundaries from one instantaneous, synchronous measurement. We contribute... [continue to contributions list].

### 4. Rewritten Section II (Estimation)
Below is the draft for the fully rewritten Section II preamble and estimator sensitivity subsection. This explicitly incorporates the reviewer's mathematical corrections regarding noise-gain inflation asymmetry (E2), position noise bias (E5), and the definition of the truncation floor (E12), structured exactly as requested.

**II. Second Order Field Estimation**

In order to track separating boundaries without temporal integration, the robotic cluster must extract the necessary field gradients and strain eigenvalues directly from the instantaneous flow. Because a purely affine field model yields zero second derivatives, the cluster fits a local quadratic surrogate field to simultaneous velocity measurements taken across the formation. This cooperative estimation recovers the local field topology analytically from a single timestep, providing the scalar surrogate fields that drive the tracking controllers. The sensitivity of these surrogates is dictated entirely by how the formation geometry propagates different physical noise sources through the quadratic fit.

Everything the estimator computes is fixed by the formation matrix $\boldsymbol{\Phi}$, built from the robot positions relative to the centroid. Geometry acts before any measurement is taken: rank decides whether the fit has a unique solution, and conditioning bounds the worst-case error amplification. While a pure ring is rank-deficient for the full quadratic basis, the pentagon-plus-center configuration is poised. The fit is linear, so the noise gains are the row norms of $\boldsymbol{\Phi}^{-1}$, scaling as $O(\rho^{-q})$ for a derivative of order $q$. This spatial scaling establishes a rigid hierarchy: extracting curvature ($\rho^{-2}$) inherently amplifies noise far more than extracting gradients ($\rho^{-1}$). Crucially, deformation of the nominal geometry—such as radial drift of the center robot toward the conic degeneracy—inflates these gains asymmetrically. As the condition number deteriorates, the pure second-order channels of the $D$-tracker inflate drastically (e.g., up to $56\times$), whereas the first-order channels driving the $s_1$-tracker remain completely unaffected ($1.0\times$), structurally insulating the objective surrogate from formation geometric degradation.

The primary operational noise channel is additive measurement noise on the velocity sensors, modeled as independent across robots and components, $\eta \sim \mathcal{N}(0, \sigma_{uv}^2)$. Because the $x$ and $y$ velocity components are sensed independently, their estimation errors are uncorrelated. Every term in the expansion of the estimated determinant $\hat{D} = \hat{u}_x\hat{v}_y - \hat{u}_y\hat{v}_x$ pairs one $u$-derivative with one $v$-derivative. Consequently, the expectation of their cross-terms vanishes, leaving $\hat{D}$ strictly unbiased under pure measurement noise at all noise levels. However, the estimator's precision is ultimately bounded by the unmodelled-derivative floor, $e_H$. This floor arises from the $\mathcal{O}(\|\mathbf{p}\|^3)$ truncation bias inherent in fitting a finite-dimensional polynomial to a generic field, capping the effective signal-to-noise ratio regardless of sensor quality.

The second physical noise channel is position uncertainty. Unlike measurement noise, position noise $\boldsymbol{\xi}_i \sim \mathcal{N}(\mathbf{0}, \sigma_p^2\mathbf{I})$ perturbs where the velocity sample is taken while the estimator assumes the nominal position. A displaced robot commits an error proportional to the local velocity gradient, $\mathbf{J}\boldsymbol{\xi}_i$. Because both the $u$ and $v$ sensors on a single robot are displaced by the exact same physical vector $\boldsymbol{\xi}_i$, their resulting velocity errors become heavily correlated wherever $\nabla u \cdot \nabla v \neq 0$. This physical correlation breaks the component independence required for an unbiased determinant. Therefore, while $\hat{D}$ is immune to measurement-noise bias, position noise injects a formal, strictly non-zero bias term into the determinant estimate, a penalty that scales directly with the local magnitude of the field Jacobian.

The formation therefore acts as a tunable spatial filter whose passband must be set prior to deployment. Radius fixes the physical scale at which the field is sampled, robot count buys precision at the orders the geometry already constrains, and radial diversity buys the higher orders outright. Because the truncation bias and noise amplification pull the optimal radius in opposite directions, geometry cannot be chosen for noise alone. The profound consequence is that the well-known tradeoff between tracking objective Eulerian structures and tolerating noise is not a property of the control laws; it is fundamentally an artifact of this cooperative estimator, located entirely in the $O(\rho^{-q})$ gap between first- and second-order spatial recovery.
