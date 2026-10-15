# CMA-ES: Covariance Matrix Adaptation Evolution Strategy

*A practical guide to black-box continuous optimization*

Based on the supplied tutorial text.

---

## Slide 1 — CMA-ES

**Covariance Matrix Adaptation Evolution Strategy**

A practical guide to black-box continuous optimization.

---

## Slide 2 — The optimization problem

### Objective

Minimize an unknown objective function:

\[
\min_{x \in \mathbb{R}^n} f(x)
\]

- **Candidate solution \(x\):** a vector of continuous decision variables.
- **Black-box evaluator \(f(x)\):** returns a scalar score; gradients may be unavailable or unhelpful.
- **Search budget:** commonly measured by the number of objective-function evaluations.

### Typical applications
- Shape optimization
- Model calibration
- Parameter calibration

---

## Slide 3 — Why continuous optimization is difficult

- **Rugged or multimodal landscapes:** many local optima, discontinuities, or noise.
- **High dimensionality:** the search volume grows rapidly with the number of variables.
- **Non-separability:** variables interact, so optimizing each variable independently may fail.
- **Ill-conditioning:** some directions are much more sensitive than others.
- **Non-smooth level sets:** sharp corners can challenge standard CMA-ES.
- **Black-box access:** reliable gradients or exploitable analytic structure may be unavailable.

---

## Slide 4 — Where CMA-ES fits

### A strong choice when
- The objective is continuous and treated as a black box.
- Gradients are unavailable, noisy, or not useful.
- Variables interact or the problem is rotated.
- The search landscape may be rugged or ill-conditioned.

### Why use it?
CMA-ES learns the search geometry from the quality and ranking of sampled candidates.

### Important caveat
If an objective has exploitable analytic structure—for example, a well-behaved convex form—a specialized method may be more efficient.

---

## Slide 5 — The CMA-ES search loop

1. **Sample:** draw \(\lambda\) candidate solutions from the current distribution.
2. **Evaluate:** compute \(f(x)\) for each candidate.
3. **Rank and select:** prefer the best \(\mu\) candidates.
4. **Update distribution:** move the mean and adapt the step size and covariance.
5. **Repeat or stop:** continue until a stopping condition is met.

CMA-ES is a stochastic, population-based method: candidate rankings guide the next distribution.

---

## Slide 6 — The search distribution: three key parameters

Candidates are sampled according to:

\[
x \sim m + \sigma \,\mathcal{N}(0,C)
\]

- **Mean \(m\):** location of the current favorite region.
- **Step size \(\sigma\):** overall scale of exploration.
- **Covariance matrix \(C\):** shape, orientation, and variable dependencies of the search distribution.

The covariance matrix controls the axes and rotation of the distribution's equal-density ellipsoid.

---

## Slide 7 — Why a multivariate normal distribution?

- **Flexible geometry:** represents round, stretched, or rotated search regions.
- **Compact description:** the mean and covariance describe location and shape.
- **Weak assumptions:** sampling does not require a specific analytic form for the objective function.

The covariance matrix's eigenvectors set the principal directions; its eigenvalues determine the squared axis lengths of equal-density ellipsoids.

In higher dimensions, Gaussian samples tend to lie near a shell rather than close to the mean.

---

## Slide 8 — Update 1: move the mean toward better candidates

### Weighted recombination

The new mean is a weighted average of the \(\mu\) best candidates.

- Better-ranked candidates receive larger positive weights.
- The weights sum to 1.
- The mean shifts toward promising regions.

### Ranking-based update

CMA-ES uses candidate rankings rather than the absolute scale of objective values.

---

## Slide 9 — Update 2: adapt the global step size \(\sigma\)

### Cumulative step-size adaptation (CSA)

1. Track a smoothed path of successive mean movements.
2. Compare its length with the expected length under random selection.
3. **Longer path:** increase \(\sigma\).
4. **Shorter path:** decrease \(\sigma\).

### Intuition
- If \(\sigma\) is too small, progress can be slow because candidates stay close together.
- If \(\sigma\) is too large, search can overshoot or lose local precision.

CSA aims to balance exploration scale with progress.

---

## Slide 10 — Update 3: learn the covariance matrix \(C\)

The covariance matrix adapts the search shape.

- Stretch along directions that repeatedly produce successful steps.
- Rotate the search region to match variable interactions.
- Represent dependencies between pairs of variables.
- Approximate second-order geometry without calculating gradients.

The search ellipsoid changes scale and orientation to follow the problem's geometry.

---

## Slide 11 — Covariance updates: complementary mechanisms

### Rank-one update
Uses an evolution path—accumulated movement—to learn persistent directions efficiently.

### Rank-\(\mu\) update
Uses several selected candidates to estimate successful-step covariance; useful with larger populations.

### Active update
Uses poor candidates to reduce variance in unpromising directions. Safeguards are needed because negative weights can threaten positive definiteness of \(C\).

Together, these mechanisms can learn long axes, use population information, and suppress unhelpful directions.

---

## Slide 12 — A major strength: invariance

### Monotonic objective transformations
If objective values are transformed monotonically, candidate rankings remain the same. CMA-ES therefore follows the same search decisions.

### Rotation and coordinate changes
A full covariance matrix can adapt to rotated, non-separable landscapes instead of favoring the original coordinate axes.

### Why it matters
- The algorithm can behave similarly across a broad class of rescaled objective values.
- Covariance learning is central to handling variable dependencies and ill-conditioning.

---

## Slide 13 — Using CMA-ES on your problem

1. **Define \(f(x)\):** return one scalar score for every candidate.
2. **Set the initial mean \(x_0\):** use a reasonable initial guess when available.
3. **Choose the initial step size \(\sigma_0\):** match it to plausible variable changes.
4. **Set stopping rules:** examples include evaluation budget, target value, axis length, or stagnation.

Default hyperparameters are usually a sensible starting point. Prioritize problem-specific initialization and termination settings first.

---

## Slide 14 — Multimodal objectives: use restarts strategically

### Randomized restarts
Start from different initial points and use a relatively small initial step size to encourage different outcomes.

### Increase population size
A larger population can reveal global structure on some difficult multimodal functions, but costs more evaluations per iteration.

### Two restart strategies

- **IPOP-CMA-ES:** restarts with increasing population size.
- **BIPOP-CMA-ES:** combines increasing-population restarts with many smaller local restarts.

---

## Slide 15 — Scaling to high dimensions

### The computational bottleneck

A full covariance matrix has \(O(n^2)\) storage and update costs, with additional cost to decompose it for sampling.

### Restricted variants

- **Sep-CMA-ES:** uses a diagonal covariance matrix.
- **VD-CMA and LM-CMA, and related variants:** use restricted models of variable dependencies.

### Trade-off

Restricted covariance models can reduce computation and speed adaptation in suitable problems. A diagonal covariance loses rotational invariance and can perform worse on non-separable problems.

---

## Slide 16 — Implementation workflow

The supplied tutorial uses Python and **pycma**.

1. Install the `pycma` package.
2. Import the `cma` module.
3. Provide the objective function, initial mean \(x_0\), and initial step size \(\sigma_0\).
4. Set options such as target fitness and maximum function evaluations.
5. Inspect convergence and evaluation statistics.

### Example from the tutorial
The source demonstrates optimizing a variable-dimensional Rosenbrock function.

Use plots and stopping diagnostics to understand progress. Do not rely on a single run for multimodal tasks. Implementation details may vary by pycma version.

---

## Slide 17 — Key takeaways

1. **CMA-ES samples:** candidates come from a multivariate normal distribution.
2. **The mean moves:** ranked candidates guide the next center.
3. **The step size adapts:** cumulative path length controls global search scale.
4. **The covariance adapts:** covariance learning handles rotation, dependencies, and ill-conditioning.
5. **The user configures the run:** initialization, stopping criteria, and restart strategy shape practical results.

**Best mental model:** CMA-ES learns both *where to search* and the *geometry of how to search*.
