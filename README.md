# Creating Neural Network Models of Types of Representation Drift to Explain Experimental Data

Figures produced for the masters thesis: *Creating neural network models of types of representation drift to explain experimental data*.

## Data Requirement

Experimental data from Driscoll et al. must be downloaded from Dryad and placed in the `RuleData/` directory before running the notebooks:

> Driscoll et al. — https://datadryad.org/dataset/doi:10.5061/dryad.gqnk98sjq

## Running the Notebooks

Each notebook is self-contained. On first run it executes simulations and caches results in `data/`; subsequent runs load from cache. Figures are saved as PDFs to `saved_figures/`.

---

## Figures

### Figure 1.1 — Baseline Drift (`fig1_1_baseline_drift.ipynb`)

A four-panel illustration of the physical reality of representational drift in the rate RNN model. Panel **A** shows the structured excitability matrix *E(t)* that transiently boosts different neuron groups on different days. Panel **B** shows the identical single-context input stimulus delivered every day. Panel **C** shows snapshots of the recurrent weight matrix at the end of each simulated day, demonstrating Hebbian redistribution. Panel **D** is a firing-rate heatmap across the full simulation, where day-to-day shifts in the active sub-population constitute the representational drift characterised in subsequent figures.

---

### Figure 1.2 — Parametric Characterisation of Drift (`fig1_2_combined.ipynb`)

A three-panel summary of how drift depends on model parameters. Panel **A** is a heatmap of cumulative drift rate Δ across the excitability amplitude *E* × synaptic volatility *σ* parameter space. Panel **B** shows PCA trajectories of population activity across four mechanistic conditions (no drift, excitability only, volatility only, combined), illustrating the distinct geometric signatures of each mechanism. Panel **C** plots Hebbian learning rate *η* against synaptic weight variance and drift rate, showing that faster learning amplifies both weight fluctuations and population-code instability.

---

### Figure 1.2.1 — Drift Rate Heatmap (`fig1_2_1_heatmap.ipynb`)

Standalone version of the *E* × *σ* drift-rate heatmap. Each cell shows the mean cumulative drift rate Δ (averaged over three random seeds), with the canonical operating point marked by a star.

---

### Figure 1.3.1 — Correlation Fingerprint (`fig1_3_1_correlation_fingerprint.ipynb`)

A grid of histograms (5 day-separations × 3 drift conditions) showing the distribution of scaled pairwise correlation change ΔC = (C₂ − C₁)/C₂ for neuron pairs that were correlated on the reference day. A two-component Gaussian mixture is fitted to each distribution. The shape of these distributions — particularly the relative weight of the high-ΔC component — provides a mechanistic fingerprint that distinguishes excitability-driven, volatility-driven, and combined drift.

---

### Figure 2.1.1 — Subspace Angle (`fig2_1_1_subspace_angle.ipynb`)

Principal angles between the day-1 neural manifold and the manifold on each subsequent day, computed for the top ten PCA dimensions. Each coloured trace is one principal-angle dimension (blue = largest / most rotated, red = smallest); the dashed black line is the mean across all dimensions. Rising angles confirm that the population-code geometry genuinely rotates over days, approaching orthogonality.

---

### Figure 2.2/2.3 — Functional Consequences of Drift (`fig2_2_3_combined.ipynb`)

Two panels examining how drift degrades downstream readout. Panel **A** tracks the accuracy of a linear readout frozen after day 1 over subsequent days at the canonical operating point, showing progressive degradation. Panel **B** maps the functional memory lifespan (number of days the frozen readout stays above 75% accuracy) across the full *E* × *σ* parameter space, revealing which parameter regimes support sustained memory despite drift.

---

### Figure 2.4.4 — Drift-Compatible Memory (`fig2_4_4_drift_compatible_memory.ipynb`)

Demonstrates that representational drift and stable context discrimination are compatible in the two-context model. Panel **A** plots two geometric metrics across days: within-context drift (rising dashed line — the population code genuinely drifts) and cross-context separation (flat solid line near 90° — the two memory subspaces stay orthogonal). Panel **B** shows that the frozen day-1 linear readout achieves near-perfect A-vs-B classification on every day because the cross-context geometry is preserved.

---

### Figure 3.6.1 — Parameter Selection (`fig3_6_1_parameter_selection.ipynb`)

A single heatmap showing the combined log-ratio fit score across the *E* × *σ* grid, computed by summing the log-ratio scores from the population-vector-correlation (PVC) MSE and the subspace-overlap MSE fits to the Driscoll et al. data. The best-fit parameter combination is marked with a star; the canonical model parameters are marked separately.

---

### Figure 3.7.1 — Model Validation (`fig3_7_1_combined.ipynb`)

A four-panel validation of the model against experimental data from Driscoll et al. (2020). Panel **A** compares the model's population vector correlation (PVC) decay across days with individual-mouse curves from Driscoll et al. Panel **B** shows the PVC mean-squared-error fit score as a heatmap over the parameter grid. Panel **C** compares coding subspace overlap trajectories between model and data. Panel **D** shows the subspace-overlap MSE fit score heatmap. Together, panels B and D identify which parameter combinations best reproduce both aspects of the experimental drift statistics.
