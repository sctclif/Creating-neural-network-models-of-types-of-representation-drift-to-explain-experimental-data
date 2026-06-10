"""
Core RNN model for representational drift.
Single-context, single-readout. Based on combined_model_readout.ipynb.

Key implementation notes:
- Threshold θ is evolved (Eq. 3) but NOT subtracted in drdt — matches notebook behaviour (dead code).
- Single context: all N neurons receive identical input amplitude.
- Recurrent weights clamped to [0, 1] via conditional update mask.
- Euler integration, dt = 1 ms.
"""
import os

import numpy as np
from dataclasses import dataclass, replace as _dc_replace


@dataclass
class ModelParams:
    # Network
    N: int = 50
    # Simulation
    nstep: int = 13000
    dt: float = 1.0
    # Firing rate dynamics
    taur: float = 50.0
    # Global inhibition (Eq. 1)
    I0: float = 5.0
    I1: float = 1.0
    I2: float = 0.05
    # Recurrent weight plasticity (Eq. 4)
    tauw: float = 1000.0
    decay: float = 1000.0
    # Threshold adaptation (Eq. 3 — tracked but not used in drdt)
    tautheta: float = 800.0
    y0_scale: float = 10.0
    # Input stimulus
    IN: float = 15.0
    Nstim: int = 10
    stim: int = 200
    duration: int = 100
    pause: int = 1000
    delay: int = 3000
    Nevent: int = 4
    # Excitability fluctuations
    E: float = 1.5
    E_neuron_divisions: int = 5
    E_time_divisions: int = 4
    E_sigma: float = 1.0
    # Synaptic volatility
    vol_mean: float = 0.0
    vol_std: float = 0.02
    # Readout neuron
    tau_out_plus: float = 200.0
    tau_out_minus: float = 1000.0
    # Reproducibility
    seed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_excitability_matrix(p: ModelParams, rng: np.random.RandomState):
    """
    Structured excitability: neuron group i is elevated during period j when i == j+1,
    plus a neuron-specific absolute-normal baseline (exc0).
    """
    exc0 = np.abs(rng.normal(0, p.E_sigma, p.N))
    Emat = np.zeros((p.N, p.nstep))
    nd = p.E_neuron_divisions
    td = p.E_time_divisions
    nr = p.N // nd
    nt = p.nstep // td
    for i in range(nd):
        for j in range(td):
            if i == j + 1:
                Emat[i * nr:(i + 1) * nr, j * nt:(j + 1) * nt] = p.E
    Emat += exc0[:, np.newaxis]
    return Emat, exc0


def build_stimulus_sequence(p: ModelParams):
    """Return list of (on, off) pulse boundary times for all days."""
    seq = []
    for ev in range(p.Nevent):
        for i in range(p.Nstim):
            seq.append(p.pause + i * p.stim + p.delay * ev)
            seq.append(p.pause + i * p.stim + p.duration + p.delay * ev)
    return seq


def compute_input(t, seq, IN, N):
    """Scalar input amplitude at time t, broadcast to all N neurons."""
    L = 0.0
    pol = 1
    for s in seq:
        L += np.tanh(t - s) * pol
        pol *= -1
    return IN * np.ones(N) * L / 2.0 + 1.0


def compute_input_trace(t_axis, seq, IN, N):
    """Vectorised input for plotting; returns array (nstep,) for neuron 0."""
    L = np.zeros(len(t_axis))
    pol = 1
    for s in seq:
        L += np.tanh(t_axis - s) * pol
        pol *= -1
    return IN * L / 2.0 + 1.0


def compute_day_timesteps(p: ModelParams):
    ts = [0]
    for ev in range(p.Nevent):
        ts.append(int(p.pause + p.delay * ev + p.Nstim * p.stim))
    return ts


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(p: ModelParams):
    """
    Euler-integrate the RNN for p.nstep steps.

    State vector layout (size = N + N² + N + N + N):
        [0:N]              firing rates r
        [N:N+N²]           recurrent weights W (flattened)
        [N+N²:N+N²+N]      excitability state (dead code — follows Emat)
        [N+N²+N:N+N²+2N]   thresholds θ
        [N+N²+2N:N+N²+3N]  readout weights W_out

    Returns dict with arrays and metadata.
    """
    rng = np.random.RandomState(p.seed)
    N = p.N

    Emat, exc0 = build_excitability_matrix(p, rng)
    seq = build_stimulus_sequence(p)

    # Initial state — all zeros except small random readout weights
    readout_W0 = rng.uniform(0, 0.02, N)
    y0 = np.concatenate([
        np.zeros(N),       # r
        np.zeros(N * N),   # W
        np.zeros(N),       # excitability (dead code)
        np.zeros(N),       # θ
        readout_W0,        # W_out
    ])

    sl_r    = slice(0, N)
    sl_W    = slice(N, N + N * N)
    sl_th   = slice(N + N * N + N, N + N * N + 2 * N)
    sl_rout = slice(N + N * N + 2 * N, N + N * N + 3 * N)

    y = np.zeros((len(y0), p.nstep))
    y[:, 0] = y0

    for step in range(p.nstep - 1):
        t = p.dt * step
        yt = y[:, step]

        r = yt[sl_r][:, np.newaxis]
        r = r * (r > 1e-5)                    # zero out floating-point negatives
        W = yt[sl_W].reshape(N, N)
        exc = Emat[:, step][:, np.newaxis]
        theta = yt[sl_th][:, np.newaxis]       # tracked but not used in drdt
        rout_W = yt[sl_rout][:, np.newaxis]

        inp = compute_input(t, seq, p.IN, N)[:, np.newaxis]

        # Eq. 1 — global inhibition
        rinhib = p.I0 + p.I1 * np.sum(r) + p.I2 * np.sum(np.maximum(0, r) * r)

        # Eq. 2 — firing rate (θ intentionally absent — dead code in this model)
        drdt = (-r + np.maximum(0, W @ r + inp - rinhib + exc)) / p.taur

        # Eq. 3 — threshold adaptation (tracked only)
        dthetadt = (r / p.y0_scale - theta) / p.tautheta

        # Eq. 4 — Hebbian weight update with volatility noise, clamped to [0, 1]
        noise = rng.normal(p.vol_mean, p.vol_std, (N, N))
        dWdt = r @ r.T / p.tauw - W / p.decay + noise
        clamp = ~((W >= 1) & (dWdt > 0)) & ~((W <= 0) & (dWdt < 0))
        dWdt *= clamp

        # Readout — homeostatic Hebbian
        h = 1.0 - np.sum(rout_W)
        readout_r = rout_W.T @ r                # no relu in weight update
        drout_Wdt = h * r * readout_r / p.tau_out_plus - rout_W / p.tau_out_minus

        dydt = np.concatenate([
            drdt.flatten(),
            dWdt.flatten(),
            np.zeros(N),           # dexcdt — dead code
            dthetadt.flatten(),
            drout_Wdt.flatten(),
        ])
        y[:, step + 1] = yt + p.dt * dydt

    return {
        "y": y,
        "Emat": Emat,
        "exc0": exc0,
        "seq": seq,
        "day_timesteps": compute_day_timesteps(p),
        "params": p,
        "slices": {"r": sl_r, "W": sl_W, "theta": sl_th, "readout_W": sl_rout},
    }


def extract_arrays(result):
    """Return named arrays from a simulation result dict."""
    y = result["y"]
    sl = result["slices"]
    p = result["params"]
    N = p.N
    return {
        "r":         y[sl["r"], :],
        "W":         y[sl["W"], :],
        "theta":     y[sl["theta"], :],
        "readout_W": y[sl["readout_W"], :],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def get_or_run(p: ModelParams, name: str = "", force: bool = False):
    """Run simulation. name and force are ignored (caching removed)."""
    print(f"Running simulation (seed={p.seed}, E={p.E}, vol_std={p.vol_std}) ...")
    return run_simulation(p)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_population_vectors(result):
    """
    End-of-day firing rate vectors.
    Returns ndarray shape (Nevent, N).
    """
    r = extract_arrays(result)["r"]
    day_ts = result["day_timesteps"]
    p = result["params"]
    return np.stack([r[:, day_ts[d + 1] - 1] for d in range(p.Nevent)])


def compute_drift_rate(result):
    """
    Delta = sum_{i=2}^{Nevent} (1 - corr(V_1, V_i))
    where V_i is the end-of-day population firing-rate vector.
    Returns scalar float. NaN correlations (zero-variance vectors) are treated as 0.
    """
    V = compute_population_vectors(result)
    delta = 0.0
    for i in range(1, len(V)):
        cc = np.corrcoef(V[0], V[i])[0, 1]
        delta += 1.0 - (0.0 if np.isnan(cc) else float(cc))
    return delta


def run_grid_sweep(E_values, vol_std_values, seeds, base_params=None,
                   name="grid_sweep", force=False):
    """
    Sweep E x vol_std x seeds.

    Returns dict with keys:
      E_values       (nE,)
      vol_std_values (nvol,)
      seeds          list
      drift          (nE, nvol, nseed)
      pop_vecs       (nE, nvol, nseed, Nevent, N)
      weight_var     (nE, nvol, nseed)  — Var(W_final)
      base_params    ModelParams
    """
    E_arr   = np.asarray(E_values,       dtype=float)
    vol_arr = np.asarray(vol_std_values, dtype=float)
    seeds   = list(seeds)

    if base_params is None:
        base_params = ModelParams()

    nE, nvol, ns  = len(E_arr), len(vol_arr), len(seeds)
    N, Nevent     = base_params.N, base_params.Nevent

    drift      = np.zeros((nE, nvol, ns))
    pop_vecs   = np.zeros((nE, nvol, ns, Nevent, N))
    weight_var = np.zeros((nE, nvol, ns))

    total, done = nE * nvol * ns, 0
    for i, E in enumerate(E_arr):
        for j, vs in enumerate(vol_arr):
            for k, seed in enumerate(seeds):
                p = _dc_replace(base_params, E=float(E), vol_std=float(vs), seed=int(seed))
                res = run_simulation(p)
                drift[i, j, k]      = compute_drift_rate(res)
                pop_vecs[i, j, k]   = compute_population_vectors(res)
                W_final             = extract_arrays(res)["W"][:, -1].reshape(N, N)
                weight_var[i, j, k] = float(np.var(W_final))
                done += 1
                print(f"\r  [{done}/{total}] E={E:.2f}  vol_std={vs:.4f}  seed={seed}",
                      end="", flush=True)
    print()

    return dict(E_values=E_arr, vol_std_values=vol_arr, seeds=seeds,
                drift=drift, pop_vecs=pop_vecs, weight_var=weight_var,
                base_params=base_params)


def run_tauw_sweep(tauw_values, seeds, base_params=None,
                   name="tauw_sweep", force=False):
    """
    Sweep recurrent Hebbian time constant tauw x seeds.

    Returns dict with keys:
      tauw_values (ntauw,)
      lr_values   (ntauw,)  = 1 / tauw
      seeds       list
      drift       (ntauw, nseed)
      weight_var  (ntauw, nseed)  — Var(W_final)
      base_params ModelParams
    """
    tauw_arr = np.asarray(tauw_values, dtype=float)
    seeds    = list(seeds)

    if base_params is None:
        base_params = ModelParams()

    ntauw, ns = len(tauw_arr), len(seeds)
    N = base_params.N

    drift      = np.zeros((ntauw, ns))
    weight_var = np.zeros((ntauw, ns))

    total, done = ntauw * ns, 0
    for i, tauw in enumerate(tauw_arr):
        for k, seed in enumerate(seeds):
            p = _dc_replace(base_params, tauw=float(tauw), seed=int(seed))
            res = run_simulation(p)
            drift[i, k]      = compute_drift_rate(res)
            W_final          = extract_arrays(res)["W"][:, -1].reshape(N, N)
            weight_var[i, k] = float(np.var(W_final))
            done += 1
            print(f"\r  [{done}/{total}] tauw={tauw:.1f}  seed={seed}",
                  end="", flush=True)
    print()

    return dict(tauw_values=tauw_arr, lr_values=1.0 / tauw_arr,
                seeds=seeds, drift=drift, weight_var=weight_var,
                base_params=base_params)


def compute_readout_accuracy(result):
    """
    Per-day binary detection accuracy using readout weights frozen at end of Day 1.

    Threshold = midpoint between mean stimulus-on and mean stimulus-off readout
    outputs on Day 1; applied unchanged to Days 2-4.

    Returns
    -------
    accuracy : ndarray (Nevent,)
    threshold : float
    """
    arrays  = extract_arrays(result)
    r       = arrays["r"]           # (N, nstep)
    rout_W  = arrays["readout_W"]   # (N, nstep)
    day_ts  = result["day_timesteps"]
    p       = result["params"]
    seq     = result["seq"]

    W_frozen = rout_W[:, day_ts[1] - 1]   # (N,) — weights at end of Day 1

    def _labels(d):
        t0, t1 = day_ts[d], day_ts[d + 1]
        T   = t1 - t0
        lbl = np.zeros(T, dtype=bool)
        for i in range(p.Nstim):
            on  = seq[d * 2 * p.Nstim + 2 * i]     - t0
            off = seq[d * 2 * p.Nstim + 2 * i + 1] - t0
            lbl[max(0, on):min(T, off)] = True
        return lbl

    rout_d1  = W_frozen @ r[:, day_ts[0]:day_ts[1]]
    lbl_d1   = _labels(0)
    mean_on  = float(rout_d1[lbl_d1].mean())   if lbl_d1.any()   else 1.0
    mean_off = float(rout_d1[~lbl_d1].mean())  if (~lbl_d1).any() else 0.0
    thresh   = (mean_on + mean_off) / 2.0

    accuracy = np.zeros(p.Nevent)
    for d in range(p.Nevent):
        t0, t1 = day_ts[d], day_ts[d + 1]
        rout        = W_frozen @ r[:, t0:t1]
        lbl         = _labels(d)
        accuracy[d] = float(np.mean((rout > thresh) == lbl))

    return accuracy, thresh


def compute_frozen_r2(result):
    """
    Normalized R² of a Day-1-frozen OLS linear decoder applied across days.

    Target: binary stimulus-presence label (1 during any stim-on, 0 elsewhere).
    Decoder weights: OLS trained on Day 1 population activity, then frozen.
    Normalization: each day's R² is divided by Day-1 R², so Day 1 = 1.0.

    This metric matches Rule et al. (2020) Fig 3d: a fixed linear readout
    trained over a baseline period and evaluated on subsequent days.

    Returns
    -------
    r2_norm : ndarray (Nevent,)  — normalized R²; Day 1 = 1.0
    r2_raw  : ndarray (Nevent,)  — raw R² values
    """
    arrays = extract_arrays(result)
    r      = arrays["r"]          # (N, nstep)
    day_ts = result["day_timesteps"]
    p      = result["params"]
    seq    = result["seq"]

    def _segment(d):
        t0, t1 = day_ts[d], day_ts[d + 1]
        X = r[:, t0:t1].T          # (T, N)
        X = X - X.mean(axis=0)     # centre per-neuron within this day (Rule et al. preprocessing)
        T = t1 - t0
        y = np.zeros(T)
        for i in range(p.Nstim):
            on  = seq[d * 2 * p.Nstim + 2 * i]     - t0
            off = seq[d * 2 * p.Nstim + 2 * i + 1] - t0
            y[max(0, on):min(T, off)] = 1.0
        return X, y

    X1, y1 = _segment(0)
    w, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)  # (N,)

    def _r2(X, y):
        yhat   = X @ w
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_raw  = np.array([_r2(*_segment(d)) for d in range(p.Nevent)])
    r2_norm = r2_raw / r2_raw[0] if r2_raw[0] > 0 else r2_raw.copy()

    return r2_norm, r2_raw


def compute_subspace_overlap(result, k=3):
    """
    Per-day overlap between the Day-1 stimulus-coding subspace and the Day-d subspace.

    For each day the k-dimensional coding subspace is the top-k right singular
    vectors of the (Nstim x N) matrix of mean stimulus response vectors (centred).
    Overlap = mean squared cosine of the k principal angles between the Day-1
    and Day-d subspaces, normalised to [0, 1] (1.0 = identical, 0 = orthogonal).

    Parameters
    ----------
    k : int
        Subspace dimension. Clamped to Nstim - 1 to stay within rank.

    Returns
    -------
    overlap : ndarray (Nevent,)  — 1.0 on Day 1, decreases with drift
    """
    arrays = extract_arrays(result)
    r      = arrays["r"]
    seq    = result["seq"]
    p      = result["params"]

    k = min(k, p.Nstim - 1)

    def _day_responses(d):
        vecs = []
        for i in range(p.Nstim):
            on  = seq[d * 2 * p.Nstim + 2 * i]
            off = seq[d * 2 * p.Nstim + 2 * i + 1]
            vecs.append(r[:, on:off].mean(axis=1))
        return np.array(vecs)   # (Nstim, N)

    def _subspace(responses):
        X = responses - responses.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt[:k]           # (k, N) — top-k right singular vectors

    S0 = _subspace(_day_responses(0))   # Day-1 reference subspace

    overlap = np.zeros(p.Nevent)
    for d in range(p.Nevent):
        Sd    = _subspace(_day_responses(d))
        sigma = np.linalg.svd(S0 @ Sd.T, compute_uv=False)
        sigma = np.clip(sigma, 0.0, 1.0)
        overlap[d] = float(np.mean(sigma ** 2))

    return overlap


def run_decoder_sweep(E_values, vol_std_values, seeds, base_params=None,
                      name="decoder_sweep", force=False):
    """
    Sweep E x vol_std x seeds; compute per-day frozen-readout accuracy.

    Returns dict with:
        E_values        (nE,)
        vol_std_values  (nvol,)
        seeds           list
        accuracy        (nE, nvol, nseed, Nevent)
        base_params     ModelParams
    """
    E_arr   = np.asarray(E_values,       dtype=float)
    vol_arr = np.asarray(vol_std_values, dtype=float)
    seeds   = list(seeds)

    if base_params is None:
        base_params = ModelParams()

    nE, nvol, ns = len(E_arr), len(vol_arr), len(seeds)
    Nevent = base_params.Nevent

    accuracy = np.zeros((nE, nvol, ns, Nevent))

    total, done = nE * nvol * ns, 0
    for i, E in enumerate(E_arr):
        for j, vs in enumerate(vol_arr):
            for k, seed in enumerate(seeds):
                p   = _dc_replace(base_params, E=float(E), vol_std=float(vs), seed=int(seed))
                res = run_simulation(p)
                acc, _ = compute_readout_accuracy(res)
                accuracy[i, j, k] = acc
                done += 1
                print(f"\r  [{done}/{total}] E={E:.2f}  vol_std={vs:.4f}  seed={seed}",
                      end="", flush=True)
    print()

    return dict(E_values=E_arr, vol_std_values=vol_arr, seeds=seeds,
                accuracy=accuracy, base_params=base_params)


def compute_subspace_angles(result, k=10):
    """
    Principal angles between the day-1 representational subspace and each
    subsequent day's subspace, in degrees.

    For each day d, the firing-rate matrix R_d (N × T_d) is mean-centred and
    its column space (a subspace of R^N) is used as the neural manifold for
    that day. scipy.linalg.subspace_angles computes the principal angles
    between col(R_1) and col(R_d) via an SVD of Q_1^T Q_d.

    Parameters
    ----------
    k : int
        Number of leading principal angles to return (sorted ascending).

    Returns
    -------
    angles_deg : ndarray (Nevent-1, k)
        Row i = principal angles for comparison (day 1, day i+2).
    days : list[int]
        Day labels [2, 3, ..., Nevent].
    """
    from scipy.linalg import subspace_angles as _sa

    r      = extract_arrays(result)["r"]
    day_ts = result["day_timesteps"]
    p      = result["params"]

    # Reduce each day to its leading k PCA directions (N × k orthonormal basis).
    # Without this step, col(R_d) spans all of R^N when T >> N, making every
    # pair of subspaces trivially identical and all angles zero.
    Q = []
    for d in range(p.Nevent):
        rd = r[:, day_ts[d]:day_ts[d + 1]].copy()
        rd -= rd.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(rd, full_matrices=False)   # U: (N, min(N,T))
        k_eff   = min(k, U.shape[1])
        Q.append(U[:, :k_eff])                              # (N, k_eff)

    rows = []
    for d in range(1, p.Nevent):
        theta  = _sa(Q[0], Q[d])                    # ascending radians
        k_ret  = min(k, len(theta))
        padded = np.full(k, np.nan)
        padded[:k_ret] = np.degrees(theta[:k_ret])
        rows.append(padded)

    return np.array(rows), list(range(2, p.Nevent + 1))
