"""
Two-context RNN model for representational drift.

Key differences from model.py:
- Context A stimulates neurons 0:N//2; Context B stimulates neurons N//2:N.
- Stimulus pulses alternate per day: A(0), B(1), A(2), ... (Nstim//2 pulses each).
- Single global excitability matrix applied across all N neurons (unchanged).
- Recurrent dynamics, volatility, and readout learning rule are identical.
- Analysis functions added: cross-context subspace angles, LDA classification.
"""
import os
import pickle

import numpy as np
from dataclasses import dataclass, replace as _dc_replace

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class ModelParams:
    # Network
    N: int = 50
    # Simulation
    nstep: int = 13000
    dt: float = 1.0
    # Firing rate dynamics
    taur: float = 50.0
    # Global inhibition
    I0: float = 5.0
    I1: float = 1.0
    I2: float = 0.05
    # Recurrent weight plasticity
    tauw: float = 1000.0
    decay: float = 1000.0
    # Threshold adaptation (tracked only, not used in drdt)
    tautheta: float = 800.0
    y0_scale: float = 10.0
    # Input stimulus
    IN: float = 15.0
    Nstim: int = 10       # total pulses per day — must be even (Nstim//2 per context)
    stim: int = 200
    duration: int = 100
    pause: int = 1000
    delay: int = 3000
    Nevent: int = 4
    # Excitability fluctuations (global, applied to full population)
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
    """Global excitability matrix — same structure as single-context model."""
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


def build_stimulus_sequences(p: ModelParams):
    """
    Returns (seq_A, seq_B): flat on/off boundary lists for each context.
    Within each day pulses alternate: A(0), B(1), A(2), B(3), ...
    seq_A and seq_B each have (Nstim//2 * Nevent * 2) elements.
    """
    seq_A, seq_B = [], []
    for ev in range(p.Nevent):
        for i in range(p.Nstim):
            on  = p.pause + i * p.stim + p.delay * ev
            off = on + p.duration
            if i % 2 == 0:
                seq_A.extend([on, off])
            else:
                seq_B.extend([on, off])
    return seq_A, seq_B


def _smooth_L(t, seq):
    """Scalar smooth-on indicator for one context's pulse list."""
    L, pol = 0.0, 1
    for s in seq:
        L += np.tanh(t - s) * pol
        pol *= -1
    return L


def compute_input(t, seq_A, seq_B, IN, N):
    """
    Context A drives neurons 0:N//2; Context B drives neurons N//2:N.
    Baseline input = 1 for both halves; stimulus adds IN to the active half.
    """
    N_half = N // 2
    L_A = _smooth_L(t, seq_A)
    L_B = _smooth_L(t, seq_B)
    inp = np.ones(N)
    inp[:N_half] += IN * L_A / 2.0
    inp[N_half:]  += IN * L_B / 2.0
    return inp


def compute_input_traces(t_axis, seq_A, seq_B, IN):
    """
    Vectorised input amplitude traces for plotting.
    Returns (trace_A, trace_B), each shape (len(t_axis),).
    """
    def _trace(seq):
        L, pol = np.zeros(len(t_axis)), 1
        for s in seq:
            L += np.tanh(t_axis - s) * pol
            pol *= -1
        return IN * L / 2.0 + 1.0
    return _trace(seq_A), _trace(seq_B)


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
    Euler-integrate the two-context RNN.
    State layout identical to model.py: [r, W, exc(dead), theta, W_out].
    """
    rng = np.random.RandomState(p.seed)
    N   = p.N

    Emat, exc0   = build_excitability_matrix(p, rng)
    seq_A, seq_B = build_stimulus_sequences(p)

    readout_W0 = rng.uniform(0, 0.02, N)
    y0 = np.concatenate([
        np.zeros(N),
        np.zeros(N * N),
        np.zeros(N),
        np.zeros(N),
        readout_W0,
    ])

    sl_r    = slice(0, N)
    sl_W    = slice(N, N + N * N)
    sl_th   = slice(N + N * N + N, N + N * N + 2 * N)
    sl_rout = slice(N + N * N + 2 * N, N + N * N + 3 * N)

    y = np.zeros((len(y0), p.nstep))
    y[:, 0] = y0

    for step in range(p.nstep - 1):
        t  = p.dt * step
        yt = y[:, step]

        r      = yt[sl_r][:, np.newaxis]
        r      = r * (r > 1e-5)
        W      = yt[sl_W].reshape(N, N)
        exc    = Emat[:, step][:, np.newaxis]
        theta  = yt[sl_th][:, np.newaxis]
        rout_W = yt[sl_rout][:, np.newaxis]

        inp = compute_input(t, seq_A, seq_B, p.IN, N)[:, np.newaxis]

        rinhib = p.I0 + p.I1 * np.sum(r) + p.I2 * np.sum(np.maximum(0, r) * r)

        drdt = (-r + np.maximum(0, W @ r + inp - rinhib + exc)) / p.taur

        dthetadt = (r / p.y0_scale - theta) / p.tautheta

        noise = rng.normal(p.vol_mean, p.vol_std, (N, N))
        dWdt  = r @ r.T / p.tauw - W / p.decay + noise
        clamp = ~((W >= 1) & (dWdt > 0)) & ~((W <= 0) & (dWdt < 0))
        dWdt *= clamp

        h         = 1.0 - np.sum(rout_W)
        readout_r = rout_W.T @ r
        drout_Wdt = h * r * readout_r / p.tau_out_plus - rout_W / p.tau_out_minus

        dydt = np.concatenate([
            drdt.flatten(),
            dWdt.flatten(),
            np.zeros(N),
            dthetadt.flatten(),
            drout_Wdt.flatten(),
        ])
        y[:, step + 1] = yt + p.dt * dydt

    return {
        "y":             y,
        "Emat":          Emat,
        "exc0":          exc0,
        "seq_A":         seq_A,
        "seq_B":         seq_B,
        "day_timesteps": compute_day_timesteps(p),
        "params":        p,
        "slices":        {"r": sl_r, "W": sl_W, "theta": sl_th, "readout_W": sl_rout},
    }


# ---------------------------------------------------------------------------
# Persistence  (identical to model.py)
# ---------------------------------------------------------------------------

def save_data(data, name: str):
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved  -> {path}")
    return path


def load_data(name: str):
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_run(p: ModelParams, name: str, force: bool = False):
    if not force:
        result = load_data(name)
        if result is not None:
            print(f"Loaded cached data: {name}")
            return result
    print(f"Running simulation (seed={p.seed}, E={p.E}, vol_std={p.vol_std}) ...")
    result = run_simulation(p)
    save_data(result, name)
    return result


def extract_arrays(result):
    y  = result["y"]
    sl = result["slices"]
    p  = result["params"]
    N  = p.N
    return {
        "r":         y[sl["r"], :],
        "W":         y[sl["W"], :],
        "theta":     y[sl["theta"], :],
        "readout_W": y[sl["readout_W"], :],
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _pulse_indices(seq, d, n_per):
    """On/off index pairs for each of the n_per pulses of a context on day d."""
    base = d * 2 * n_per
    return [(seq[base + 2 * i], seq[base + 2 * i + 1]) for i in range(n_per)]


def compute_pulse_vectors(result):
    """
    Per-pulse mean firing rate vectors for each day and context.

    Returns ndarray shape (Nevent, 2, n_per, N):
        axis 1: 0 = context A, 1 = context B
        axis 2: pulse index within day (0 .. n_per-1)
    """
    r     = extract_arrays(result)["r"]   # (N, nstep)
    seq_A = result["seq_A"]
    seq_B = result["seq_B"]
    p     = result["params"]
    n_per = p.Nstim // 2

    vecs = np.zeros((p.Nevent, 2, n_per, p.N))
    for d in range(p.Nevent):
        for ci, seq in enumerate([seq_A, seq_B]):
            for i, (on, off) in enumerate(_pulse_indices(seq, d, n_per)):
                vecs[d, ci, i] = r[:, on:off].mean(axis=1)
    return vecs


def compute_population_vectors_by_context(result):
    """
    Mean firing rate vector per (day, context), averaged across pulses.

    Returns ndarray (Nevent, 2, N):
        [:, 0, :] = context A mean,  [:, 1, :] = context B mean.
    """
    vecs = compute_pulse_vectors(result)          # (Nevent, 2, n_per, N)
    return vecs.mean(axis=2)                      # (Nevent, 2, N)


def compute_context_classification_accuracy(result):
    """
    Binary context-A-vs-B classification accuracy for each day.

    Static readout:
        w = mean(A) - mean(B) from Day 1, frozen for Days 2-4.
        Each pulse classified as A if (pulse_vec @ w) > 0.

    Adaptive readout (LOO-CV linear discriminant):
        For each pulse, train w on the remaining pulses of that day,
        test on the held-out pulse.  Gives unbiased per-day accuracy.

    Returns
    -------
    acc_static   : ndarray (Nevent,)
    acc_adaptive : ndarray (Nevent,)
    """
    pulse_vecs = compute_pulse_vectors(result)    # (Nevent, 2, n_per, N)
    p     = result["params"]
    n_per = p.Nstim // 2
    total = 2 * n_per                             # pulses per day (A + B)

    # Day-1 static readout weight
    w_static = (pulse_vecs[0, 0].mean(axis=0)
                - pulse_vecs[0, 1].mean(axis=0))  # (N,)

    acc_static   = np.zeros(p.Nevent)
    acc_adaptive = np.zeros(p.Nevent)

    for d in range(p.Nevent):
        X = np.vstack([pulse_vecs[d, 0], pulse_vecs[d, 1]])  # (total, N)
        y = np.array([1] * n_per + [0] * n_per)              # 1=A, 0=B

        # Static
        acc_static[d] = float(np.mean((X @ w_static > 0) == y))

        # Adaptive LOO-CV
        correct = 0
        for k in range(total):
            mask      = np.ones(total, dtype=bool)
            mask[k]   = False
            X_tr, y_tr = X[mask], y[mask]
            w_cv      = X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0)
            correct  += int((X[k] @ w_cv > 0) == y[k])
        acc_adaptive[d] = correct / total

    return acc_static, acc_adaptive


def compute_cross_context_subspace_angles(result, k=10):
    """
    Two angle metrics across days, each averaged over k principal angles:

    within_angles (Nevent-1,)
        Mean principal angle between Day-1 context-A subspace and
        Day-d context-A subspace (d = 2, 3, 4).
        Tracks within-context representational drift.

    cross_angles  (Nevent,)
        Mean principal angle between Day-d context-A subspace and
        Day-d context-B subspace (d = 1, 2, 3, 4).
        Should stay near 90 deg if contexts remain orthogonal.

    Returns
    -------
    within_angles, within_days, cross_angles, cross_days
    """
    from scipy.linalg import subspace_angles as _sa

    r     = extract_arrays(result)["r"]   # (N, nstep)
    seq_A = result["seq_A"]
    seq_B = result["seq_B"]
    p     = result["params"]
    n_per = p.Nstim // 2

    def _basis(seq, d):
        """k leading left-singular vectors from all context pulses on day d."""
        cols = [r[:, on:off]
                for on, off in _pulse_indices(seq, d, n_per)]
        R    = np.hstack(cols)
        R   -= R.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(R, full_matrices=False)
        k_eff   = min(k, U.shape[1])
        return U[:, :k_eff]                # (N, k_eff)

    Q_A = [_basis(seq_A, d) for d in range(p.Nevent)]
    Q_B = [_basis(seq_B, d) for d in range(p.Nevent)]

    within_angles = np.array([
        float(np.degrees(_sa(Q_A[0], Q_A[d])).mean())
        for d in range(1, p.Nevent)
    ])

    cross_angles = np.array([
        float(np.degrees(_sa(Q_A[d], Q_B[d])).mean())
        for d in range(p.Nevent)
    ])

    return (within_angles, list(range(2, p.Nevent + 1)),
            cross_angles,  list(range(1, p.Nevent + 1)))
