"""
Algorithm 2 - General Monte Carlo Simulation of Ferroelectric Polarization
Reversal under arbitrary input waveforms.
==========================================================================

Explicit SQUARE-PULSE waveform V(t):

    0   (0.0 - 1.0 us)   idle
   -4   (1.0 - 2.0 us)   Reset      -> poles toward -P_R
    0   (2.0 - 2.1 us)   hold
   +2   (2.1 - 2.2 us)   Set
    0   (2.2 - 2.3 us)   hold
   +3.0 (2.3 - 2.4 us)   Program 1
    0   (2.4 - 2.5 us)   hold
   +3.2 (2.5 - 2.6 us)   Program 2
    0   (2.6 - 2.7 us)   hold
   +3.4 (2.7 - 2.8 us)   Program 3
    0   (2.8 - 2.9 us)   hold
   +3.6 (2.9 - 3.0 us)   Program 4
    0   (3.0 - 3.1 us)   hold
   +3.8 (3.1 - 3.2 us)   Program 5
    0   (3.2 - 3.3 us)   hold
   +4.0 (3.3 - 3.4 us)   Program 6

All active pulses are 100 ns wide; all holds are T_H = 100 ns.
The staircase of programming pulses is CUMULATIVE: each pulse acts on
the polarization state left by the previous one.

Material:   HZO film, T_FE = 8.3 nm
Parameters: Table I, column g(E_a')
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma


# ----------------------------------------------------------------------
def beta_fn(p, q):
    return gamma(p) * gamma(q) / gamma(p + q)


# ----------------------------------------------------------------------
# Material parameters  --  Table I, column g(E_a')
# ----------------------------------------------------------------------
PARAMS_HZO = dict(
    a       = 12.1,
    b       = 1.79,        # MV/cm
    p       = 0.691,
    q       = 0.633,
    P_R     = 22.9,        # uC/cm^2
    tau_inf = 387e-9,      # s
    alpha   = 4.11,
    beta    = 2.07,
)
T_FE_HZO  = 8.3e-7         # film thickness in cm (8.3 nm)
EPS_R_LIN = 15.0           # effective linear permittivity (saturation slope)


# ----------------------------------------------------------------------
def gb2_sample(N, a, b, p, q, rng):
    """Sample N values from GB2(a, b, p, q) via Beta-prime transformation."""
    y = rng.beta(p, q, size=N)
    return b * (y / (1.0 - y)) ** (1.0 / a)


# ======================================================================
# Square-wave builder  -  arbitrary sequence of flat (square) pulses
# ======================================================================
def build_square_waveform(pulse_seq, dt_active, n_hold_pts=80):
    """Build an arbitrary SQUARE-WAVE V(t) from a list of flat segments."""
    ts, Vs = [], []
    seg_info = []
    t_offset = 0.0
    n_total  = 0

    for i, seg in enumerate(pulse_seq):
        d = seg['duration']

        if seg.get('is_hold', False):
            n = max(2, n_hold_pts)
            t_local = (np.arange(n) + 1.0) / n * d
            V_local = np.zeros(n)
        else:
            n = max(2, int(round(d / dt_active)) + 1)
            t_local = np.linspace(0.0, d, n)
            V_local = np.full(n, float(seg['V']))
            if i > 0:                 # drop leading point (instantaneous edge)
                t_local = t_local[1:]
                V_local = V_local[1:]

        ts.append(t_offset + t_local)
        Vs.append(V_local)
        n_total += len(t_local)

        seg_info.append({
            'name':      seg.get('name', f'seg{i}'),
            'V':         float(seg.get('V', 0.0)),
            'duration':  d,
            'is_hold':   seg.get('is_hold', False),
            'start':     t_offset,
            'end_index': n_total - 1,
        })
        t_offset += d

    return np.concatenate(ts), np.concatenate(Vs), seg_info


# ======================================================================
# Algorithm 2  -  General Monte Carlo simulation under arbitrary V(t)
# ======================================================================
def run_general_mc(t, V, T_FE, params, N=5000, seed=0, s_init=+1):
    """Algorithm 2 (verbatim) with the auxiliary history parameter h^(i)(t)."""
    rng = np.random.default_rng(seed)
    a, b, p, q     = params['a'], params['b'], params['p'], params['q']
    alpha, beta_w  = params['alpha'], params['beta']
    tau_inf, P_R   = params['tau_inf'], params['P_R']

    Ea = gb2_sample(N, a, b, p, q, rng)              # MV/cm
    E  = V / (T_FE * 1.0e6)                          # MV/cm

    if np.isscalar(s_init):
        s = np.full(N, s_init, dtype=np.int8)
    else:
        s = np.array(s_init, dtype=np.int8)
    h = np.zeros(N)

    P = np.empty(len(t))
    P[0] = P_R * s.mean()

    for k in range(1, len(t)):
        dt_k = t[k] - t[k - 1]
        E_k  = E[k]

        if abs(E_k) < 1.0e-15 or dt_k <= 0.0:
            P[k] = P_R * s.mean()
            continue

        cond = (s * E_k) < 0
        if not cond.any():
            P[k] = P_R * s.mean()
            continue

        idx   = np.where(cond)[0]
        Ea_c  = Ea[idx]
        h_c   = h[idx]
        tau_c = tau_inf * np.exp((Ea_c / abs(E_k)) ** alpha)
        h_new = h_c + dt_k / tau_c

        P_sw = 1.0 - np.exp(h_c ** beta_w - h_new ** beta_w)
        P_sw = np.clip(P_sw, 0.0, 1.0)

        h[idx] = h_new
        flips     = rng.random(len(idx)) < P_sw
        flip_idx  = idx[flips]
        s[flip_idx] *= -1
        h[flip_idx]  = 0.0

        P[k] = P_R * s.mean()

    return P, s, h


# ----------------------------------------------------------------------
def linear_capacitance_contribution(V, T_FE, eps_r):
    if eps_r <= 1:
        return np.zeros_like(V)
    return (eps_r - 1) * 8.854e-8 * V / T_FE


# ======================================================================
# ===========  EDITABLE BLOCK  =========================================
# ======================================================================
if __name__ == "__main__":

    dt_active    = 1.0e-9               # timestep inside the waveform (s)
    N_grains     = 5000
    seed         = 42
    s_init       = -1                   # start fully poled at -P_R
    us           = 1.0e-6
    ns           = 1.0e-9

    # ---- explicit input waveform V(t) (exactly as specified) -----
    pulse_seq = [
        {'name': 'Idle',      'V':  0.0, 'duration': 1.0 * us, 'is_hold': True},
        {'name': 'Reset',     'V': -4.0, 'duration': 1.0 * us},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Set',       'V': +2.0, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 3.0',  'V': +3.0, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 3.2',  'V': +3.2, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 3.4',  'V': +3.4, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 3.6',  'V': +3.6, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 3.8',  'V': +3.8, 'duration': 100 * ns},
        {'name': 'hold',      'V':  0.0, 'duration': 100 * ns, 'is_hold': True},
        {'name': 'Prog 4.0',  'V': +4.0, 'duration': 100 * ns},
        {'name': 'Final',     'V':  0.0, 'duration': 0.6 * us, 'is_hold': True},
    ]

    t, V, seg_info = build_square_waveform(pulse_seq, dt_active)
    P_FE, _, _ = run_general_mc(
        t, V, T_FE_HZO, PARAMS_HZO,
        N=N_grains, seed=seed, s_init=s_init,
    )
    P_meas = P_FE # + linear_capacitance_contribution(V, T_FE_HZO, EPS_R_LIN)
    d = dict(t=t, V=V, P_FE=P_FE, P_meas=P_meas, seg_info=seg_info)

    active = [pi for pi in seg_info if abs(pi['V']) > 1e-9]

    # ---- figure : input waveform + polarization waveform ---------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                             constrained_layout=True)

    # (top) input square waveform on a TRUE time axis
    ax = axes[0]
    ax.plot(d['t'] * 1e6, d['V'], lw=1.8, color='C0')
    ax.set_ylabel(r'$V_A$ (V)')
    ax.set_title('Input square waveform  V(t)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylim(-5.0, 5.0)
    for pi in active:
        t_mid = (pi['start'] + 0.5 * pi['duration']) * 1e6
        ax.annotate(f"{pi['V']:+.1f}",
                    xy=(t_mid, pi['V']),
                    xytext=(0, 8 if pi['V'] > 0 else -16),
                    textcoords='offset points',
                    ha='center', fontsize=8, color='C0', weight='bold')

    # (bottom) polarization vs time, with markers at end of each segment
    ax = axes[1]
    ax.plot(d['t'] * 1e6, d['P_FE'], lw=1.6, color='C3',
            label=r'$P_\mathrm{FE}$ (remanent)')
    ax.plot(d['t'] * 1e6, d['P_meas'], lw=1.0, color='C0', alpha=0.5,
            label=r'$P_\mathrm{meas}$ (FE + linear)')
    for pi in active:
        ei = pi['end_index']
        tp = d['t'][ei] * 1e6
        Pp = d['P_FE'][ei]
        ax.plot(tp, Pp, 'o', ms=6, color='k', zorder=5)
        ax.annotate(f"{Pp:+.1f}", xy=(tp, Pp), xytext=(0, 8),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='k')
    ax.set_xlabel(r'time ($\mu$s)')
    ax.set_ylabel(r'Polarization ($\mu$C/cm$^2$)')
    ax.set_title('Polarization response  P(t)  '
                 '(markers = value at end of each pulse)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylim(-PARAMS_HZO['P_R'] * 1.35, PARAMS_HZO['P_R'] * 1.35)
    ax.legend(loc='lower right', frameon=False)

    fig.suptitle(
        r'HZO ($T_\mathrm{FE}$ = 8.3 nm) - Algorithm 2,  '
        f'N = {N_grains} grains\n'
        'Reset -4 V -> Set +2 V -> programming staircase 3.0..4.0 V '
        '(100 ns pulses, T_H = 100 ns)',
        fontsize=11,
    )
    plt.savefig('different_vp_squarewaveform.png', dpi=140,
                bbox_inches='tight')
    plt.show()

    # ---- summary : polarization for EACH applied voltage ---------
    def _fmt_dur(dur):
        if dur < 1e-6:   return f"{dur*1e9:g} ns"
        if dur < 1e-3:   return f"{dur*1e6:g} us"
        if dur < 1.0:    return f"{dur*1e3:g} ms"
        return f"{dur:g} s"

    print("=" * 80)
    print("Explicit square-pulse waveform - polarization at the end of each segment")
    print(f"  N = {N_grains} grains,  s_init = {s_init:+d}  (start at +P_R)")
    print("-" * 80)
    print(f"  {'Segment':<10}{'t-window (us)':>16}{'V (V)':>8}"
          f"{'P_FE':>12}{'P_meas':>12}")
    print("-" * 80)
    for pi in d['seg_info']:
        ei = pi['end_index']
        t0 = pi['start'] * 1e6
        t1 = (pi['start'] + pi['duration']) * 1e6
        win = f"{t0:.1f}-{t1:.1f}"
        print(f"  {pi['name']:<10}{win:>16}{pi['V']:>8.1f}"
              f"{d['P_FE'][ei]:>12.2f}{d['P_meas'][ei]:>12.2f}")
    print("=" * 80)
    print("Note: P_FE = remanent (switched-dipole) polarization;")
    print("      P_meas = P_FE + linear-dielectric term while V is applied.")
    print("=" * 80)
