"""
==========================================================================

This version drives the film with an ARBITRARY SQUARE-PULSE protocol:

    (1) Reset   : -4 V for 1 us       -> drives toward -P_R
    (2) Hold    :  0 V for T_H        -> relaxation (T_H as in the code)
    (3) Set     : +2 V for 100 ns     -> partial / set switching
    (4) Program : +4 V for 100 ns     -> programming (full +P_R)

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
    ts, Vs = [], []
    seg_info = []
    t_offset = 0.0
    n_total  = 0                      # running length of the concatenated arrays

    for i, seg in enumerate(pulse_seq):
        d = seg['duration']

        if seg.get('is_hold', False):
            n = max(2, n_hold_pts)
            # samples strictly after the previous edge (avoids duplicate t)
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
def run_general_mc(t, V, T_FE, params, N=5000, seed=0, s_init=-1):
    """
    Algorithm 2 with the auxiliary history parameter h^(i)(t).
    """
    rng = np.random.default_rng(seed)
    a, b, p, q     = params['a'], params['b'], params['p'], params['q']
    alpha, beta_w  = params['alpha'], params['beta']
    tau_inf, P_R   = params['tau_inf'], params['P_R']

    Ea = gb2_sample(N, a, b, p, q, rng)              # MV/cm
    E  = V / (T_FE * 1.0e6)                          # MV/cm

    if isinstance(s_init, str) and s_init.lower() in ("random", "rand"):
        # Unknown initial domain alignment: each grain is independently
        # +1 or -1 with equal probability (unpoled / as-fabricated film).
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    elif np.isscalar(s_init):
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
    # eps_0 = 8.854e-14 F/cm  ->  P [uC/cm^2] = (eps_r-1) * 8.854e-8 * V / T_FE
    return (eps_r - 1) * 8.854e-8 * V / T_FE

# ======================================================================
if __name__ == "__main__":

    # ---- explicit input waveform V(t) ----------------------------
    #   V(t) = {  0   (0.0 us <= t < 1.0 us)
    #            -4   (1.0 us <= t < 2.0 us)   Reset
    #             0   (2.0 us <= t < 2.1 us)
    #            +2   (2.1 us <= t < 2.2 us)   Set
    #             0   (2.2 us <= t < 2.3 us)
    #            +4   (2.3 us <= t < 2.4 us)   Program
    #             0   (t >= 2.4 us) }
    dt_active    = 1.0e-9               # timestep inside the waveform (s)
    N_grains     = 5000
    seed         = 42
    s_init       = "random"             # unknown initial domain alignment (each grain random +/-1)
    us           = 1.0e-6

    pulse_seq = [
        {'name': 'Idle',     'V':  0.0, 'duration': 1.0 * us},   # 0.0 -> 1.0 us
        {'name': 'Reset',    'V': -4.0, 'duration': 1.0 * us},   # 1.0 -> 2.0 us
        {'name': '0 V',      'V':  0.0, 'duration': 0.1 * us},   # 2.0 -> 2.1 us
        {'name': 'Set',      'V': +2.0, 'duration': 0.1 * us},   # 2.1 -> 2.2 us
        {'name': '0 V ',     'V':  0.0, 'duration': 0.1 * us},   # 2.2 -> 2.3 us
        {'name': 'Program',  'V': +4.0, 'duration': 0.1 * us},   # 2.3 -> 2.4 us
        {'name': 'Final',    'V':  0.0, 'duration': 0.6 * us},   # t >= 2.4 us
    ]

    t, V, seg_info = build_square_waveform(pulse_seq, dt_active)
    P_FE, _, _ = run_general_mc(
        t, V, T_FE_HZO, PARAMS_HZO,
        N=N_grains, seed=seed, s_init=s_init,
    )
    P_meas = P_FE #+ linear_capacitance_contribution(V, T_FE_HZO, EPS_R_LIN)
  
    d = dict(t=t, V=V, P_FE=P_FE, P_meas=P_meas, seg_info=seg_info)

    # active pulses (the non-zero voltages) we annotate on the P-V graph
    active = [pi for pi in seg_info if abs(pi['V']) > 1e-9]

    # ---- figure : input waveform (real time)  +  P-V graph -------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             constrained_layout=True)

    # ---- (left) input square waveform on a TRUE time axis --------
    ax = axes[0]
    ax.plot(d['t'] * 1e6, d['V'], lw=1.8, color='C0')
    ax.set_xlabel(r'time ($\mu$s)')
    ax.set_ylabel(r'$V_A$ (V)')
    ax.set_title('Input square waveform  V(t)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylim(-5.5, 5.5)
    for pi in active:
        t_mid = (pi['start'] + 0.5 * pi['duration']) * 1e6
        ax.annotate(f"{pi['name']}\n{pi['V']:+.0f} V",
                    xy=(t_mid, pi['V']),
                    xytext=(0, 10 if pi['V'] > 0 else -28),
                    textcoords='offset points',
                    ha='center', fontsize=9, color='C0', weight='bold')

    # ---- (right) polarization-vs-voltage graph -------------------
    ax = axes[1]
    ax.plot(d['V'], d['P_meas'], lw=1.3, color='C0', alpha=0.55,
            label='Trajectory')

    for pi in active:
        ei = pi['end_index']
        Vp, Pp = d['V'][ei], d['P_meas'][ei]
        ax.plot(Vp, Pp, 'o', ms=8, color='C3', zorder=5)
        ax.annotate(
            f"{pi['name']}\n{Vp:+.0f} V -> {Pp:+.1f}",
            xy=(Vp, Pp),
            xytext=(0, 18 if Pp >= 0 else -30), textcoords='offset points',
            ha='center', fontsize=9, color='C3', weight='bold',
        )

    ax.set_xlabel(r'$V_A$ (V)')
    ax.set_ylabel(r'Polarization ($\mu$C/cm$^2$)')
    ax.set_title('P-V  (polarization vs applied voltage)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-PARAMS_HZO['P_R'] * 1.35, PARAMS_HZO['P_R'] * 1.35)
    ax.legend(loc='lower right', frameon=False)

    fig.suptitle(
        r'HZO ($T_\mathrm{FE}$ = 8.3 nm) - Algorithm 2,  '
        f'N = {N_grains} grains\n'
        'Reset -4 V (1.0-2.0 us)  ->  Set +2 V (2.1-2.2 us)  ->  '
        'Program +4 V (2.3-2.4 us)',
        fontsize=11,
    )
    plt.savefig('algorithm2_square_pulse_pv.png', dpi=140, bbox_inches='tight')
    plt.show()

    # ---- summary : polarization for EACH applied voltage ---------
    def _fmt_dur(dur):
        if dur < 1e-6:   return f"{dur*1e9:g} ns"
        if dur < 1e-3:   return f"{dur*1e6:g} us"
        if dur < 1.0:    return f"{dur*1e3:g} ms"
        return f"{dur:g} s"

    print("=" * 74)
    print("Square-pulse waveform - polarization reached at the end of each segment")
    s_desc = s_init if isinstance(s_init, str) else f"{s_init:+d}"
    print(f"  N = {N_grains} grains,  s_init = {s_desc}  (initial domain alignment)")
    print("-" * 74)
    print(f"  {'Segment':<10}{'t-window (us)':>16}{'V (V)':>8}"
          f"{'P_FE':>12}{'P_meas':>12}")
    for pi in d['seg_info']:
        ei = pi['end_index']
        t0 = pi['start'] * 1e6
        t1 = (pi['start'] + pi['duration']) * 1e6
        win = f"{t0:.1f}-{t1:.1f}"
        print(f"  {pi['name']:<10}{win:>16}{pi['V']:>8.1f}"
              f"{d['P_FE'][ei]:>12.2f}{d['P_meas'][ei]:>12.2f}")
    print("=" * 74)
