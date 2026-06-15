"""
Algorithm 2 — General Monte Carlo Simulation of Ferroelectric Polarization
Reversal under arbitrary input waveforms.
==========================================================================

This version supports MULTIPLE back-to-back triangles per pulse position
(e.g. two negative triangles, hold, two positive triangles, hold, ...),
producing the saw-tooth pulse trains shown in the experimental figure.

    ① n_sub negative triangles (back-to-back)  ->  drives to -P_R
    Hold T_H
    ② n_sub positive triangles                 ->  drives to +P_R
    Hold T_H
    ③ n_sub negative triangles
    Hold T_H
    ④ n_sub positive triangles

Default n_sub_pulses = 2 (i.e. "double triangle" per polarity).
Set n_sub_pulses = 1 for the original single-triangle behaviour.

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
# Multi-pulse builder  —  n_sub triangles per polarity, then hold
# ======================================================================
def build_multi_pulse(V_max, t_triangle, T_H, dt_active,
                      polarities=(-1, +1, -1, +1),
                      n_sub_pulses=2,
                      n_hold_pts=80):
    """
    Generate a sequence of pulse GROUPS separated by hold periods.

    Each GROUP contains `n_sub_pulses` triangular pulses of the same polarity,
    placed back-to-back (saw-tooth).  Each triangle: 0 -> sign*V_max -> 0
    over duration t_triangle.  Between groups, V is held at zero for T_H s.

    Parameters
    ----------
    V_max          peak voltage (V)
    t_triangle     duration of ONE triangle in the group (s)
    T_H            hold time between groups (s); use 0 to suppress holds
    dt_active      timestep used inside the triangles (s)
    polarities     iterable of +/-1 — one entry per pulse group
    n_sub_pulses   how many triangles per group  (e.g. 2 = double triangle)
    n_hold_pts     number of samples placed inside each hold (plotting only)

    Returns
    -------
    t, V           numpy arrays of equal length
    pulse_info     list of dicts with keys:
                       'start'  = first time of the group
                       'sign'   = +1 / -1
                       'width'  = total duration of the group (s)
                       'n_sub'  = n_sub_pulses
    """
    n_half = max(2, int(round((t_triangle / 2.0) / dt_active)))
    base_V_full = np.concatenate([
        np.linspace(0.0, 1.0, n_half + 1),
        np.linspace(1.0, 0.0, n_half + 1)[1:]
    ])
    base_t_full = np.linspace(0.0, t_triangle, len(base_V_full))

    # For sub-triangles j>=1, drop the leading V=0 point to keep t strictly
    # monotonic (it would coincide with the trailing V=0 of the previous one).
    base_V_tail = base_V_full[1:]
    base_t_tail = base_t_full[1:]

    ts, Vs = [], []
    pulse_info = []
    t_offset = 0.0

    for ip, sign in enumerate(polarities):
        group_start = t_offset

        for j in range(n_sub_pulses):
            if j == 0:
                ts.append(t_offset + base_t_full)
                Vs.append(sign * V_max * base_V_full)
            else:
                ts.append(t_offset + base_t_tail)
                Vs.append(sign * V_max * base_V_tail)
            t_offset += t_triangle

        pulse_info.append({
            'start':  group_start,
            'sign':   sign,
            'width':  n_sub_pulses * t_triangle,
            'n_sub':  n_sub_pulses,
        })

        # hold between groups (skip after the last group)
        if ip < len(polarities) - 1 and T_H > 1e-12:
            t_hold = t_offset + (np.arange(n_hold_pts) + 0.5) * (T_H / n_hold_pts)
            ts.append(t_hold)
            Vs.append(np.zeros(n_hold_pts))
            t_offset += T_H

    return np.concatenate(ts), np.concatenate(Vs), pulse_info


# ======================================================================
# Algorithm 2  -  General Monte Carlo simulation under arbitrary V(t)
# ======================================================================
def run_general_mc(t, V, T_FE, params, N=5000, seed=0, s_init=+1):
    """
    Algorithm 2 (verbatim) with the auxiliary history parameter h^(i)(t).
    """
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
    # eps_0 = 8.854e-14 F/cm  ->  P [uC/cm^2] = (eps_r-1) * 8.854e-8 * V / T_FE
    return (eps_r - 1) * 8.854e-8 * V / T_FE


# ======================================================================
# ===========  EDITABLE BLOCK  =========================================
# ======================================================================
if __name__ == "__main__":

    # ---- waveform settings ---------------------------------------
    V_max        = 2.5                  # peak voltage (V)
    t_triangle   = 5.0e-3               # ONE-triangle width (s)
    dt_active    = 5.0e-6               # timestep inside the triangles (s)
    N_grains     = 5000
    seed         = 42
    s_init       = +1
    polarities   = (-1, +1, -1, +1)     # 4 groups: ①②③④
    n_sub_pulses = 2                    # <-- two back-to-back triangles per group

    # ---- hold times to compare -----------------------------------
    hold_cases = [('T_H = 10 s', 10.0),
                  ('T_H = 1 ms', 1.0e-3)]

    # ---- run all cases -------------------------------------------
    data = {}
    for label, T_H in hold_cases:
        t, V, pulse_info = build_multi_pulse(
            V_max, t_triangle, T_H, dt_active,
            polarities=polarities,
            n_sub_pulses=n_sub_pulses,
        )
        P_FE, _, _ = run_general_mc(
            t, V, T_FE_HZO, PARAMS_HZO,
            N=N_grains, seed=seed, s_init=s_init,
        )
        P_meas = P_FE + linear_capacitance_contribution(V, T_FE_HZO, EPS_R_LIN)
        data[label] = dict(t=t, V=V, P_FE=P_FE, P_meas=P_meas,
                           pulse_info=pulse_info, T_H=T_H)

    # ---- figure ---------------------------------------------------
    circ = ['\u2460', '\u2461', '\u2462', '\u2463']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4),
                              constrained_layout=True)

    for col, (label, T_H) in enumerate(hold_cases):
        d = data[label]

        # ---- (top) applied waveform -----------------------------
        ax = axes[0, col]
        ax.plot(d['t'], d['V'], lw=1.3, color='C0')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$V_A$ (V)')
        ax.set_title(f'Applied waveform — {label} '
                     f'($n_\\mathrm{{sub}}$ = {n_sub_pulses} per group)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylim(-V_max * 1.4, V_max * 1.4)
        for pi, cl in zip(d['pulse_info'], circ):
            y_lab = pi['sign'] * V_max * 1.18
            ax.text(pi['start'] + pi['width'] / 2.0, y_lab, cl,
                    ha='center', va='center', fontsize=15,
                    color='C3', weight='bold')

        # ---- (bottom) P-V hysteresis loop -----------------------
        ax = axes[1, col]
        ax.plot(d['V'], d['P_meas'], lw=1.5, color='C3', alpha=0.85,
                label='Simulated')
        ax.set_xlabel(r'$V_A$ (V)')
        ax.set_ylabel(r'Polarization ($\mu$C/cm$^2$)')
        ax.set_title(f'P–V hysteresis loop — {label}')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.4)
        ax.axvline(0, color='k', lw=0.4)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-PARAMS_HZO['P_R'] * 1.25, PARAMS_HZO['P_R'] * 1.25)
        ax.legend(loc='lower right', frameon=False)

        ann_kw = dict(fontsize=13, color='C3', weight='bold',
                      ha='center', va='center',
                      bbox=dict(boxstyle='circle,pad=0.15',
                                fc='white', ec='C3', lw=1))
        ax.text(-1.4, -8, circ[0], **ann_kw)
        ax.text(+1.0, -8, circ[1], **ann_kw)
        ax.text(-1.0, +8, circ[2], **ann_kw)
        ax.text(+1.4, +8, circ[3], **ann_kw)

    fig.suptitle(
        r'HZO ($T_\mathrm{FE}$ = 8.3 nm) — Algorithm 2,   '
        f'N = {N_grains} grains,   '
        f'$V_\\mathrm{{max}}$ = {V_max} V,   '
        f'$t_\\mathrm{{tri}}$ = {t_triangle*1e3:g} ms,   '
        f'sub-pulses/group = {n_sub_pulses}',
        fontsize=12,
    )
    plt.savefig('algorithm2_double_triangle_pv.png', dpi=140,
                bbox_inches='tight')
    plt.show()

    # ---- summary -------------------------------------------------
    print("=" * 64)
    print(f"Multi-sub-pulse measurement protocol — summary")
    print(f"  V_max = {V_max} V,  t_triangle = {t_triangle*1e3:g} ms")
    print(f"  polarities = {polarities},  n_sub_pulses = {n_sub_pulses}")
    for label, T_H in hold_cases:
        d = data[label]
        print(f"  {label}:  final P_meas = {d['P_meas'][-1]:+6.2f} uC/cm^2  "
              f"(P_FE = {d['P_FE'][-1]:+6.2f})")
    print("=" * 64)
