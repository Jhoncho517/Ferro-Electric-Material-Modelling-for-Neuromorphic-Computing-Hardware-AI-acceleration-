"""
Algorithm 2 — General Monte Carlo Simulation of Ferroelectric Polarization
Reversal under arbitrary input waveforms.
==========================================================================

Reproduces the experimental four-pulse measurement protocol shown in the
figure: a sequence of triangular pulses ①②③④ of alternating polarity,
separated by hold periods T_H.

    ① negative triangle  ->  drives polarization to -P_R
    Hold T_H
    ② positive triangle  ->  drives polarization to +P_R
    Hold T_H
    ③ negative triangle  ->  drives polarization to -P_R
    Hold T_H
    ④ positive triangle  ->  drives polarization to +P_R

The P-V hysteresis loop is built by plotting P(t) vs V(t) for the full
waveform.  Pulses ② and ④ trace the right (positive-going) branch;
pulses ① and ③ trace the left (negative-going) branch — so the loop
is traced twice, matching the experimental double-line appearance.

Material:   HZO film, T_FE = 8.3 nm
Parameters: Table I, column g(E_a')

EDITABLE BLOCK at the bottom — change V_max, t_triangle, T_H, the sequence
of polarities, or the material parameters to run other experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma


# ======================================================================
# Beta function  (replaces scipy.special.beta)
# ======================================================================
def beta_fn(p, q):
    return gamma(p) * gamma(q) / gamma(p + q)


# ======================================================================
# Material parameters  --  Table I, column g(E_a')
# ======================================================================
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
EPS_R_LIN = 15.0           # effective linear permittivity (loop saturation slope)


# ======================================================================
# GB2 (Generalized Beta of the Second Kind) sampling
# ======================================================================
def gb2_sample(N, a, b, p, q, rng):
    """Sample N values from GB2(a, b, p, q) via Beta-prime transformation."""
    y = rng.beta(p, q, size=N)
    return b * (y / (1.0 - y)) ** (1.0 / a)


# ======================================================================
# Multi-pulse triangular waveform with hold periods
# ======================================================================
def build_multi_pulse(V_max, t_triangle, T_H, dt_active,
                      polarities=(-1, +1, -1, +1),
                      n_hold_pts=80):
    """
    Sequence of triangular pulses separated by hold periods.

    Each pulse: 0 -> sign*V_max -> 0   over duration t_triangle.
    Hold:        V = 0 for T_H seconds between consecutive pulses.

    Parameters
    ----------
    V_max         peak voltage (V)
    t_triangle    duration of one full triangular pulse (s)
    T_H           hold time between pulses (s) — use 0 to disable holds
    dt_active     timestep inside the pulses (s)
    polarities    iterable of pulse signs (+1 or -1) — length = # of pulses
    n_hold_pts    number of samples placed inside each hold (plotting only)

    Returns
    -------
    t, V          numpy arrays of equal length
    pulse_info    list of dicts {'start': t0, 'sign': +/-1} for each pulse
    """
    n_half = max(2, int(round((t_triangle / 2.0) / dt_active)))
    base_V = np.concatenate([
        np.linspace(0.0, 1.0, n_half + 1),
        np.linspace(1.0, 0.0, n_half + 1)[1:]
    ])
    base_t = np.linspace(0.0, t_triangle, len(base_V))

    ts, Vs = [], []
    pulse_info = []
    t_offset = 0.0
#Triangle waveform
    for ip, sign in enumerate(polarities):
        pulse_info.append({'start': t_offset, 'sign': sign})
        ts.append(t_offset + base_t)
        Vs.append(sign * V_max * base_V)
        t_offset += t_triangle

        # add hold (skip after the last pulse)
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
    Implements Algorithm 2 with the auxiliary history parameter h^(i)(t).

    For each timestep [t, t+dt] and each grain:
        if s^(i) * E(t) < 0:                    # field opposes polarization
            tau^(i)   = tau_inf * exp((E_a^(i) / |E(t)|)^alpha)
            h_new     = h^(i) + dt / tau^(i)
            P^(i)     = 1 - exp( h^(i)^beta  -  h_new^beta )
            h^(i)     = h_new
            if Bernoulli(P^(i)) == 1:
                flip s^(i)
                h^(i) = 0           (relaxation upon switching)
    """
    rng = np.random.default_rng(seed)
    a, b, p, q     = params['a'], params['b'], params['p'], params['q']
    alpha, beta_w  = params['alpha'], params['beta']
    tau_inf, P_R   = params['tau_inf'], params['P_R']

    # 1) sample activation fields
    Ea = gb2_sample(N, a, b, p, q, rng)              # MV/cm
    # 2) convert applied voltage to electric field (MV/cm)
    E  = V / (T_FE * 1.0e6)

    # 3) initialise state and history
    if np.isscalar(s_init):
        s = np.full(N, s_init, dtype=np.int8)
    else:
        s = np.array(s_init, dtype=np.int8)
    h = np.zeros(N)

    # 4) time-stepping loop
    P = np.empty(len(t))
    P[0] = P_R * s.mean()

    for k in range(1, len(t)):
        dt_k = t[k] - t[k - 1]
        E_k  = E[k]

        # No field  -> nothing can happen (condition s*E < 0 not satisfied)
        if abs(E_k) < 1.0e-15:
            P[k] = P_R * s.mean()
            continue

        # Candidates: grains whose polarization opposes the current field
        cond = (s * E_k) < 0
        if not cond.any():
            P[k] = P_R * s.mean()
            continue

        idx   = np.where(cond)[0]
        Ea_c  = Ea[idx]
        h_c   = h[idx]
        tau_c = tau_inf * np.exp((Ea_c / abs(E_k)) ** alpha)
        h_new = h_c + dt_k / tau_c

        # conditional Weibull switching probability for [h, h_new]
        P_sw = 1.0 - np.exp(h_c ** beta_w - h_new ** beta_w)
        P_sw = np.clip(P_sw, 0.0, 1.0)

        # advance h, then Bernoulli draws
        h[idx] = h_new
        flips     = rng.random(len(idx)) < P_sw
        flip_idx  = idx[flips]
        s[flip_idx] *= -1                 # flip the polarization
        h[flip_idx]  = 0.0                # h-relaxation after a switch

        P[k] = P_R * s.mean()

    return P, s, h


# ======================================================================
# Linear dielectric (capacitive) background — gives the saturation slope
# ======================================================================
def linear_capacitance_contribution(V, T_FE, eps_r):
    """Linear P-contribution in uC/cm^2."""
    if eps_r <= 1:
        return np.zeros_like(V)
    # eps_0 = 8.854e-14 F/cm
    # P_lin [C/cm^2] = (eps_r - 1) * eps_0 * V / T_FE
    # Convert to uC/cm^2 by multiplying by 1e6:
    return (eps_r - 1) * 8.854e-8 * V / T_FE


# ======================================================================
# ===========  EDITABLE BLOCK  =========================================
# ======================================================================
if __name__ == "__main__":

    # ---- waveform settings ---------------------------------------
    V_max      = 2.5                      # peak voltage (V)
    t_triangle = 5.0e-3                   # pulse width (s)
    dt_active  = 5.0e-6                   # timestep inside the pulses (s)
    N_grains   = 5000                     # MC ensemble size
    seed       = 42
    s_init     = +1                       # initial state  (+1 -> start at +P_R)
    polarities = (-1, +1, -1, +1)         # pulses ① ② ③ ④

    # ---- hold times to compare -----------------------------------
    hold_cases = [('T_H = 10 s', 10.0),
                  ('T_H = 1 ms', 1.0e-3)]

    # ---- run all cases -------------------------------------------
    data = {}
    for label, T_H in hold_cases:
        t, V, pulse_info = build_multi_pulse(
            V_max, t_triangle, T_H, dt_active, polarities=polarities,
        )
        P_FE, _, _ = run_general_mc(
            t, V, T_FE_HZO, PARAMS_HZO,
            N=N_grains, seed=seed, s_init=s_init,
        )
        P_meas = P_FE + linear_capacitance_contribution(V, T_FE_HZO, EPS_R_LIN)
        data[label] = dict(t=t, V=V, P_FE=P_FE, P_meas=P_meas,
                           pulse_info=pulse_info, T_H=T_H)

    # ---- figure ---------------------------------------------------
    circ = ['\u2460', '\u2461', '\u2462', '\u2463']     # ① ② ③ ④
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2),
                              constrained_layout=True)

    for col, (label, T_H) in enumerate(hold_cases):
        d = data[label]

        # ---- (a) applied waveform -------------------------------
        ax = axes[0, col]
        ax.plot(d['t'], d['V'], lw=1.3, color='C0')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$V_A$ (V)')
        ax.set_title(f'Applied waveform — {label}')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylim(-V_max * 1.4, V_max * 1.4)
        for pi, cl in zip(d['pulse_info'], circ):
            y_lab = pi['sign'] * V_max * 1.18
            ax.text(pi['start'] + t_triangle / 2.0, y_lab, cl,
                    ha='center', va='center', fontsize=15,
                    color='C3', weight='bold')

        # ---- (b/c) P-V hysteresis loop --------------------------
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

        # annotations -- ① ③ on left side (negative switching),
        #               ② ④ on right side (positive switching)
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
        f'$t_\\mathrm{{tri}}$ = {t_triangle * 1e3:g} ms,   '
        f'$\\varepsilon_r^\\mathrm{{lin}}$ = {EPS_R_LIN:g}',
        fontsize=12,
    )
    plt.savefig('algorithm2_4pulse_pv_loops.png', dpi=140,
                bbox_inches='tight')
    plt.show()

    # ---- summary -------------------------------------------------
    print("=" * 64)
    print(f"4-pulse measurement protocol — summary")
    print(f"  V_max = {V_max} V,  t_triangle = {t_triangle*1e3:g} ms,  "
          f"N = {N_grains},  polarities = {polarities}")
    for label, T_H in hold_cases:
        d = data[label]
        print(f"  {label}:  final P_meas = {d['P_meas'][-1]:+6.2f} uC/cm^2  "
              f"(P_FE = {d['P_FE'][-1]:+6.2f})")
    print("=" * 64)
