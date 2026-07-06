"""
Monte Carlo Polarization Reversal in HZO  vs.  Analytic NLS Model
=================================================================

Reproduces the comparison:
  * MC simulation with N = 5000 grains  -> indistinguishable from NLS
  * 10 MC runs with N = 100 grains      -> visible scatter around NLS
  * NLS analytic model                  -> reference curve

Parameters are taken from Table I, column g(E_a'):
    a = 12.1,   b = 1.79 MV/cm,   p = 0.691,   q = 0.633
    P_R = 22.9 uC/cm^2,   tau_inf = 387 ns
    alpha = 4.11,   beta = 2.07

g(E_a') is a Generalized Beta of the second kind (GB2) distribution:

                 (a/b) (x/b)^(a p - 1)
    g(x) = ---------------------------------
            B(p, q) [1 + (x/b)^a]^(p + q)

Switching algorithm (Algorithm 1):
    tau_i = tau_inf * exp( (E_a_i / E_app)^alpha )
    For each [t, t+dt]:
        P_i  = 1 - exp( (t/tau_i)^beta - ((t+dt)/tau_i)^beta )
        if Bernoulli(P_i): s_i <- +1

HZO film with T_FE = 8.3 nm.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_fn


# ----------------------------------------------------------------------
# Table I  --  g(E_a')  parameters
# ----------------------------------------------------------------------
PARAMS = dict(
    a       = 12.1,
    b       = 1.79,        # MV/cm
    p       = 0.691,
    q       = 0.633,
    P_R     = 22.9,        # uC/cm^2
    tau_inf = 387e-9,      # s
    alpha   = 4.11,
    beta    = 2.07,
)

E_APP = 3.0   # applied electric field in MV/cm (HZO, T_FE = 8.3 nm)


# ----------------------------------------------------------------------
# GB2 distribution  (Generalized Beta of the Second Kind)
# ----------------------------------------------------------------------
def gb2_pdf(x, a, b, p, q):
    """PDF of GB2(a, b, p, q)."""
    return (a / b) * (x / b) ** (a * p - 1) \
           / (beta_fn(p, q) * (1.0 + (x / b) ** a) ** (p + q))


def gb2_sample(N, a, b, p, q, rng):
    """Sample via the Beta-prime transformation:
       Y ~ Beta(p, q)  =>  X = b * (Y / (1 - Y))**(1/a)  ~  GB2(a, b, p, q)."""
    y = rng.beta(p, q, size=N)
    return b * (y / (1.0 - y)) ** (1.0 / a)


# ----------------------------------------------------------------------
# Monte Carlo  --  Algorithm 1
# ----------------------------------------------------------------------
def run_mc(N, E_app, params, t_grid, seed):
    rng = np.random.default_rng(seed)

    # 1. Sample activation fields from g(E_a')
    Ea  = gb2_sample(N, params['a'], params['b'],
                     params['p'], params['q'], rng)

    # 2. Initialization
    s   = -np.ones(N, dtype=np.int8)                       # all grains at -P_R/-P_S (Polarization reverse state or saturated state)
    tau = params['tau_inf'] * np.exp((Ea / E_app) ** params['alpha'])

    frac = np.zeros_like(t_grid)
    beta_w = params['beta']

    # 3. Time-stepping
    for k in range(1, len(t_grid)):
        t0, t1 = t_grid[k - 1], t_grid[k]
        unsw = (s == -1)
        if not unsw.any():
            frac[k:] = 1.0
            break

        tau_u = tau[unsw] #unswitched grains' time constants
        P_sw  = 1.0 - np.exp((t0 / tau_u) ** beta_w
                             - (t1 / tau_u) ** beta_w)
        P_sw  = np.clip(P_sw, 0.0, 1.0)

        flips = rng.random(P_sw.size) < P_sw
        s[np.where(unsw)[0][flips]] = 1
        frac[k] = (s == 1).mean() 

    polarization = params['P_R'] * (2.0 * frac - 1.0)
    return frac, polarization


# ----------------------------------------------------------------------
# Analytic NLS model
#     frac(t) = integral_0^inf [1 - exp(-(t/tau(E))^beta)] g(E) dE
# ----------------------------------------------------------------------
def nls_analytic(t_array, E_app, params, n_E=8000, E_max_factor=25):
    a, b, p, q = params['a'], params['b'], params['p'], params['q']
    alpha, beta_w = params['alpha'], params['beta']
    tau_inf, P_R  = params['tau_inf'], params['P_R']

    E_grid   = np.linspace(1e-4, E_max_factor * b, n_E)
    g_vals   = gb2_pdf(E_grid, a, b, p, q)
    tau_vals = tau_inf * np.exp((E_grid / E_app) ** alpha)

    # Vectorized trapezoidal integration over E for each t
    t_col   = t_array[:, None]
    tau_row = tau_vals[None, :]
    integrand = (1.0 - np.exp(-(t_col / tau_row) ** beta_w)) * g_vals[None, :]
    frac = np.trapz(integrand, E_grid, axis=1)

    polarization = P_R * (2.0 * frac - 1.0)
    return frac, polarization


# ----------------------------------------------------------------------
# Demo / figure
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # log-spaced time grid (with t=0 prepended)
    t_grid = np.logspace(-9, -2, 350)
    t_grid = np.insert(t_grid, 0, 0.0)

    # 1) NLS analytic reference
    frac_nls, P_nls = nls_analytic(t_grid, E_APP, PARAMS)

    # 2) MC with N = 5000  (one run)
    frac_5k, P_5k = run_mc(5000, E_APP, PARAMS, t_grid, seed=42)

    # 3) MC with N = 100, 10 independent runs
    runs_100_frac = []
    runs_100_P    = []
    for k in range(10):
        f, P = run_mc(100, E_APP, PARAMS, t_grid, seed=1000 + k)
        runs_100_frac.append(f)
        runs_100_P.append(P)
    runs_100_frac = np.array(runs_100_frac)
    runs_100_P    = np.array(runs_100_P)

    # -------- plot --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    # (a)  MC N=5000  vs  NLS
    ax = axes[0]
    ax.semilogx(t_grid[1:], P_nls[1:], 'k-',  lw=2.6, label='NLS analytic', zorder=3)
    ax.semilogx(t_grid[1:], P_5k[1:],  'o',   ms=3.2, color='C0',
                alpha=0.85, label='MC,  N = 5000', zorder=4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'Polarization  $P$  ($\mu$C/cm$^2$)')
    ax.set_title(f'MC (N = 5000) vs NLS   —   $E_\\mathrm{{app}}$ = {E_APP} MV/cm')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(-PARAMS['P_R'] * 1.08, PARAMS['P_R'] * 1.08)

    # (b)  10 runs of N=100  vs  NLS
    ax = axes[1]
    for r in runs_100_P:
        ax.semilogx(t_grid[1:], r[1:], color='C1', lw=1.0, alpha=0.55)
    ax.semilogx(t_grid[1:], P_nls[1:], 'k-', lw=2.6, label='NLS analytic', zorder=4)
    ax.plot([], [], color='C1', lw=1.4, alpha=0.85,
            label='MC,  N = 100  (10 runs)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('time (s)')
    ax.set_title('Stochastic scatter at small N')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('HZO polarization reversal  ($T_\\mathrm{FE}$ = 8.3 nm) — '
                 'Algorithm 1', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig('mc_vs_nls_hzo.png', dpi=140, bbox_inches='tight')
    plt.show()

    # -- numerical summary -------------------------------------------
    print("="*60)
    print(f"Applied field  E_app = {E_APP} MV/cm")
    print(f"NLS  final P  = {P_nls[-1]:+7.3f}  uC/cm^2")
    print(f"MC   N=5000   = {P_5k[-1]:+7.3f}  uC/cm^2"
          f"   (deviation {P_5k[-1]-P_nls[-1]:+.3f})")
    print(f"MC   N=100    (10 runs):")
    for k, Pf in enumerate(runs_100_P[:, -1]):
        print(f"    run {k+1:2d}:  {Pf:+7.3f}  uC/cm^2"
              f"   (deviation {Pf-P_nls[-1]:+.3f})")
    print(f"    spread (std)  = {runs_100_P[:, -1].std():.3f}  uC/cm^2")
    print("="*60)