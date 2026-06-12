"""
Monte Carlo Polarization Reversal Simulation for Ferroelectric Materials
========================================================================

Implements the algorithm:

    Instantiate FE: parameters {P_S, beta, alpha, tau_inf}
    Sample N activation fields E_a^(i) from g(E_a)

    Initialization for grains i = 1..N:
        s^(i)   <- -1
        tau^(i) <- tau_inf * exp( (E_a^(i) / E_app)^alpha )

    For each time step [t, t+dt] and each grain i:
        if s^(i) == -1:
            P^(i) = 1 - exp( (t/tau^(i))^beta - ((t+dt)/tau^(i))^beta )
            if Bernoulli(P^(i)) == 1:
                s^(i) <- +1

This is a Kolmogorov-Avrami-Ishibashi (KAI) / nucleation-limited switching
model. tau is grain-specific (Merz-law form), and the conditional Weibull
hazard between t and t+dt gives the per-step flip probability.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Activation-field distribution g(E_a)
# ----------------------------------------------------------------------
def sample_activation_fields(N, mean=1.2, std=0.2, dist="lognormal", rng=None):
    """Draw N activation fields from g(E_a). Lognormal is a common choice
    (positive support, heavy tail); a truncated Gaussian is also offered."""
    rng = rng or np.random.default_rng()
    if dist == "lognormal":
        # parameterize lognormal so that the mean/std of E_a match `mean`, `std`
        sigma2 = np.log(1.0 + (std / mean) ** 2)
        mu = np.log(mean) - 0.5 * sigma2
        return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=N)
    elif dist == "gaussian":
        Ea = rng.normal(loc=mean, scale=std, size=N)
        return np.clip(Ea, 1e-6, None)
    else:
        raise ValueError(f"unknown distribution: {dist}")


# ----------------------------------------------------------------------
# Core Monte Carlo simulation
# ----------------------------------------------------------------------
def run_simulation(
    N=5_000,            # number of grains
    P_S=22.9,            # spontaneous polarization (uC/cm^2)
    beta=2.07,            # Avrami / Weibull exponent
    alpha=4.11,           # field exponent (Merz law)
    tau_inf=387.0e-9,      # prefactor relaxation time (s)
    E_app=1.77,           # applied electric field (MV/cm)
    Ea_mean=1.79,         # mean of g(E_a)            (MV/cm)
    Ea_std=0.20,         # spread of g(E_a)          (MV/cm)
    t_max=1.0,           # total simulation time (s)
    n_steps=400,         # number of (log-spaced) timesteps
    Ea_dist="lognormal",
    seed=0,
):
    rng = np.random.default_rng(seed)

    # -- 1. Sample activation fields ----------------------------------
    Ea = sample_activation_fields(N, Ea_mean, Ea_std, Ea_dist, rng)

    # -- 2. Initialization --------------------------------------------
    s   = -np.ones(N, dtype=np.int8)                    # all grains at -P_S
    tau = tau_inf * np.exp((Ea / E_app) ** alpha)        # grain-specific tau

    # -- 3. Time grid (log-spaced so we resolve fast and slow grains) -
    t_grid = np.logspace(np.log10(tau_inf * 1e-2),
                         np.log10(t_max), n_steps)
    t_grid = np.insert(t_grid, 0, 0.0)

    polarization      = np.full_like(t_grid, -P_S)
    fraction_switched = np.zeros_like(t_grid)

    # -- 4. Time-stepping loop ----------------------------------------
    for k in range(1, len(t_grid)):
        t0, t1 = t_grid[k - 1], t_grid[k]

        unswitched = (s == -1)
        if not unswitched.any():
            fraction_switched[k:] = 1.0
            polarization[k:]      = P_S
            break

        tau_u = tau[unswitched]
        # conditional Weibull switching probability over [t0, t1]
        P_sw = 1.0 - np.exp((t0 / tau_u) ** beta
                            - (t1 / tau_u) ** beta)
        P_sw = np.clip(P_sw, 0.0, 1.0)

        # Bernoulli draw for each still-unswitched grain
        flips = rng.random(P_sw.size) < P_sw

        # update states
        idx_unsw = np.where(unswitched)[0]
        s[idx_unsw[flips]] = 1

        fraction_switched[k] = (s == 1).mean()
        polarization[k]      = P_S * (2.0 * fraction_switched[k] - 1.0)

    return {
        "t": t_grid,
        "polarization": polarization,
        "fraction_switched": fraction_switched,
        "Ea": Ea,
        "tau": tau,
        "states": s,
    }


# ----------------------------------------------------------------------
# Demo / plotting
# ----------------------------------------------------------------------
if __name__ == "__main__":

    result = run_simulation(
        N=5_000,
        P_S=22.9,
        beta=2.07,
        alpha=4.11,
        tau_inf=387.0e-9,
        E_app=1.77,
        Ea_mean=1.79,
        Ea_std=0.20,
        t_max=1.0,
        n_steps=400,
        Ea_dist="lognormal",
        seed=42,
    )

    t    = result["t"]
    P    = result["polarization"]
    frac = result["fraction_switched"]
    Ea   = result["Ea"]
    tau  = result["tau"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) switched fraction vs time
    axes[0].semilogx(t[1:], frac[1:], lw=2)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("switched fraction")
    axes[0].set_title("KAI switching kinetics")
    axes[0].grid(True, which="both", alpha=0.3)

    # (b) polarization vs time
    axes[1].semilogx(t[1:], P[1:], lw=2, color="C3")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel(r"$P$  ($\mu$C/cm$^2$)")
    axes[1].set_title("Polarization reversal")
    axes[1].grid(True, which="both", alpha=0.3)

    # (c) distribution of grain time constants
    axes[2].hist(np.log10(tau), bins=60, color="C2", alpha=0.85)
    axes[2].set_xlabel(r"$\log_{10}\,\tau$  (s)")
    axes[2].set_ylabel("# grains")
    axes[2].set_title("Distribution of grain relaxation times")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("polarization_reversal.png", dpi=130)
    plt.show()

    # -- quick numerical summary -------------------------------------
    print(f"Final switched fraction : {frac[-1]:.4f}")
    print(f"Final polarization      : {P[-1]:+.3f} uC/cm^2")
    print(f"Median grain tau        : {np.median(tau):.3e} s")
    print(f"Mean activation field   : {Ea.mean():.3f} MV/cm")