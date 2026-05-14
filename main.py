"""
main.py — Black-Scholes Model: Indian Index Options Demo
=========================================================
Walkthrough of BSM features calibrated to Indian markets:

  1. Pricing — Nifty, Bank Nifty, Sensex ATM/ITM/OTM options (INR)
  2. Greeks  — Contract-level INR greeks with lot size
  3. Put-call parity verification
  4. Implied volatility recovery
  5. Sensitivity analysis — India VIX scenarios
  6. Transaction cost breakdown (STT, GST, Exchange charges)
  7. Generate and save all plots

Usage:
    python main.py               # full demo + save all plots
    python main.py --no-plots    # skip plot generation (faster)

Indian Market Parameters Used:
    RBI Repo Rate  : 6.5%  (risk-free rate)
    India VIX      : 14%   (calm market baseline)
    Nifty 50       : 24,500  lot size 75
    Bank Nifty     : 52,000  lot size 30
    Sensex (BSE)   : 81,000  lot size 20
"""

import argparse
import os
import numpy as np

from black_scholes import (
    BlackScholesModel, BSMInputs, INDICES, RBI_REPO_RATE, DEFAULT_INDIA_VIX,
)
from black_scholes.implied_vol import implied_volatility
from black_scholes.visualisations import (
    plot_payoff_diagram, plot_price_surface,
    plot_greeks_profile, plot_vol_smile, plot_pnl_heatmap,
)

PLOTS_DIR = "plots"

# ---- Live-ish Indian index levels (as of mid-2025) ----
NIFTY_LEVEL      = 24_500
BANKNIFTY_LEVEL  = 52_000
SENSEX_LEVEL     = 81_000
RBI_RATE         = RBI_REPO_RATE   # 6.5%
INDIA_VIX        = DEFAULT_INDIA_VIX  # 14%


def banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def section(title: str):
    print(f"\n{'—' * 55}")
    print(f"  {title}")
    print("—" * 55)


def demo_pricing():
    banner("1. OPTION PRICING — INDIAN INDEX OPTIONS (INR)")

    print("\n  All options: European-style, cash-settled (NSE/BSE)")
    print(f"  Risk-free rate: RBI Repo {RBI_RATE*100:.1f}%  |  India VIX: {INDIA_VIX*100:.0f}%\n")

    # Weekly Nifty expiry (7 days)
    scenarios = [
        ("Nifty ATM  weekly",   dict(S=24500, K=24500, T=7/365,  r=RBI_RATE, sigma=INDIA_VIX,  index="NIFTY")),
        ("Nifty ATM  monthly",  dict(S=24500, K=24500, T=30/365, r=RBI_RATE, sigma=INDIA_VIX,  index="NIFTY")),
        ("Nifty ITM  call",     dict(S=24500, K=24000, T=15/365, r=RBI_RATE, sigma=INDIA_VIX,  index="NIFTY")),
        ("Nifty OTM  put",      dict(S=24500, K=24000, T=15/365, r=RBI_RATE, sigma=0.16,        index="NIFTY")),
        ("BankNifty ATM weekly",dict(S=52000, K=52000, T=7/365,  r=RBI_RATE, sigma=0.18,        index="BANKNIFTY")),
        ("Sensex ATM monthly",  dict(S=81000, K=81000, T=30/365, r=RBI_RATE, sigma=INDIA_VIX,  index="SENSEX")),
        ("Nifty High VIX",      dict(S=24500, K=24500, T=30/365, r=RBI_RATE, sigma=0.30,        index="NIFTY")),
    ]

    print(f"  {'Scenario':<24} {'Index':>5} {'S':>7} {'K':>7} {'T(d)':>5} "
          f"{'σ':>5}  {'Call/unit':>10} {'Put/unit':>10} {'C-contract':>12} {'P-contract':>12}")
    print("  " + "-" * 105)

    for name, kw in scenarios:
        inp = BSMInputs(**kw)
        res = BlackScholesModel(inp).price()
        lot = inp.lot_size
        print(
            f"  {name:<24} {kw['index']:>5} {kw['S']:>7,.0f} {kw['K']:>7,.0f} "
            f"{kw['T']*365:>5.0f} {kw['sigma']*100:>4.0f}%  "
            f"₹{res.call_price:>9,.2f}  ₹{res.put_price:>9,.2f}  "
            f"₹{res.call_premium:>10,.0f}  ₹{res.put_premium:>10,.0f}"
        )

    section("Full Summary — Nifty ATM 30-day option")
    params = BSMInputs(S=NIFTY_LEVEL, K=NIFTY_LEVEL, T=30/365, r=RBI_RATE,
                       sigma=INDIA_VIX, index="NIFTY")
    print(BlackScholesModel(params).summary())


def demo_greeks():
    banner("2. THE GREEKS — CONTRACT LEVEL (INR)")

    params = BSMInputs(S=NIFTY_LEVEL, K=NIFTY_LEVEL, T=30/365,
                       r=RBI_RATE, sigma=INDIA_VIX, index="NIFTY")
    g   = BlackScholesModel(params).greeks()
    lot = params.lot_size

    print(f"""
  Nifty ATM call | S={NIFTY_LEVEL:,}  K={NIFTY_LEVEL:,}  T=30d  σ={INDIA_VIX*100:.0f}%  lot={lot}

  Δ Delta    Call: {g.delta_call:+.4f}   Put: {g.delta_put:+.4f}
             Contract: Call Δ = {g.delta_call_contract:+.1f} index pts/contract
             If Nifty moves 100 pts, call gains ₹{g.delta_call_contract*100:,.0f}/contract

  Γ Gamma        {g.gamma:.8f}  (same for C & P)
             Delta changes by {g.gamma:.6f} per 1-pt Nifty move

  Θ Theta    Call: ₹{g.theta_call_contract:,.0f}/day/contract   Put: ₹{g.theta_put_contract:,.0f}/day/contract
             Call loses ₹{abs(g.theta_call_contract):,.0f} per calendar day  (time decay)

  ν Vega     ₹{g.vega_contract:,.0f} per 1% India VIX move / contract
             If India VIX rises from {INDIA_VIX*100:.0f}% to {INDIA_VIX*100+1:.0f}%, gain ₹{g.vega_contract:,.0f}/contract

  ρ Rho      Call: {g.rho_call:+.2f}   Put: {g.rho_put:+.2f} per 1% RBI rate change

  Vanna  {g.vanna:+.4f}  (dDelta/dVol — how delta shifts if VIX moves)
  Volga  {g.volga:+.4f}  (dVega/dVol  — convexity of vega to vol)
""")


def demo_parity():
    banner("3. PUT-CALL PARITY — VERIFICATION (Indian Options)")
    print("\n  Equation: C − P = S − K·e^{−rT}  (holds for European options)\n")
    print(f"  {'Scenario':<20} {'LHS (C-P)':>12}  {'RHS (S-Ke^-rT)':>14}  {'Diff':>10}  Status")
    print("  " + "-" * 75)
    for S, label in [(24000, "OTM"), (24500, "ATM"), (25000, "ITM")]:
        params = BSMInputs(S=S, K=24500, T=30/365, r=RBI_RATE, sigma=INDIA_VIX, index="NIFTY")
        pcp    = BlackScholesModel(params).put_call_parity()
        status = "HOLDS" if pcp["parity_holds"] else "VIOLATED"
        print(f"  {'Nifty '+label:<20} {pcp['lhs (C - P)']:>12.4f}  "
              f"{pcp['rhs (S - Ke^-rT)']:>14.4f}  {pcp['difference']:.2e}  {status}")


def demo_implied_vol():
    banner("4. IMPLIED VOLATILITY SOLVER — NIFTY OPTIONS")
    print("\n  Round-trip: price option with known σ → recover IV from market price\n")
    print(f"  {'True σ':>8}  {'IV zone':>12}  {'Call price':>12}  {'Recovered σ':>12}  {'Error':>10}")
    print("  " + "-" * 65)

    vix_zones = [(0.10, "Very low"), (0.14, "Calm (VIX~14)"),
                 (0.20, "Moderate"), (0.30, "Elevated"),
                 (0.40, "High stress"), (0.60, "Crisis")]
    for true_sigma, zone in vix_zones:
        params    = BSMInputs(S=NIFTY_LEVEL, K=NIFTY_LEVEL, T=30/365,
                              r=RBI_RATE, sigma=true_sigma, index="NIFTY")
        mkt_price = BlackScholesModel(params).price().call_price
        iv_result = implied_volatility(mkt_price, S=NIFTY_LEVEL, K=NIFTY_LEVEL,
                                       T=30/365, r=RBI_RATE)
        rec   = iv_result["implied_vol"]
        err   = abs(rec - true_sigma) if rec else float("nan")
        print(f"  {true_sigma*100:>6.0f}%  {zone:>12}  "
              f"₹{mkt_price:>10,.2f}  {rec*100 if rec else 0:>10.4f}%  {err:.2e}")


def demo_sensitivity():
    banner("5. SENSITIVITY — NIFTY CALL PRICE vs INDEX LEVEL x DAYS TO EXPIRY")
    print("\n  Strike K=24500  σ=14% (India VIX)  r=6.5% (RBI repo)\n")

    T_vals = [2, 7, 15, 30, 45]    # typical NSE expiry days
    S_vals = [23500, 24000, 24250, 24500, 24750, 25000, 25500]

    print(f"  {'S\\T →':>6}", end="")
    for T in T_vals:
        print(f"  {T}d expiry", end="")
    print()
    print("  " + "-" * 70)

    for S in S_vals:
        lbl = " (ATM)" if S == 24500 else ""
        print(f"  {S:>6,}{lbl:<6}", end="")
        for T in T_vals:
            params = BSMInputs(S=S, K=24500, T=T/365, r=RBI_RATE,
                               sigma=INDIA_VIX, index="NIFTY")
            res = BlackScholesModel(params).price()
            print(f"  ₹{res.call_price:>8,.2f}", end="")
        print()


def demo_costs():
    banner("6. TRANSACTION COSTS — INDIAN OPTIONS (SEBI/NSE)")
    print("\n  Breakdown for buying 1 Nifty ATM call contract\n")

    scenarios = [
        ("Nifty weekly ATM",   dict(S=24500, K=24500, T=7/365,  r=RBI_RATE, sigma=0.14, index="NIFTY")),
        ("Nifty monthly ATM",  dict(S=24500, K=24500, T=30/365, r=RBI_RATE, sigma=0.14, index="NIFTY")),
        ("BankNifty weekly",   dict(S=52000, K=52000, T=7/365,  r=RBI_RATE, sigma=0.18, index="BANKNIFTY")),
        ("Sensex monthly",     dict(S=81000, K=81000, T=30/365, r=RBI_RATE, sigma=0.14, index="SENSEX")),
    ]

    print(f"  {'Scenario':<22} {'Lot':>4} {'Premium/unit':>13} {'Contract val':>13} "
          f"{'STT':>8} {'Exch+GST':>10} {'Stamp':>8} {'Total cost':>11}")
    print("  " + "-" * 100)

    for name, kw in scenarios:
        inp   = BSMInputs(**kw)
        bsm   = BlackScholesModel(inp)
        r     = bsm.price()
        costs = bsm.transaction_costs("call")
        lot   = inp.lot_size
        cval  = r.call_price * lot
        print(f"  {name:<22} {lot:>4}  ₹{r.call_price:>10,.2f}  ₹{cval:>11,.0f}  "
              f"₹{costs.stt:>6.2f}  ₹{costs.exchange_charge+costs.gst:>8.4f}  "
              f"₹{costs.stamp_duty:>6.2f}  ₹{costs.total:>9.2f}")


def generate_plots():
    banner("7. GENERATING PLOTS — INDIAN INDEX OPTIONS")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n  Saving to ./{PLOTS_DIR}/\n")

    # Nifty realistic IV smile with NSE put skew
    # OTM puts: VIX+4–8 pts higher (downside protection premium)
    nifty_atm   = NIFTY_LEVEL
    strike_offsets = [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]
    K_list = [nifty_atm + o for o in strike_offsets]
    # Realistic NSE skew: steep put skew, mild call skew
    iv_list = [0.22, 0.20, 0.18, 0.16, 0.14, 0.145, 0.15, 0.155, 0.17]

    print("  [1/5] Payoff diagram (Nifty) ...", end=" ", flush=True)
    plot_payoff_diagram(S=NIFTY_LEVEL, K=NIFTY_LEVEL, T=30/365,
                        r=RBI_RATE, sigma=INDIA_VIX, index="NIFTY",
                        save_path=f"{PLOTS_DIR}/01_payoff_diagram.png")
    print("saved.")

    print("  [2/5] 3-D price surface (Nifty) ...", end=" ", flush=True)
    plot_price_surface(K=NIFTY_LEVEL, r=RBI_RATE, sigma=INDIA_VIX,
                       index="NIFTY", option_type="call",
                       save_path=f"{PLOTS_DIR}/02_price_surface.png")
    print("saved.")

    print("  [3/5] Greeks profile (Nifty) ...", end=" ", flush=True)
    plot_greeks_profile(K=NIFTY_LEVEL, T=30/365, r=RBI_RATE,
                        sigma=INDIA_VIX, index="NIFTY",
                        save_path=f"{PLOTS_DIR}/03_greeks_profile.png")
    print("saved.")

    print("  [4/5] IV smile (NSE put skew) ...", end=" ", flush=True)
    plot_vol_smile(S=NIFTY_LEVEL, K_list=K_list, T=30/365,
                   r=RBI_RATE, iv_list=iv_list, index="NIFTY",
                   save_path=f"{PLOTS_DIR}/04_vol_smile.png")
    print("saved.")

    print("  [5/5] P&L heatmap — India VIX scenarios ...", end=" ", flush=True)
    plot_pnl_heatmap(S=NIFTY_LEVEL, K=NIFTY_LEVEL, T=30/365,
                     r=RBI_RATE, sigma=INDIA_VIX, index="NIFTY",
                     option_type="call", position="long",
                     save_path=f"{PLOTS_DIR}/05_pnl_heatmap.png")
    print("saved.")

    print(f"\n  All plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSM Indian Index Options Demo")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print("\n" + "█" * 65)
    print("  BLACK-SCHOLES OPTIONS PRICING — INDIAN MARKETS EDITION")
    print("  NSE Nifty | Bank Nifty | BSE Sensex | Midcap Nifty")
    print("█" * 65)

    demo_pricing()
    demo_greeks()
    demo_parity()
    demo_implied_vol()
    demo_sensitivity()
    demo_costs()

    if not args.no_plots:
        generate_plots()
    else:
        print("\n  [Plots skipped — run without --no-plots to generate them]")

    print("\n" + "█" * 65)
    print("  Done. Run `pytest tests/ -v` to execute the test suite.")
    print("█" * 65 + "\n")
