"""
Visualisations — Indian Index Options Edition
=============================================
All matplotlib plotting functions adapted for NSE/BSE index options:
  - INR (₹) currency labels
  - Nifty / Sensex / Bank Nifty realistic index levels
  - India VIX-calibrated volatility smile (pronounced skew)
  - Contract-level P&L (lot size multiplied)
  - SEBI-aligned expiry annotation (weekly Thursday)

Functions
---------
plot_payoff_diagram   — Call/put payoff with INR contract premiums
plot_price_surface    — 3-D price surface over realistic Nifty range
plot_greeks_profile   — Greeks vs index level with contract-level values
plot_vol_smile        — India VIX smile with realistic NSE skew
plot_pnl_heatmap      — P&L heatmap: index level x India VIX scenario
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from .model import BlackScholesModel, BSMInputs, INDICES


# ---------------------------------------------------------------------------
# Shared style — saffron/navy Indian market aesthetic
# ---------------------------------------------------------------------------

CALL_COLOR   = "#1A237E"   # deep navy  (NSE blue)
PUT_COLOR    = "#B71C1C"   # deep red
ACCENT_SAFFRON = "#FF6F00" # saffron
ACCENT_GREEN   = "#1B5E20" # dark green (profit)
GRID_COLOR   = "#E8EAF6"

STYLE = "seaborn-v0_8-whitegrid"


def _style():
    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.dpi":       120,
        "axes.facecolor":   "#FAFAFA",
        "figure.facecolor": "white",
    })


def _inr(v: float) -> str:
    """Format a value as INR with comma separation."""
    if abs(v) >= 1e5:
        return f"₹{v:,.0f}"
    return f"₹{v:,.2f}"


# ---------------------------------------------------------------------------
# 1. Payoff diagram — Indian index options
# ---------------------------------------------------------------------------

def plot_payoff_diagram(S: float, K: float, T: float, r: float, sigma: float,
                        index: str = "NIFTY",
                        save_path: str = None) -> plt.Figure:
    """
    Plot call and put payoff vs BSM price for Indian index options.
    Y-axis shows per-unit (index point) values; secondary annotation shows
    contract value in INR (lot size multiplied).
    """
    _style()
    lot   = INDICES[index.upper()]["lot_size"]
    exch  = INDICES[index.upper()]["exchange"]

    # Realistic spot range: +/- 15% around current index level
    lo, hi      = K * 0.80, K * 1.20
    spot_range  = np.linspace(lo, hi, 300)

    call_payoff = np.maximum(spot_range - K, 0)
    put_payoff  = np.maximum(K - spot_range, 0)
    call_prices = [BlackScholesModel(BSMInputs(s, K, T, r, sigma, index)).price().call_price
                   for s in spot_range]
    put_prices  = [BlackScholesModel(BSMInputs(s, K, T, r, sigma, index)).price().put_price
                   for s in spot_range]

    # Entry premiums
    entry       = BlackScholesModel(BSMInputs(S, K, T, r, sigma, index)).price()
    call_entry  = entry.call_price
    put_entry   = entry.put_price

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        f"Black-Scholes Payoff Diagram  |  {index.upper()} {exch}  "
        f"K={K:,.0f}  T={T*365:.0f}d  σ={sigma*100:.0f}%  r={r*100:.1f}% (RBI repo)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, payoff, prices, entry_p, color, label in [
        (axes[0], call_payoff, call_prices, call_entry, CALL_COLOR, "Call"),
        (axes[1], put_payoff,  put_prices,  put_entry,  PUT_COLOR,  "Put"),
    ]:
        ax.fill_between(spot_range, 0, payoff, alpha=0.08, color=color)
        ax.plot(spot_range, payoff,  color=color, linestyle="--", lw=1.5,
                label=f"{label} payoff at expiry")
        ax.plot(spot_range, prices, color=color, lw=2.5,
                label=f"{label} BSM price today")

        ax.axvline(K, color="gray",  lw=1.0, linestyle=":",  label=f"Strike K={K:,.0f}")
        ax.axvline(S, color="black", lw=1.0, linestyle="-.", alpha=0.6,
                   label=f"Current S={S:,.0f}")
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)

        # Annotate entry premium
        ax.annotate(
            f"Premium: ₹{entry_p:.1f}/unit\nContract: ₹{entry_p*lot:,.0f}  (lot {lot})",
            xy=(S, entry_p),
            xytext=(S + (hi - lo) * 0.05, entry_p + (max(payoff) * 0.15)),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

        ax.set_xlabel(f"{index.upper()} Index Level at Expiry")
        ax.set_ylabel("Value (₹ per unit / index point)")
        ax.set_title(f"{'Long ' + label} Option")
        ax.legend(loc="upper left" if label == "Call" else "upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. 3-D price surface — realistic Nifty/Sensex levels
# ---------------------------------------------------------------------------

def plot_price_surface(K: float, r: float, sigma: float,
                       index: str = "NIFTY",
                       option_type: str = "call",
                       save_path: str = None) -> plt.Figure:
    """
    3-D call/put price surface: index level (S) x time to expiry (T).
    Covers realistic weekly-to-monthly expiry range for NSE/BSE options.
    """
    _style()
    lot  = INDICES[index.upper()]["lot_size"]
    exch = INDICES[index.upper()]["exchange"]

    # Realistic range: 85%–115% of strike for index options
    S_vals = np.linspace(K * 0.82, K * 1.18, 55)
    # Weekly (2d) to 3-month expiry
    T_vals = np.linspace(2/365, 90/365, 55)
    SS, TT = np.meshgrid(S_vals, T_vals)

    ZZ = np.zeros_like(SS)
    for i in range(SS.shape[0]):
        for j in range(SS.shape[1]):
            res = BlackScholesModel(BSMInputs(SS[i,j], K, TT[i,j], r, sigma, index)).price()
            ZZ[i,j] = res.call_price if option_type == "call" else res.put_price

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Saffron -> navy colormap
    cmap = plt.cm.RdYlBu_r
    surf = ax.plot_surface(SS, TT * 365, ZZ, cmap=cmap, alpha=0.92,
                           linewidth=0, antialiased=True)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10,
                         label="Option Price (₹ per unit)")
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))

    ax.set_xlabel(f"{index.upper()} Level (S)", labelpad=10)
    ax.set_ylabel("Days to Expiry", labelpad=10)
    ax.set_zlabel("Option Price (₹/unit)", labelpad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(
        f"{'Call' if option_type == 'call' else 'Put'} Price Surface  "
        f"|  {index.upper()} {exch}  K={K:,.0f}  σ={sigma*100:.0f}%  r={r*100:.1f}%",
        fontsize=13, fontweight="bold"
    )
    ax.view_init(elev=28, azim=-55)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Greeks profile — contract-level annotation
# ---------------------------------------------------------------------------

def plot_greeks_profile(K: float, T: float, r: float, sigma: float,
                        index: str = "NIFTY",
                        save_path: str = None) -> plt.Figure:
    """
    4-panel Greeks plot vs index level.
    Shows per-unit Greeks on main axis and contract-level INR values annotated.
    """
    _style()
    lot  = INDICES[index.upper()]["lot_size"]
    exch = INDICES[index.upper()]["exchange"]

    spot_range = np.linspace(K * 0.80, K * 1.20, 300)

    def get_g(s):
        return BlackScholesModel(BSMInputs(s, K, T, r, sigma, index)).greeks()

    greeks_list  = [get_g(s) for s in spot_range]
    delta_c = [g.delta_call      for g in greeks_list]
    delta_p = [g.delta_put       for g in greeks_list]
    gamma   = [g.gamma           for g in greeks_list]
    theta_c = [g.theta_call_contract for g in greeks_list]  # contract INR/day
    theta_p = [g.theta_put_contract  for g in greeks_list]
    vega_c  = [g.vega_contract       for g in greeks_list]  # contract INR per 1%

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Option Greeks vs {index.upper()} Level  "
        f"|  {index.upper()} {exch}  K={K:,.0f}  T={T*365:.0f}d  σ={sigma*100:.0f}%  r={r*100:.1f}%",
        fontsize=13, fontweight="bold"
    )

    def _fmt_x(ax):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.axvline(K, color="gray", lw=0.8, linestyle=":", alpha=0.6)

    # Delta
    ax = axes[0, 0]
    ax.plot(spot_range, delta_c, color=CALL_COLOR, lw=2, label="Call Δ")
    ax.plot(spot_range, delta_p, color=PUT_COLOR,  lw=2, label="Put Δ")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Δ Delta  (unitless, 0–1)"); ax.set_ylabel("Delta"); ax.legend()
    _fmt_x(ax)

    # Gamma
    ax = axes[0, 1]
    ax.plot(spot_range, gamma, color=ACCENT_SAFFRON, lw=2, label="Γ (same for C & P)")
    ax.fill_between(spot_range, 0, gamma, alpha=0.12, color=ACCENT_SAFFRON)
    ax.set_title("Γ Gamma  (per index point²)"); ax.set_ylabel("Gamma"); ax.legend()
    _fmt_x(ax)

    # Theta — contract level INR/day
    ax = axes[1, 0]
    ax.plot(spot_range, theta_c, color=CALL_COLOR, lw=2, label="Call Θ (₹/day/contract)")
    ax.plot(spot_range, theta_p, color=PUT_COLOR,  lw=2, label="Put Θ (₹/day/contract)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"Θ Theta — INR per Day  (lot {lot})")
    ax.set_ylabel("Theta (₹/day per contract)")
    ax.set_xlabel(f"{index.upper()} Index Level")
    ax.legend(); _fmt_x(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))

    # Vega — contract level INR per 1% vol
    ax = axes[1, 1]
    ax.plot(spot_range, vega_c, color="#4A148C", lw=2,
            label="ν Vega (₹ per 1% σ / contract)")
    ax.fill_between(spot_range, 0, vega_c, alpha=0.10, color="#4A148C")
    ax.set_title(f"ν Vega — INR per 1% India VIX  (lot {lot})")
    ax.set_ylabel("Vega (₹ per 1% σ per contract)")
    ax.set_xlabel(f"{index.upper()} Index Level")
    ax.legend(); _fmt_x(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Volatility smile — India VIX calibrated, realistic NSE skew
# ---------------------------------------------------------------------------

def plot_vol_smile(S: float, K_list: list, T: float, r: float,
                   iv_list: list,
                   index: str = "NIFTY",
                   save_path: str = None) -> plt.Figure:
    """
    Plot India VIX-calibrated IV smile with realistic NSE put skew.
    OTM puts command higher IV (downside protection demand from institutions).
    """
    _style()
    exch      = INDICES[index.upper()]["exchange"]
    moneyness = [np.log(S / K) for K in K_list]
    iv_pct    = [iv * 100 for iv in iv_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Implied Volatility Smile  |  {index.upper()} {exch}  "
        f"S={S:,.0f}  T={T*365:.0f}d  r={r*100:.1f}% (RBI repo)",
        fontsize=13, fontweight="bold"
    )

    # Left: IV vs Strike
    ax1.plot(K_list, iv_pct, "o-", color=CALL_COLOR, lw=2, ms=7, zorder=3)
    ax1.fill_between(K_list, min(iv_pct), iv_pct, alpha=0.08, color=CALL_COLOR)
    ax1.axvline(S, color="red", lw=1.5, linestyle="--", alpha=0.8, label=f"ATM (S={S:,.0f})")
    # Shade OTM puts (left of ATM) — typically higher IV on NSE
    otm_strikes = [k for k in K_list if k <= S]
    otm_ivs     = [iv_pct[i] for i, k in enumerate(K_list) if k <= S]
    ax1.fill_between(otm_strikes, min(iv_pct) - 1, otm_ivs,
                     alpha=0.05, color=PUT_COLOR, label="Put skew zone")
    ax1.set_xlabel(f"{index.upper()} Strike (K)")
    ax1.set_ylabel("Implied Volatility (%)")
    ax1.set_title("IV vs Strike  (India VIX calibrated)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    # Annotate ATM IV
    atm_iv = iv_pct[len(iv_pct) // 2]
    ax1.annotate(f"ATM IV\n{atm_iv:.1f}%", xy=(S, atm_iv),
                 xytext=(S + (K_list[-1] - K_list[0]) * 0.08, atm_iv + 1.5),
                 fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    # Right: IV vs Log-moneyness
    ax2.plot(moneyness, iv_pct, "o-", color="#880E4F", lw=2, ms=7, zorder=3)
    ax2.fill_between(moneyness, min(iv_pct), iv_pct, alpha=0.08, color="#880E4F")
    ax2.axvline(0, color="red", lw=1.5, linestyle="--", alpha=0.8, label="ATM (ln S/K=0)")
    ax2.axvspan(min(moneyness), 0, alpha=0.04, color=PUT_COLOR, label="OTM puts (left)")
    ax2.set_xlabel("Log-Moneyness ln(S/K)")
    ax2.set_ylabel("Implied Volatility (%)")
    ax2.set_title("IV vs Log-Moneyness  (NSE put skew visible)")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. P&L heatmap — INR contract-level, India VIX scenarios
# ---------------------------------------------------------------------------

def plot_pnl_heatmap(S: float, K: float, T: float, r: float, sigma: float,
                     index: str = "NIFTY",
                     option_type: str = "call", position: str = "long",
                     save_path: str = None) -> plt.Figure:
    """
    P&L heatmap in INR per contract:
      rows    = India VIX scenarios (8%–50%)
      columns = index level scenarios (S ± 10%)
    """
    _style()
    lot  = INDICES[index.upper()]["lot_size"]
    exch = INDICES[index.upper()]["exchange"]

    entry     = BlackScholesModel(BSMInputs(S, K, T, r, sigma, index)).price()
    entry_p   = entry.call_price if option_type == "call" else entry.put_price
    entry_contract = entry_p * lot

    # Realistic NSE range: tighter index window, VIX 8–50%
    S_vals   = np.linspace(S * 0.88, S * 1.12, 50)
    vol_vals = np.linspace(0.08, 0.50, 40)
    PNL      = np.zeros((len(vol_vals), len(S_vals)))

    for i, v in enumerate(vol_vals):
        for j, s in enumerate(S_vals):
            res = BlackScholesModel(BSMInputs(s, K, T, r, v, index)).price()
            price = res.call_price if option_type == "call" else res.put_price
            pnl_unit = (price - entry_p) * (1 if position == "long" else -1)
            PNL[i, j] = pnl_unit * lot   # INR per contract

    # Custom diverging colormap: red (loss) -> yellow -> green (profit)
    colors_rg = [(0.8, 0.1, 0.1), (1, 0.95, 0.5), (0.1, 0.5, 0.1)]
    cmap_rg   = LinearSegmentedColormap.from_list("indian_pnl", colors_rg, N=256)

    fig, ax = plt.subplots(figsize=(13, 6))
    vmax = np.percentile(np.abs(PNL), 95)
    im = ax.imshow(
        PNL, aspect="auto", origin="lower",
        cmap=cmap_rg, vmin=-vmax, vmax=vmax,
        extent=[S_vals[0], S_vals[-1], vol_vals[0]*100, vol_vals[-1]*100]
    )

    cbar = fig.colorbar(im, ax=ax, label="P&L per contract (₹)")
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))

    ax.axvline(S, color="white",  lw=1.8, linestyle="--", alpha=0.9,
               label=f"Entry S={S:,.0f}")
    ax.axhline(sigma*100, color="white", lw=1.8, linestyle=":", alpha=0.9,
               label=f"Entry σ={sigma*100:.0f}% (India VIX)")

    # Mark breakeven region
    be_mask = np.abs(PNL) < vmax * 0.05
    Y_coords = np.linspace(vol_vals[0]*100, vol_vals[-1]*100, len(vol_vals))
    X_coords = np.linspace(S_vals[0], S_vals[-1], len(S_vals))

    ax.set_xlabel(f"{index.upper()} Index Level")
    ax.set_ylabel("India VIX — Implied Volatility (%)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(
        f"P&L Heatmap (₹ per contract, lot {lot})  —  {position.title()} {option_type.title()}  "
        f"|  {index.upper()} {exch}  K={K:,.0f}  T={T*365:.0f}d  "
        f"Entry ₹{entry_contract:,.0f}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper left", framealpha=0.7, fontsize=9)

    # Add India VIX zone labels on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([10, 15, 20, 30, 40, 50])
    ax2.set_yticklabels(["VIX 10\n(calm)", "VIX 15", "VIX 20\n(typical)",
                          "VIX 30\n(stress)", "VIX 40", "VIX 50\n(crisis)"],
                         fontsize=7)
    ax2.set_ylabel("India VIX Zone", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
