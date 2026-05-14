"""
black_scholes — Indian Index Options Edition
=============================================
Black-Scholes-Merton pricing adapted for NSE/BSE index options:
  Nifty 50 | Bank Nifty | Sensex | Midcap Nifty | Fin Nifty

- INR-denominated pricing (per unit + per contract)
- RBI repo rate as risk-free rate
- India VIX-calibrated volatility
- SEBI-compliant lot sizes & transaction costs

Quick Start
-----------
>>> from black_scholes import BlackScholesModel, BSMInputs
>>> # Nifty ATM weekly call
>>> params = BSMInputs(S=24500, K=24500, T=7/365, r=0.065, sigma=0.14, index='NIFTY')
>>> bsm = BlackScholesModel(params)
>>> print(bsm.summary())
"""

from .model import BlackScholesModel, BSMInputs, BSMResult, Greeks, INDICES, RBI_REPO_RATE, DEFAULT_INDIA_VIX
from .implied_vol import implied_volatility, iv_surface_row
from .visualisations import (
    plot_payoff_diagram,
    plot_price_surface,
    plot_greeks_profile,
    plot_vol_smile,
    plot_pnl_heatmap,
)

__version__ = "1.0.0"
__author__  = "Black-Scholes Quant Project"

__all__ = [
    "BlackScholesModel", "BSMInputs", "BSMResult", "Greeks",
    "implied_volatility", "iv_surface_row",
    "plot_payoff_diagram", "plot_price_surface", "plot_greeks_profile",
    "plot_vol_smile", "plot_pnl_heatmap",
]
