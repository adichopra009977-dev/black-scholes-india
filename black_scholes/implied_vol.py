"""
Implied Volatility Solver
==========================
Invert the Black-Scholes formula to recover the market-implied volatility
from an observed option price using two methods:

1. Newton-Raphson  — fast convergence near ATM, uses Vega as derivative
2. Bisection       — robust fallback, guaranteed to converge on [lo, hi]

Both methods use Jaeckel (2015) style initial guesses for speed.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal

from .model import BlackScholesModel, BSMInputs


# ---------------------------------------------------------------------------
# Initial guess heuristics
# ---------------------------------------------------------------------------

def _brenner_subrahmanyam_guess(price: float, S: float, K: float, T: float) -> float:
    """
    Brenner & Subrahmanyam (1988) ATM approximation:
        σ ≈ price * sqrt(2π / T) / S     (for ATM options)
    """
    if T <= 0:
        return 0.3
    return (price / S) * np.sqrt(2 * np.pi / T)


def _corrado_miller_guess(price: float, S: float, K: float, T: float,
                           r: float, option_type: str) -> float:
    """
    Corrado & Miller (1996) closed-form IV approximation.
    More accurate away from ATM than Brenner-Subrahmanyam.
    """
    df = np.exp(-r * T)
    F  = S * np.exp(r * T)
    if option_type == "call":
        adjusted = price - max(S - K * df, 0)
    else:
        adjusted = price - max(K * df - S, 0)

    inner = (F - K) / 2
    discriminant = adjusted ** 2 - (2 / np.pi) * (F - K) ** 2
    if discriminant < 0:
        discriminant = 0.0
    sigma_guess = (
        np.sqrt(2 * np.pi / T) / (F + K) *
        (adjusted + inner + np.sqrt(adjusted ** 2 + inner ** 2 - discriminant + 1e-12))
    )
    return max(sigma_guess, 1e-4)


# ---------------------------------------------------------------------------
# Core IV solver
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    method: Literal["newton", "bisection"] = "newton",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> dict:
    """
    Compute implied volatility from a market option price.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S, K, T, r : float
        Spot, strike, time-to-expiry (years), risk-free rate.
    option_type : "call" or "put"
    method : "newton" (fast) or "bisection" (robust)
    tol : float
        Convergence tolerance on |model_price - market_price|.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    dict with keys:
        implied_vol    : float or None
        iterations     : int
        converged      : bool
        final_error    : float
        method_used    : str

    Examples
    --------
    >>> from black_scholes.implied_vol import implied_volatility
    >>> result = implied_volatility(10.45, S=100, K=100, T=1.0, r=0.05)
    >>> print(f"IV: {result['implied_vol']*100:.2f}%")
    IV: 19.99%
    """
    if T <= 0:
        return {"implied_vol": None, "iterations": 0, "converged": False,
                "final_error": float("inf"), "method_used": method}

    def model_price(sigma: float) -> float:
        inp = BSMInputs(S=S, K=K, T=T, r=r, sigma=max(sigma, 1e-6))
        res = BlackScholesModel(inp).price()
        return res.call_price if option_type == "call" else res.put_price

    def model_vega(sigma: float) -> float:
        inp = BSMInputs(S=S, K=K, T=T, r=r, sigma=max(sigma, 1e-6))
        return BlackScholesModel(inp).greeks().vega * 100  # vega in price units

    # Lower-bound price check
    df = np.exp(-r * T)
    lb = max(S - K * df, 0) if option_type == "call" else max(K * df - S, 0)
    if market_price <= lb:
        return {"implied_vol": None, "iterations": 0, "converged": False,
                "final_error": abs(market_price - lb), "method_used": method}

    # Initial guess
    sigma = _corrado_miller_guess(market_price, S, K, T, r, option_type)
    sigma = np.clip(sigma, 0.001, 10.0)

    if method == "newton":
        for i in range(1, max_iter + 1):
            p    = model_price(sigma)
            err  = p - market_price
            vega = model_vega(sigma)
            if abs(err) < tol:
                return {"implied_vol": sigma, "iterations": i, "converged": True,
                        "final_error": abs(err), "method_used": "newton"}
            if abs(vega) < 1e-12:
                break   # Vega too small — fall through to bisection
            sigma -= err / vega
            sigma  = np.clip(sigma, 1e-6, 10.0)
        # Newton didn't converge — fall back to bisection
        method = "bisection (fallback)"

    # Bisection
    lo, hi = 1e-6, 10.0
    for i in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        err = model_price(mid) - market_price
        if abs(err) < tol or (hi - lo) / 2 < tol:
            return {"implied_vol": mid, "iterations": i, "converged": True,
                    "final_error": abs(err), "method_used": method}
        if err < 0:
            lo = mid
        else:
            hi = mid

    return {"implied_vol": (lo + hi) / 2, "iterations": max_iter,
            "converged": False, "final_error": abs(model_price((lo + hi) / 2) - market_price),
            "method_used": method}


def iv_surface_row(
    S: float,
    K_list: list[float],
    T: float,
    r: float,
    market_prices: list[float],
    option_type: Literal["call", "put"] = "call",
) -> list[dict]:
    """
    Compute implied volatilities for a list of strikes at a single expiry.
    Useful for plotting the volatility smile / skew.

    Returns a list of dicts: [{strike, market_price, implied_vol, moneyness}, ...]
    """
    results = []
    for K, price in zip(K_list, market_prices):
        iv_result = implied_volatility(price, S=S, K=K, T=T, r=r, option_type=option_type)
        iv = iv_result["implied_vol"]
        results.append({
            "strike":       K,
            "market_price": price,
            "implied_vol":  round(iv * 100, 4) if iv else None,  # in %
            "moneyness":    round(S / K, 4),
            "log_moneyness": round(np.log(S / K), 4),
            "converged":    iv_result["converged"],
        })
    return results
