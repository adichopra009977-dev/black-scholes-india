"""
Black-Scholes Options Pricing Model — Indian Markets Edition
=============================================================
Adapted for NSE/BSE Indian index options:
  - Nifty 50    (NSE) — lot size 75, settled in INR
  - Sensex      (BSE) — lot size 20, settled in INR
  - Bank Nifty  (NSE) — lot size 30, settled in INR
  - Midcap Nifty (NSE) — lot size 75
  - Fin Nifty   (NSE) — lot size 40

Key Indian market parameters:
  - Risk-free rate: RBI repo rate (~6.5% as of 2025)
  - Expiry convention: weekly (Thursday) + monthly (last Thursday)
  - Settlement: cash-settled, European-style
  - Lot sizes enforced for contract-level P&L

Formula:
    Call: C = S·N(d1) - K·e^{-rT}·N(d2)
    Put:  P = K·e^{-rT}·N(-d2) - S·N(-d1)

Where:
    S  = Current index level (e.g., Nifty ~24,000)
    K  = Strike price (in index points)
    r  = RBI repo rate (annualised, e.g. 0.065)
    sigma = India VIX-calibrated volatility (annualised)
    T  = Time to expiry in years
    N  = Cumulative standard normal CDF
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Indian market constants
# ---------------------------------------------------------------------------

INDICES = {
    "NIFTY":      {"lot_size": 75,  "exchange": "NSE", "tick": 0.05},
    "BANKNIFTY":  {"lot_size": 30,  "exchange": "NSE", "tick": 0.05},
    "SENSEX":     {"lot_size": 20,  "exchange": "BSE", "tick": 0.05},
    "MIDCPNIFTY": {"lot_size": 75,  "exchange": "NSE", "tick": 0.05},
    "FINNIFTY":   {"lot_size": 40,  "exchange": "NSE", "tick": 0.05},
}

RBI_REPO_RATE        = 0.065   # RBI repo rate 2025
DEFAULT_INDIA_VIX    = 0.14    # India VIX ~14% (calm market baseline)
STT_RATE_SELL        = 0.000625
EXCHANGE_CHARGE_RATE = 0.00053


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BSMInputs:
    """
    Container for Black-Scholes model inputs — Indian index options edition.

    Parameters
    ----------
    S     : float — Current index level (e.g., 24500 for Nifty)
    K     : float — Strike price (index points)
    T     : float — Time to expiry in years (e.g., 7/365 for weekly)
    r     : float — Risk-free rate (RBI repo, e.g. 0.065)
    sigma : float — Annualised volatility (India VIX / 100, e.g. 0.14)
    index : str   — Index name: 'NIFTY', 'BANKNIFTY', 'SENSEX', etc.
    """
    S:     float
    K:     float
    T:     float
    r:     float
    sigma: float
    index: str = "NIFTY"

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Index level S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiry T must be non-negative, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
        if self.index.upper() not in INDICES:
            raise ValueError(f"Unknown index '{self.index}'. Choose from {list(INDICES)}")
        self.index = self.index.upper()

    @property
    def lot_size(self) -> int:
        return INDICES[self.index]["lot_size"]

    @property
    def expiry_days(self) -> float:
        return self.T * 365


@dataclass
class BSMResult:
    call_price:      float   # per unit (index point)
    put_price:       float
    call_premium:    float   # call_price x lot_size (INR per contract)
    put_premium:     float
    d1:              float
    d2:              float
    N_d1:            float
    N_d2:            float
    intrinsic_call:  float
    intrinsic_put:   float
    time_value_call: float
    time_value_put:  float
    moneyness:       str
    lot_size:        int
    index:           str


@dataclass
class Greeks:
    delta_call:  float
    delta_put:   float
    gamma:       float
    vega:        float        # per 1% sigma move, per unit
    theta_call:  float        # per calendar day, per unit
    theta_put:   float
    rho_call:    float        # per 1% rate move
    rho_put:     float
    vanna:       float
    volga:       float
    # Contract-level (multiplied by lot_size)
    delta_call_contract: float
    delta_put_contract:  float
    vega_contract:       float
    theta_call_contract: float
    theta_put_contract:  float


@dataclass
class IndianCosts:
    stt:             float
    exchange_charge: float
    gst:             float
    sebi_fee:        float
    stamp_duty:      float
    total:           float


# ---------------------------------------------------------------------------
# Black-Scholes Model class
# ---------------------------------------------------------------------------

class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing — Indian Index Options edition.

    Example
    -------
    >>> params = BSMInputs(S=24500, K=24500, T=7/365, r=0.065, sigma=0.14, index='NIFTY')
    >>> bsm = BlackScholesModel(params)
    >>> r = bsm.price()
    >>> print(f"Call: INR {r.call_price:.2f}  Contract: INR {r.call_premium:,.0f}")
    """

    def __init__(self, inputs: BSMInputs):
        self.inputs = inputs
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self):
        p = self.inputs
        if p.T == 0:
            return 0.0, 0.0
        d1 = (np.log(p.S / p.K) + (p.r + 0.5 * p.sigma ** 2) * p.T) / (
            p.sigma * np.sqrt(p.T))
        d2 = d1 - p.sigma * np.sqrt(p.T)
        return float(d1), float(d2)

    def _discount_factor(self):
        return np.exp(-self.inputs.r * self.inputs.T)

    def _moneyness_label(self):
        ratio = self.inputs.S / self.inputs.K
        if abs(ratio - 1.0) < 0.005:
            return "ATM"
        return "ITM" if ratio > 1.0 else "OTM"

    def price(self) -> BSMResult:
        p   = self.inputs
        df  = self._discount_factor()
        d1, d2 = self._d1, self._d2
        lot = p.lot_size

        if p.T == 0:
            call = max(p.S - p.K, 0.0)
            put  = max(p.K - p.S, 0.0)
        else:
            call = p.S * norm.cdf(d1) - p.K * df * norm.cdf(d2)
            put  = p.K * df * norm.cdf(-d2) - p.S * norm.cdf(-d1)

        intr_c = max(p.S - p.K, 0.0)
        intr_p = max(p.K - p.S, 0.0)

        return BSMResult(
            call_price=float(call), put_price=float(put),
            call_premium=float(call) * lot, put_premium=float(put) * lot,
            d1=d1, d2=d2,
            N_d1=float(norm.cdf(d1)), N_d2=float(norm.cdf(d2)),
            intrinsic_call=intr_c, intrinsic_put=intr_p,
            time_value_call=float(call) - intr_c,
            time_value_put=float(put) - intr_p,
            moneyness=self._moneyness_label(),
            lot_size=lot, index=p.index,
        )

    def greeks(self) -> Greeks:
        p   = self.inputs
        df  = self._discount_factor()
        d1, d2 = self._d1, self._d2
        sqT = np.sqrt(p.T) if p.T > 0 else 1e-9
        lot = p.lot_size
        phi_d1 = norm.pdf(d1)

        delta_call = float(norm.cdf(d1))
        delta_put  = float(delta_call - 1.0)
        vega       = float(p.S * phi_d1 * sqT / 100)
        theta_call = float((-p.S * phi_d1 * p.sigma / (2 * sqT)
                            - p.r * p.K * df * norm.cdf(d2)) / 365)
        theta_put  = float((-p.S * phi_d1 * p.sigma / (2 * sqT)
                            + p.r * p.K * df * norm.cdf(-d2)) / 365)
        rho_call   = float(p.K * p.T * df * norm.cdf(d2) / 100)
        rho_put    = float(-p.K * p.T * df * norm.cdf(-d2) / 100)
        gamma      = float(phi_d1 / (p.S * p.sigma * sqT))
        vanna      = float(-phi_d1 * d2 / p.sigma)
        volga      = float(p.S * phi_d1 * sqT * d1 * d2 / p.sigma)

        return Greeks(
            delta_call=delta_call, delta_put=delta_put,
            gamma=gamma, vega=vega,
            theta_call=theta_call, theta_put=theta_put,
            rho_call=rho_call, rho_put=rho_put,
            vanna=vanna, volga=volga,
            delta_call_contract=delta_call * lot,
            delta_put_contract=delta_put * lot,
            vega_contract=vega * lot,
            theta_call_contract=theta_call * lot,
            theta_put_contract=theta_put * lot,
        )

    def transaction_costs(self, option_type: str = "call") -> IndianCosts:
        p   = self.inputs
        res = self.price()
        premium = res.call_price if option_type == "call" else res.put_price
        contract_val = premium * p.lot_size
        stt    = contract_val * STT_RATE_SELL
        exch   = contract_val * EXCHANGE_CHARGE_RATE
        gst    = exch * 0.18
        sebi   = contract_val * 0.000001
        stamp  = contract_val * 0.00003
        return IndianCosts(
            stt=round(stt, 2), exchange_charge=round(exch, 2),
            gst=round(gst, 4), sebi_fee=round(sebi, 4),
            stamp_duty=round(stamp, 2),
            total=round(stt + exch + gst + sebi + stamp, 2),
        )

    def put_call_parity(self) -> dict:
        result = self.price()
        p  = self.inputs
        lhs = result.call_price - result.put_price
        rhs = p.S - p.K * self._discount_factor()
        return {
            "lhs (C - P)":      round(lhs, 8),
            "rhs (S - Ke^-rT)": round(rhs, 8),
            "difference":       round(abs(lhs - rhs), 10),
            "parity_holds":     abs(lhs - rhs) < 1e-6,
        }

    def implied_forward(self) -> float:
        return self.inputs.S * np.exp(self.inputs.r * self.inputs.T)

    def summary(self) -> str:
        r    = self.price()
        g    = self.greeks()
        pcp  = self.put_call_parity()
        p    = self.inputs
        c    = self.transaction_costs("call")
        info = INDICES[p.index]
        cur  = "INR"

        lines = [
            "=" * 65,
            f"  BSM MODEL — INDIAN INDEX OPTIONS  [{p.index} | {info['exchange']}]",
            "=" * 65,
            f"  Index Level (S) : {cur} {p.S:>12,.2f}",
            f"  Strike (K)      : {cur} {p.K:>12,.2f}",
            f"  Expiry          : {p.T*365:.1f} calendar days ({p.T:.4f}y)",
            f"  RBI Repo Rate   : {p.r*100:.2f}%   India VIX proxy: {p.sigma*100:.2f}%",
            f"  Lot Size        : {p.lot_size}         Moneyness: {r.moneyness}",
            f"  Forward Level   : {cur} {self.implied_forward():>12,.2f}",
            "-" * 65,
            f"  {'':32s}    CALL          PUT",
            f"  {'Price (per unit)':32s}   {cur} {r.call_price:>8.2f}   {cur} {r.put_price:>8.2f}",
            f"  {'Contract Premium (lot x price)':32s}   {cur} {r.call_premium:>8,.0f}   {cur} {r.put_premium:>8,.0f}",
            f"  {'Intrinsic Value':32s}   {cur} {r.intrinsic_call:>8.2f}   {cur} {r.intrinsic_put:>8.2f}",
            f"  {'Time Value':32s}   {cur} {r.time_value_call:>8.2f}   {cur} {r.time_value_put:>8.2f}",
            f"  {'d1 / d2':32s}    {r.d1:>9.4f}     {r.d2:>9.4f}",
            "-" * 65,
            f"  GREEKS (per unit  /  per contract)       CALL                PUT",
            f"  {'Delta':32s}   {g.delta_call:>6.4f} / {g.delta_call_contract:>6.1f}    {g.delta_put:>6.4f} / {g.delta_put_contract:>6.1f}",
            f"  {'Gamma (shared)':32s}   {g.gamma:.8f}",
            f"  {'Vega per 1% sigma':32s}   {g.vega:>6.2f} / {cur} {g.vega_contract:>8,.0f}",
            f"  {'Theta per day':32s}   {g.theta_call:>6.2f} / {cur} {g.theta_call_contract:>8,.0f}    {g.theta_put:>6.2f} / {cur} {g.theta_put_contract:>8,.0f}",
            f"  {'Rho per 1% rate':32s}   {g.rho_call:>6.2f}          {g.rho_put:>6.2f}",
            "-" * 65,
            f"  TRANSACTION COSTS — buy 1 call contract",
            f"  {'STT (0.0625% on premium)':32s}   {cur} {c.stt:>8.2f}",
            f"  {'Exchange charge (NSE/BSE)':32s}   {cur} {c.exchange_charge:>8.2f}",
            f"  {'GST (18% on exchange charge)':32s}   {cur} {c.gst:>8.4f}",
            f"  {'SEBI fee':32s}   {cur} {c.sebi_fee:>8.4f}",
            f"  {'Stamp duty (0.003%)':32s}   {cur} {c.stamp_duty:>8.2f}",
            f"  {'TOTAL COSTS':32s}   {cur} {c.total:>8.2f}",
            "-" * 65,
            f"  Put-Call Parity: {'HOLDS' if pcp['parity_holds'] else 'VIOLATED'}  (diff={pcp['difference']:.2e})",
            "=" * 65,
        ]
        return "\n".join(lines)
