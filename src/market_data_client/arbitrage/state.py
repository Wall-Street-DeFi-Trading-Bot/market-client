# src/market_data_client/arbitrage/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exchange import TradeResult


PriceResolver = Callable[[str], float]
"""
Return the price of 1 unit of `asset` denominated in USDT.
Example:
    resolver("USDT") -> 1.0
    resolver("BNB")  -> 700.0
If the asset is unknown, return 0.0.
"""


@dataclass
class AccountState:
    """
    Per (exchange, instrument) account state.

    Stores simple asset balances like:
        balances = {"USDT": 1000.0, "BNB": 2.5}
    """
    exchange: str
    instrument: str
    balances: Dict[str, float] = field(default_factory=dict)

    def deposit(self, asset: str, amount: float) -> None:
        """Increase balance for the given asset."""
        self.balances[asset] = float(self.balances.get(asset, 0.0)) + float(amount)

    def withdraw(self, asset: str, amount: float) -> None:
        """Decrease balance, raising if insufficient funds."""
        current = float(self.balances.get(asset, 0.0))
        amount_f = float(amount)
        if amount_f > current:
            raise ValueError(
                f"Insufficient balance for {asset}: have {current}, need {amount_f}"
            )
        self.balances[asset] = current - amount_f


@dataclass
class ArbitrageExecutionRecord:
    """
    Optional higher-level record to group multiple TradeResult objects
    belonging to one arbitrage execution.

    You can use this from TradeExecutor / bot layer if you want:
      - record expected vs realized (demo) price
      - attach opportunity metadata
    """
    executed_at: datetime
    symbol: str
    trades: List[TradeResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BotState:
    """
    Global bot state used by:
      - Exchange clients (balances)
      - TradeExecutor (trade history)
      - Bot summary (P&L, stats)

    Notes:
      - Balances are the source of truth for paper/demo accounting.
      - `initial_snapshot` is captured once after initial deposits.
    """
    accounts: Dict[Tuple[str, str], AccountState] = field(default_factory=dict)
    executed_trades: List[TradeResult] = field(default_factory=list)
    executions: List[ArbitrageExecutionRecord] = field(default_factory=list)

    started_at: datetime = field(default_factory=datetime.utcnow)

    initial_snapshot: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    initial_captured_at: Optional[datetime] = None

    def get_or_create_account(self, exchange: str, instrument: str) -> AccountState:
        """Return existing account for (exchange, instrument) or create a new one."""
        key = (exchange, instrument)
        if key not in self.accounts:
            self.accounts[key] = AccountState(exchange=exchange, instrument=instrument)
        return self.accounts[key]

    def record_trade(self, result: TradeResult) -> None:
        """Append executed trade to in-memory history."""
        self.executed_trades.append(result)

    def record_execution(
        self,
        symbol: str,
        trades: List[TradeResult],
        metadata: Optional[Dict[str, Any]] = None,
        executed_at: Optional[datetime] = None,
    ) -> None:
        """Record one arbitrage execution consisting of one or more trades."""
        self.executions.append(
            ArbitrageExecutionRecord(
                executed_at=executed_at or datetime.utcnow(),
                symbol=symbol,
                trades=list(trades),
                metadata=dict(metadata or {}),
            )
        )

    def snapshot_balances(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Convenience helper for logging / debugging."""
        return {key: dict(account.balances) for key, account in self.accounts.items()}

    def capture_initial_snapshot(self) -> None:
        """
        Capture initial balances for all accounts.

        Call this once after you deposit initial balances (paper/demo).
        """
        self.initial_snapshot = self.snapshot_balances()
        self.initial_captured_at = datetime.utcnow()

    def account_delta(
        self,
        key: Tuple[str, str],
        *,
        include_zero_assets: bool = True,
    ) -> Dict[str, float]:
        """
        Compute per-asset balance delta for a given account key.
        Delta = current - initial.
        """
        current = self.accounts.get(key).balances if key in self.accounts else {}
        initial = self.initial_snapshot.get(key, {})

        assets = set(initial.keys()) | set(current.keys())
        out: Dict[str, float] = {}
        for a in assets:
            d = float(current.get(a, 0.0)) - float(initial.get(a, 0.0))
            if include_zero_assets or abs(d) > 0:
                out[a] = d
        return out

    def total_asset_balance(self, asset: str) -> float:
        """Sum `asset` balance across all accounts."""
        total = 0.0
        for account in self.accounts.values():
            total += float(account.balances.get(asset, 0.0))
        return total

    def compute_total_equity_usdt(
        self,
        price_resolver: Optional[PriceResolver] = None,
    ) -> float:
        """
        Compute total equity across all accounts in USDT terms.

        If no resolver is provided, only USDT balances are counted.
        """
        resolver = price_resolver or (lambda a: 1.0 if a.upper() == "USDT" else 0.0)

        total = 0.0
        for account in self.accounts.values():
            for asset, amt in account.balances.items():
                px = float(resolver(asset))
                if px <= 0:
                    continue
                total += float(amt) * px
        return total

    def compute_initial_equity_usdt(
        self,
        price_resolver: Optional[PriceResolver] = None,
    ) -> float:
        """
        Compute initial equity in USDT terms based on captured snapshot.

        If initial snapshot was not captured, returns 0.0.
        """
        if not self.initial_snapshot:
            return 0.0

        resolver = price_resolver or (lambda a: 1.0 if a.upper() == "USDT" else 0.0)

        total = 0.0
        for _, bal in self.initial_snapshot.items():
            for asset, amt in bal.items():
                px = float(resolver(asset))
                if px <= 0:
                    continue
                total += float(amt) * px
        return total

    def compute_roi_pct(
        self,
        price_resolver: Optional[PriceResolver] = None,
    ) -> float:
        """
        ROI% = (final_equity - initial_equity) / initial_equity * 100
        """
        initial = self.compute_initial_equity_usdt(price_resolver=price_resolver)
        if initial <= 0:
            return 0.0
        final = self.compute_total_equity_usdt(price_resolver=price_resolver)
        return (final - initial) / initial * 100.0

    def estimate_fee_amount_usdt(self, trade: TradeResult) -> float:
        """
        Best-effort fee amount estimate in USDT terms for reporting.

        Priority:
          1) metadata contains explicit quote deltas (before/after fee)
          2) fallback to abs(quantity * price) * fee_rate
        """
        meta = trade.metadata or {}

        for k in ("binance_demo", "pancake_demo"):
            if k in meta and isinstance(meta[k], dict):
                m = meta[k]
                q_before = m.get("quote_delta_before_fee")
                q_after = m.get("quote_delta_after_fee")
                if isinstance(q_before, (int, float)) and isinstance(q_after, (int, float)):
                    return float(abs(float(q_before) - float(q_after)))

        try:
            fee_rate = float(trade.fee)
            notion = abs(float(trade.quantity) * float(trade.price))
            if fee_rate <= 0 or notion <= 0:
                return 0.0
            return notion * fee_rate
        except Exception:
            return 0.0

    def total_estimated_fees_usdt(self) -> float:
        """Sum estimated fees for all recorded trades."""
        total = 0.0
        for t in self.executed_trades:
            total += self.estimate_fee_amount_usdt(t)
        return total
