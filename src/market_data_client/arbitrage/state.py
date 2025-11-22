# src/market_data_client/arbitrage/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

from .exchange import TradeResult


@dataclass
class AccountState:
    """
    Per (exchange, instrument) account state for paper trading.

    Stores simple asset balances like:
        balances = { "USDT": 1000.0, "BNB": 2.5 }
    """
    exchange: str
    instrument: str
    balances: Dict[str, float] = field(default_factory=dict)

    def deposit(self, asset: str, amount: float) -> None:
        """Increase balance for the given asset."""
        self.balances[asset] = self.balances.get(asset, 0.0) + amount

    def withdraw(self, asset: str, amount: float) -> None:
        """Decrease balance, raising if insufficient funds."""
        current = self.balances.get(asset, 0.0)
        if amount > current:
            raise ValueError(
                f"Insufficient balance for {asset}: have {current}, need {amount}"
            )
        self.balances[asset] = current - amount


@dataclass
class BotState:
    """
    Global bot state used by:
      - PaperExchangeClient (balances)
      - TradeExecutor (trade history)
      - Bot summary (P&L, stats)
    """
    accounts: Dict[Tuple[str, str], AccountState] = field(default_factory=dict)
    executed_trades: List[TradeResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)

    def get_or_create_account(self, exchange: str, instrument: str) -> AccountState:
        """Return existing account for (exchange, instrument) or create a new one."""
        key = (exchange, instrument)
        if key not in self.accounts:
            self.accounts[key] = AccountState(exchange=exchange, instrument=instrument)
        return self.accounts[key]

    def record_trade(self, result: TradeResult) -> None:
        """Append executed trade to in-memory history."""
        self.executed_trades.append(result)

    def snapshot_balances(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Convenience helper for logging / debugging."""
        return {
            key: dict(account.balances)
            for key, account in self.accounts.items()
        }
