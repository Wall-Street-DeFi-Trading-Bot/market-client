# src/market_data_client/arbitrage/pairs.py

from __future__ import annotations

from typing import Dict, Tuple

# Canonical base/quote mapping for symbols used by the bot.
# Use wrapped tokens for on-chain routing where applicable (BNB -> WBNB, BTC -> BTCB).
PAIR_BASE_QUOTE: Dict[str, Tuple[str, str]] = {
    "BNBUSDT": ("WBNB", "USDT"),
    "USDTWBNB": ("USDT", "WBNB"),
    "BTCBUSDT": ("BTCB", "USDT"),
    "CAKEUSDT": ("CAKE", "USDT"),
    "CAKEWBNB": ("CAKE", "WBNB"),
    "TWTWBNB": ("TWT", "WBNB"),
    "TWTUSDT": ("TWT", "USDT"),
    "SFPWBNB": ("SFP", "WBNB"),
    "SFPUSDT": ("SFP", "USDT"),
}

def resolve_base_quote(symbol: str) -> Tuple[str, str]:
    s = symbol.replace("/", "").replace("-", "").upper()
    if s not in PAIR_BASE_QUOTE:
        raise ValueError(f"Unknown base/quote for symbol={symbol}")
    return PAIR_BASE_QUOTE[s]
