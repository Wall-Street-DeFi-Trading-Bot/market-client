"""
Paper trading example for cross-exchange arbitrage.

This script:
- Finds arbitrage opportunities using ArbitrageDetector
- Executes them on a simulated paper account (no real orders)
- Uses run_arbitrage_bot(config=..., mode=PAPER)
"""

import asyncio
import signal

from market_data_client.arbitrage.bot import run_arbitrage_bot
from market_data_client.arbitrage.config import BotConfig, ExecutionMode


async def main() -> None:
    # Symbols to monitor
    symbols = ["BNBUSDT"]

    # (exchange, instrument) pairs
    exchanges = [
        ("Binance", "spot"),
        ("Binance", "perpetual"),
        ("PancakeSwapV2", "swap"),
        ("PancakeSwapV3", "swap"),
    ]

    # Paper trading: starting balances per (exchange, instrument)
    paper_initial_balances = {
        ("Binance", "spot"): {
            "USDT": 10_000.0,
            "BNB": 0.0,
        },
        ("Binance", "perpetual"): {
            "USDT": 10_000.0,
            "BNB": 0.0,
        },
        ("PancakeSwapV2", "swap"): {
            "USDT": 10_000.0,
            "BNB": 10,
        },
        ("PancakeSwapV3", "swap"): {
            "USDT": 10_000.0,
            "BNB": 10,
        },
    }

    config = BotConfig(
        mode=ExecutionMode.PAPER,
        nats_url="nats://127.0.0.1:4222",
        symbols=symbols,
        exchanges=exchanges,
        min_profit_pct=0.1,        # execute only if net profit >= 0.1%
        trade_notional_usd=100.0,  # trade size per opportunity (in USDT)
        scan_interval=5.0,         # seconds
        paper_initial_balances=paper_initial_balances,
        # plus any risk params you defined in BotConfig (max_daily_loss, etc.)
    )
    
    # CEX symbol -> DEX pair mapping
    symbol_mapping = {
        "BNBUSDT": {
            "PancakeSwapV2": "USDTWBNB",
            "PancakeSwapV3": "USDTWBNB",
        },
    }
    
    stop_event = asyncio.Event()

    def _handle_sigint(signum, frame):
        """
        Handle Ctrl+C (SIGINT).

        First Ctrl+C:
            - Set stop_event so the bot can exit its loop cleanly.
        Second Ctrl+C:
            - Raise KeyboardInterrupt to force exit.
        """
        if not stop_event.is_set():
            print("\n\n✅ Stopping paper arbitrage bot gracefully...")
            stop_event.set()
        else:
            print("\n\n⛔ Force exit.")
            raise KeyboardInterrupt()

    # Register SIGINT handler (Ctrl+C)
    signal.signal(signal.SIGINT, _handle_sigint)

    await run_arbitrage_bot(
        config=config,
        symbol_mapping=symbol_mapping,
        stop_event=stop_event,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped paper arbitrage bot")
