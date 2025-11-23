# src/market_data_client/arbitrage/bot.py
"""
High-level arbitrage bot orchestration.

This wires together:
- MarketDataClient (price/fee/slippage feed)
- ArbitrageDetector (finding opportunities)
- ExchangeClient implementations (live / paper)
- RiskManager + TradeExecutor
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..market_data_client import CexConfig, DexConfig, MarketDataClient
from .arbitrage_detector import ArbitrageDetector, ArbitrageOpportunity
from .config import BotConfig, ExecutionMode
from .exchange import (
    BinanceExchangeClient,
    ExchangeClient,
    PaperExchangeClient,
    PancakeSwapExchangeClient,
)
from .executor import TradeExecutor
from .risk import RiskManager
from .state import BotState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)


def _build_market_data_client(config: BotConfig, symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> MarketDataClient:
    """
    Create a MarketDataClient similar to run_arbitrage_detector, but reusable for the bot.
    """
    symbols = config.symbols
    exchanges = config.exchanges
    symbol_mapping = symbol_mapping or {}

    cex_configs: List[CexConfig] = []
    dex_configs: List[DexConfig] = []

    # Group CEX exchanges by name
    cex_by_exchange: Dict[str, Dict[str, set]] = {}
    for exchange, instrument in exchanges:
        if instrument != "swap":
            if exchange not in cex_by_exchange:
                cex_by_exchange[exchange] = {"instruments": set(), "symbols": set(symbols)}
            cex_by_exchange[exchange]["instruments"].add(instrument)

    for exchange, data in cex_by_exchange.items():
        cfg = CexConfig(
            exchange=exchange,
            symbols=list(data["symbols"]),
            instruments=list(data["instruments"]),
            want=("tick", "funding", "fee", "volume"),
        )
        cex_configs.append(cfg)
        logger.info(
            f"[BOT] CEX Config: {exchange} - symbols={cfg.symbols}, instruments={cfg.instruments}"
        )

    # Group DEX exchanges by (exchange, chain)
    dex_by_exchange: Dict[Tuple[str, str], Dict[str, set]] = {}
    for exchange, instrument in exchanges:
        if instrument == "swap":
            key = (exchange, "BSC")  # default chain
            if key not in dex_by_exchange:
                dex_by_exchange[key] = {"pairs": set()}

            for symbol in symbols:
                if symbol in symbol_mapping and exchange in symbol_mapping[symbol]:
                    dex_by_exchange[key]["pairs"].add(symbol_mapping[symbol][exchange])
                else:
                    dex_by_exchange[key]["pairs"].add(symbol)

    for (exchange, chain), data in dex_by_exchange.items():
        cfg = DexConfig(
            exchange=exchange,
            chain=chain,
            pairs=list(data["pairs"]),
            want=("tick", "slippage", "fee", "volume"),
        )
        dex_configs.append(cfg)
        logger.info(
            f"[BOT] DEX Config: {exchange}({chain}) - pairs={cfg.pairs}"
        )

    client = MarketDataClient(
        nats_url=config.nats_url,
        use_jetstream=False,
        cex=cex_configs or None,
        dex=dex_configs or None,
        enable_csv=True,
    )

    return client


def _build_exchange_clients(config: BotConfig, state: BotState) -> Dict[Tuple[str, str], ExchangeClient]:
    """
    Build ExchangeClient instances for all (exchange, instrument) pairs
    defined in the bot configuration.
    """
    clients: Dict[Tuple[str, str], ExchangeClient] = {}

    # Initialize paper balances if configured
    for (ex, inst), balances in config.paper_initial_balances.items():
        account = state.get_or_create_account(ex, inst)
        for asset, amount in balances.items():
            account.deposit(asset, amount)

    for ex, inst in config.exchanges:
        key = (ex, inst)

        if config.mode == ExecutionMode.PAPER:
            client = PaperExchangeClient(name=ex, instrument=inst, state=state)
            clients[key] = client
            logger.info(f"[BOT] Using PAPER client for {ex}({inst})")
        else:
            # LIVE mode
            if ex == "Binance":
                client = BinanceExchangeClient(instrument=inst, mode=ExecutionMode.LIVE)
                clients[key] = client
                logger.info(f"[BOT] Using LIVE Binance client for {inst}")
            elif ex in ("PancakeSwapV2", "PancakeSwapV3"):
                client = PancakeSwapExchangeClient(
                    exchange_name=ex,
                    instrument=inst,
                    mode=ExecutionMode.LIVE,
                )
                clients[key] = client
                logger.info(f"[BOT] Using LIVE {ex} client")
            else:
                raise ValueError(f"Unknown exchange for LIVE mode: {ex}")

    return clients

def _total_usdt_equity(state: BotState) -> float:
    """
    Sum all USDT balances across all paper accounts.

    Assumes arbitrage keeps you roughly flat in base asset,
    so total equity can be approximated by total USDT.
    """
    total = 0.0
    for (_, _), account in state.accounts.items():
        # account.balances is assumed to be { "USDT": float, "BNB": float, ... }
        usdt = account.balances.get("USDT", 0.0)
        try:
            total += float(usdt)
        except (TypeError, ValueError):
            continue
    return total


async def run_arbitrage_bot(
    config: BotConfig,
    symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Main entry point to run the arbitrage bot (PAPER or LIVE).

    Steps:
      1. Build MarketDataClient and start it.
      2. Create ArbitrageDetector.
      3. Build ExchangeClients (paper or live).
      4. Build RiskManager and TradeExecutor.
      5. In a loop:
           - scan opportunities
           - execute selected ones
           - track paper PnL (in PAPER mode)
    """
    logger.info("=" * 80)
    logger.info("Starting Arbitrage Bot")
    logger.info("=" * 80)
    logger.info(f"  Mode: {config.mode.value}")
    logger.info(f"  Symbols: {', '.join(config.symbols)}")
    logger.info(
        f"  Exchanges: {', '.join([f'{ex}({inst})' for ex, inst in config.exchanges])}"
    )
    logger.info(f"  Min profit: {config.min_profit_pct}%")
    logger.info(f"  Trade notional: {config.trade_notional_usd} USDT")
    logger.info(f"  Scan interval: {config.scan_interval}s")
    logger.info("=" * 80)

    # 1) Market data client
    client = _build_market_data_client(config, symbol_mapping)
    await client.start()
    logger.info("[BOT] MarketDataClient started")

    # 2) Detector
    detector = ArbitrageDetector(
        client=client,
        min_profit_pct=config.min_profit_pct,
        symbols=config.symbols,
        exchanges=config.exchanges,
        symbol_mapping=symbol_mapping or {},
    )

    # 3) State + exchange clients
    state = BotState()
    exchange_clients = _build_exchange_clients(config, state)

    initial_equity_usdt: float = 0.0
    total_realized_pnl_usdt: float = 0.0
    total_trades_executed: int = 0

    if config.mode == ExecutionMode.PAPER:
        initial_equity_usdt = _total_usdt_equity(state)
        logger.info(
            f"[BOT] Initial PAPER equity (USDT-only approximation): "
            f"{initial_equity_usdt:.4f} USDT"
        )

    # 4) Risk + executor
    risk = RiskManager(config=config)
    executor = TradeExecutor(exchange_clients=exchange_clients, risk_manager=risk)

    try:
        scan_count = 0
        while True:

            if stop_event is not None and stop_event.is_set():
                logger.info("[BOT] Stop event set, breaking scan loop.")
                break

            scan_count += 1
            logger.info("\n" + "=" * 80)
            logger.info(
                f"[BOT] Scan #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.info("=" * 80)

            opportunities: List[ArbitrageOpportunity] = await detector.scan_opportunities()

            if not opportunities:
                logger.info("[BOT] No profitable opportunities this round.")
            else:
                # Sort by net profit descending
                opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)

                for opp in opportunities:
                    equity_before = None
                    if config.mode == ExecutionMode.PAPER:
                        equity_before = _total_usdt_equity(state)

                    try:
                        await executor.execute_opportunity(opp)
                    except ValueError as e:
                        logger.info("[BOT] Execution failed for %s: %s", opp.symbol, e)
                        continue
                    except Exception as exc:
                        logger.warning(f"[BOT] Failed to execute opportunity: {exc}")
                        continue

                    if config.mode == ExecutionMode.PAPER:
                        equity_after = _total_usdt_equity(state)
                        pnl = equity_after - (equity_before or 0.0)
                        pnl_pct = 0.0
                        if equity_before and equity_before > 0:
                            pnl_pct = pnl / equity_before * 100.0

                        total_realized_pnl_usdt += pnl
                        total_trades_executed += 1

                        logger.info(
                            "[BOT] PAPER trade executed:\n"
                            f"  Symbol        : {opp.symbol}\n"
                            f"  Path          : BUY {opp.buy_exchange}({opp.buy_instrument}) "
                            f"-> SELL {opp.sell_exchange}({opp.sell_instrument})\n"
                            f"  Theoretical   : {opp.net_profit_pct:.3f}% "
                            f"on {config.trade_notional_usd:.2f} USDT "
                            f"(≈ {config.trade_notional_usd * opp.net_profit_pct / 100.0:.4f} USDT)\n"
                            f"  Equity (USDT) : {equity_before:.4f} -> {equity_after:.4f} "
                            f"(Δ={pnl:.4f} USDT, {pnl_pct:.3f}%)"
                        )

            logger.info(f"[BOT] Sleeping {config.scan_interval}s before next scan...")
            await asyncio.sleep(config.scan_interval)
    
    finally:
        await client.stop()
        logger.info("[BOT] MarketDataClient stopped")

        if config.mode == ExecutionMode.PAPER:
            logger.info("\n" + "=" * 80)
            logger.info("[BOT] PAPER trading summary")
            logger.info("=" * 80)
            logger.info(f"Total paper trades executed: {total_trades_executed}")

            # Per-account balance comparison: initial vs final
            for key, account in state.accounts.items():
                ex, inst = key
                initial_bal = config.paper_initial_balances.get(key, {})  # 초기 잔고
                current_bal = account.balances                             # 최종 잔고

                logger.info(f"\nAccount {ex}({inst}) balance changes:")
                logger.info(f"{'Asset':<8} {'Initial':>14} {'Final':>14} {'Δ':>14}")
                logger.info("-" * 54)

                asset_set = set(initial_bal.keys()) | set(current_bal.keys())
                for asset in sorted(asset_set):
                    init = float(initial_bal.get(asset, 0.0) or 0.0)
                    final = float(current_bal.get(asset, 0.0) or 0.0)
                    delta = final - init
                    logger.info(
                        f"{asset:<8} {init:>14.6f} {final:>14.6f} {delta:>14.6f}"
                    )

            final_equity_usdt = _total_usdt_equity(state)
            logger.info("\n--- Equity summary (USDT only) ---")
            logger.info(f"Initial USDT equity : {initial_equity_usdt:.4f} USDT")
            logger.info(f"Final   USDT equity : {final_equity_usdt:.4f} USDT")
            logger.info(
                f"Realized PnL (USDT) : {total_realized_pnl_usdt:.4f} USDT "
                f"(by equity delta)"
            )

            if initial_equity_usdt > 0.0:
                total_roi_pct = (
                    (final_equity_usdt - initial_equity_usdt)
                    / initial_equity_usdt
                    * 100.0
                )
                logger.info(f"Total ROI           : {total_roi_pct:.3f}%")
            logger.info("=" * 80)

