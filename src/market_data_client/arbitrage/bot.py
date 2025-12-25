# src/market_data_client/arbitrage/bot.py

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..market_data_client import CexConfig, DexConfig, MarketDataClient
from .arbitrage_detector import ArbitrageDetector, ArbitrageOpportunity
from .config import BotConfig, ExecutionMode
from .exchange import (
    BinanceExchangeClient,
    BinanceDemoExchangeClient,
    BinanceDemoParams,
    ExchangeClient,
    PaperExchangeClient,
    PancakeSwapExchangeClient,
    PancakeSwapDemoExchangeClient,
    PancakeDemoParams,
)
from .executor import TradeExecutor
from .risk import RiskManager
from .state import BotState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


def _log_demo_trade(trade: Any) -> None:
    """
    Log DEMO trade metadata in a compact, human-friendly way.
    """
    meta = getattr(trade, "metadata", None) or {}

    ex = getattr(trade, "exchange", "?")
    inst = getattr(trade, "instrument", "?")
    sym = getattr(trade, "symbol", "?")
    side = getattr(trade, "side", "?")
    qty = getattr(trade, "quantity", None)
    px = getattr(trade, "price", None)

    logger.info(
        "trade executed exchange=%s instrument=%s symbol=%s side=%s qty=%s price=%s",
        ex,
        inst,
        sym,
        getattr(side, "value", side),
        f"{qty:.8f}" if isinstance(qty, (int, float)) else str(qty),
        f"{px:.8f}" if isinstance(px, (int, float)) else str(px),
    )

    if "binance_demo" in meta:
        b = meta["binance_demo"]
        logger.info(
            "binance_demo theoretical_price=%s execution_price=%s slippage_bps=%s fee_rate=%s",
            b.get("theoretical_price"),
            b.get("execution_price"),
            b.get("slippage_bps"),
            b.get("fee_rate"),
        )

    if "binance_testnet" in meta:
        t = meta["binance_testnet"]
        logger.info(
            "binance_testnet status=%s endpoint=%s",
            t.get("status"),
            t.get("endpoint"),
        )

    if "binance_testnet_error" in meta:
        logger.info("binance_testnet_error %s", meta["binance_testnet_error"])

    if "pancake_demo" in meta:
        p = meta["pancake_demo"]
        avg_fill = p.get("avg_fill_price")
        avg_ret = p.get("avg_return_vs_hint")
        logger.info("pancake_demo avg_fill_price=%s avg_return_vs_hint=%s", avg_fill, avg_ret)

        per_block = p.get("per_block_results") or []
        for r in per_block:
            logger.info(
                "pancake_block fork_block=%s status=%s fill_price=%s return_vs_hint=%s tx_hash=%s",
                r.get("fork_block"),
                r.get("status"),
                r.get("fill_price"),
                r.get("return_vs_hint"),
                r.get("tx_hash"),
            )


def _build_market_data_client(
    config: BotConfig, symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None
) -> MarketDataClient:
    symbols = config.symbols
    exchanges = config.exchanges
    symbol_mapping = symbol_mapping or {}

    cex_configs: List[CexConfig] = []
    dex_configs: List[DexConfig] = []

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

    dex_by_exchange: Dict[Tuple[str, str], Dict[str, set]] = {}
    for exchange, instrument in exchanges:
        if instrument == "swap":
            key = (exchange, "BSC")
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

    return MarketDataClient(
        nats_url=config.nats_url,
        use_jetstream=False,
        cex=cex_configs or None,
        dex=dex_configs or None,
        enable_csv=config.enable_csv,
    )


def _build_exchange_clients(
    config: BotConfig,
    state: BotState,
    demo_binance: Optional[Dict[Tuple[str, str], BinanceDemoParams]] = None,
    demo_pancake: Optional[Dict[Tuple[str, str], PancakeDemoParams]] = None,
) -> Dict[Tuple[str, str], ExchangeClient]:
    clients: Dict[Tuple[str, str], ExchangeClient] = {}

    if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
        for (ex, inst), balances in config.paper_initial_balances.items():
            account = state.get_or_create_account(ex, inst)
            for asset, amount in balances.items():
                account.deposit(asset, amount)

    for ex, inst in config.exchanges:
        key = (ex, inst)

        if config.mode == ExecutionMode.PAPER:
            clients[key] = PaperExchangeClient(name=ex, instrument=inst, state=state)

        elif config.mode == ExecutionMode.LIVE:
            if ex == "Binance":
                clients[key] = BinanceExchangeClient(name=ex, instrument=inst, state=state)
            elif ex in ("PancakeSwapV2", "PancakeSwapV3"):
                clients[key] = PancakeSwapExchangeClient(name=ex, instrument=inst, state=state)
            else:
                raise ValueError(f"LIVE mode does not support exchange {ex}")

        elif config.mode == ExecutionMode.DEMO:
            if ex == "Binance":
                if demo_binance is None or key not in demo_binance:
                    raise ValueError(f"DEMO mode: missing BinanceDemoParams for {ex}({inst})")
                clients[key] = BinanceDemoExchangeClient(
                    name=ex,
                    instrument=inst,
                    state=state,
                    params=demo_binance[key],
                )

            elif ex in ("PancakeSwapV2", "PancakeSwapV3"):
                if demo_pancake is None or key not in demo_pancake:
                    raise ValueError(f"DEMO mode: missing PancakeDemoParams for {ex}({inst})")
                clients[key] = PancakeSwapDemoExchangeClient(
                    exchange_name=ex,
                    instrument=inst,
                    state=state,
                    params=demo_pancake[key],
                )
            else:
                clients[key] = PaperExchangeClient(name=ex, instrument=inst, state=state, fee_rate=0.0005)

        else:
            raise ValueError(f"Unsupported execution mode: {config.mode}")

    return clients


def _total_usdt_equity(state: BotState) -> float:
    total = 0.0
    for account in state.accounts.values():
        total += float(account.balances.get("USDT", 0.0) or 0.0)
    return total

def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(r: List[str]) -> str:
        return " | ".join(r[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def _balances_table(state: BotState, config: BotConfig) -> str:
    rows: List[List[str]] = []

    for (ex, inst), account in sorted(state.accounts.items(), key=lambda x: (x[0][0], x[0][1])):
        initial_bal = config.paper_initial_balances.get((ex, inst), {})
        current_bal = account.balances

        assets = sorted(set(initial_bal.keys()) | set(current_bal.keys()))
        for asset in assets:
            init = float(initial_bal.get(asset, 0.0) or 0.0)
            final = float(current_bal.get(asset, 0.0) or 0.0)
            delta = final - init

            rows.append([
                ex,
                inst,
                asset,
                f"{init:.8f}",
                f"{final:.8f}",
                f"{delta:.8f}",
            ])

    headers = ["exchange", "instrument", "asset", "initial", "final", "delta"]
    return _format_table(headers, rows)


async def run_arbitrage_bot(
    config: BotConfig,
    symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    stop_event: Optional[asyncio.Event] = None,
    demo_binance_params: Optional[Dict[Tuple[str, str], BinanceDemoParams]] = None,
    demo_pancake_params: Optional[Dict[Tuple[str, str], PancakeDemoParams]] = None,
) -> None:
    logger.info("Starting arbitrage bot mode=%s symbols=%s", config.mode.value, ",".join(config.symbols))

    client = _build_market_data_client(config, symbol_mapping)
    await client.start()
    logger.info("MarketDataClient started")

    detector = ArbitrageDetector(
        client=client,
        min_profit_pct=config.min_profit_pct,
        symbols=config.symbols,
        exchanges=config.exchanges,
        symbol_mapping=symbol_mapping or {},
    )

    state = BotState()
    exchange_clients = _build_exchange_clients(
        config=config,
        state=state,
        demo_binance=demo_binance_params,
        demo_pancake=demo_pancake_params,
    )

    initial_equity_usdt = 0.0
    total_realized_pnl_usdt = 0.0
    total_trades_executed = 0

    if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
        initial_equity_usdt = _total_usdt_equity(state)
        logger.info("Initial equity_usdt=%s", f"{initial_equity_usdt:.6f}")

    risk = RiskManager(config=config)
    executor = TradeExecutor(exchange_clients=exchange_clients, risk_manager=risk)

    try:
        scan_count = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop event set. Exiting scan loop.")
                break

            scan_count += 1
            logger.info("scan=%s time=%s", scan_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            opportunities: List[ArbitrageOpportunity] = await detector.scan_opportunities()

            if not opportunities:
                await asyncio.sleep(config.scan_interval)
                continue

            opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)

            for opp in opportunities:
                equity_before = (
                    _total_usdt_equity(state)
                    if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO)
                    else None
                )
                trades_before = len(state.executed_trades)

                try:
                    await executor.execute_opportunity(opp)
                except Exception as exc:
                    logger.info("execution failed symbol=%s error=%s", opp.symbol, str(exc))
                    continue

                new_trades = state.executed_trades[trades_before:]
                for t in new_trades:
                    _log_demo_trade(t)

                if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
                    equity_after = _total_usdt_equity(state)
                    pnl = equity_after - float(equity_before or 0.0)
                    total_realized_pnl_usdt += pnl
                    total_trades_executed += 1

                    logger.info(
                        "trade_summary symbol=%s path=BUY %s(%s)->SELL %s(%s) theo_net=%s pnl_usdt=%s equity_usdt=%s",
                        opp.symbol,
                        opp.buy_exchange,
                        opp.buy_instrument,
                        opp.sell_exchange,
                        opp.sell_instrument,
                        f"{opp.net_profit_pct:.6f}",
                        f"{pnl:.6f}",
                        f"{equity_after:.6f}",
                    )
            await asyncio.sleep(config.scan_interval)

    finally:
        await client.stop()
        logger.info("MarketDataClient stopped")

        if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
            final_equity_usdt = _total_usdt_equity(state)
            roi_pct = 0.0
            if initial_equity_usdt > 0:
                roi_pct = (final_equity_usdt - initial_equity_usdt) / initial_equity_usdt * 100.0

            logger.info("final_summary mode=%s trades=%s", config.mode.value, total_trades_executed)
            logger.info("equity_usdt initial=%s final=%s pnl=%s roi_pct=%s",
                        f"{initial_equity_usdt:.6f}",
                        f"{final_equity_usdt:.6f}",
                        f"{(final_equity_usdt - initial_equity_usdt):.6f}",
                        f"{roi_pct:.6f}")

            table = _balances_table(state, config)
            logger.info("balances:\n%s", table)
