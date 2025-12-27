from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..market_data_client import CexConfig, DexConfig, MarketDataClient
from .arbitrage_detector import ArbitrageDetector, ArbitrageOpportunity
from .config import BotConfig, ExecutionMode
from .exchange import (
    BinanceDemoExchangeClient,
    BinanceDemoParams,
    BinanceExchangeClient,
    ExchangeClient,
    PancakeDemoParams,
    PancakeSwapDemoExchangeClient,
    PancakeSwapExchangeClient,
    PaperExchangeClient,
)
from .executor import TradeExecutor
from .risk import RiskManager
from .state import BotState

logger = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


def _dedupe_logger_handlers(l: logging.Logger) -> None:
    """
    Remove duplicated handlers (common cause of double logs).
    Dedupe key: (handler type, stream id if StreamHandler else handler id)
    """
    seen: set[tuple] = set()
    new_handlers: list[logging.Handler] = []
    for h in list(l.handlers):
        stream = getattr(h, "stream", None)
        key = (type(h), id(stream) if stream is not None else id(h))
        if key in seen:
            continue
        seen.add(key)
        new_handlers.append(h)
    l.handlers = new_handlers


def configure_market_data_client_logging(level: int = logging.INFO) -> None:
    """
    Configure logging so that:
      - only 'market_data_client' package logger owns a StreamHandler
      - all child loggers propagate to it (no per-module handlers)
      - duplicated handlers on root/package are removed
    This removes the most common "every line printed twice" problem.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Dedupe root handlers (people often attach multiple StreamHandlers)
    root = logging.getLogger()
    _dedupe_logger_handlers(root)

    pkg = logging.getLogger("market_data_client")
    _dedupe_logger_handlers(pkg)

    if not pkg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        pkg.addHandler(h)

    pkg.setLevel(level)
    pkg.propagate = False

    # Clear handlers of existing child loggers and make them bubble to package logger
    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name.startswith("market_data_client") and name != "market_data_client":
            obj.handlers = []
            obj.propagate = True
            obj.setLevel(level)

    _LOGGING_CONFIGURED = True


# Configure on import (idempotent). If you configure elsewhere, it still won't double-configure.
configure_market_data_client_logging()


def _norm_asset_key(a: Any) -> str:
    return str(a).upper()


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
        logger.info("binance_testnet status=%s endpoint=%s", t.get("status"), t.get("endpoint"))

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


def _snapshot_balances(state: BotState) -> Dict[Tuple[str, str], Dict[str, float]]:
    snap: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, account in state.accounts.items():
        snap[key] = {_norm_asset_key(a): float(v or 0.0) for a, v in (account.balances or {}).items()}
    return snap


def _total_equity_usdt_from_snapshot(
    snapshot: Dict[Tuple[str, str], Dict[str, float]],
    price_usdt: Dict[str, float],
) -> float:
    total = 0.0
    for _key, bals in snapshot.items():
        for asset, amt in (bals or {}).items():
            px = price_usdt.get(asset)
            if px is None:
                continue
            total += float(amt or 0.0) * float(px)
    return total


def _asset_universe_from_config(config: BotConfig) -> set[str]:
    assets: set[str] = set()
    for (_ex, _inst), bals in (config.paper_initial_balances or {}).items():
        assets |= {_norm_asset_key(a) for a in bals.keys()}
    assets |= {"USDT", "USDC", "WBNB", "BNB", "BTC", "BTCB", "ETH", "WETH", "CAKE", "TWT", "SFP"}
    return assets


def _build_symbol_splitter(quotes: set[str]):
    # Longest suffix wins (BTCB before BTC)
    quote_list = sorted({q.upper() for q in quotes}, key=len, reverse=True)

    def split(symbol: str) -> tuple[str, str]:
        s = "".join(ch for ch in symbol.upper() if ch.isalnum())
        for q in quote_list:
            if s.endswith(q) and len(s) > len(q):
                return s[:-len(q)], q
        raise ValueError(f"Cannot split symbol={symbol}. quotes={quote_list}")

    return split


def _derive_price_usdt_from_opps(
    opps: Iterable[ArbitrageOpportunity],
    split_symbol,
) -> Dict[str, float]:
    """
    Build a minimal asset->USDT price map from opportunities.
    Works best when symbols look like BNBUSDT, ETHUSDT, etc.
    """
    px: Dict[str, float] = {"USDT": 1.0, "USDC": 1.0}

    for opp in opps:
        sym = getattr(opp, "symbol", None)
        if not sym:
            continue

        try:
            base, quote = split_symbol(sym)
        except Exception:
            continue

        buy_p = getattr(opp, "buy_price", None)
        sell_p = getattr(opp, "sell_price", None)

        price: Optional[float] = None
        if isinstance(buy_p, (int, float)) and isinstance(sell_p, (int, float)):
            price = (float(buy_p) + float(sell_p)) / 2.0
        elif isinstance(buy_p, (int, float)):
            price = float(buy_p)
        elif isinstance(sell_p, (int, float)):
            price = float(sell_p)

        if price is None:
            continue

        quote = quote.upper()
        base = base.upper()

        if quote in ("USDT", "USDC"):
            px[base] = price

        # Simple aliases
        if base == "BNB":
            px["WBNB"] = px.get("BNB", price)
        if base == "WBNB":
            px["BNB"] = px.get("WBNB", price)
        if base == "ETH":
            px["WETH"] = px.get("ETH", price)
        if base == "WETH":
            px["ETH"] = px.get("WETH", price)

    return px


def _total_equity_usdt(state: BotState, price_usdt: Dict[str, float]) -> float:
    """
    Mark-to-market total equity in USDT using the provided price map.
    Missing assets are skipped but logged once.
    """
    total = 0.0
    missing: set[str] = set()

    for account in state.accounts.values():
        for asset, amt in (account.balances or {}).items():
            a = _norm_asset_key(asset)
            q = float(amt or 0.0)
            if q == 0.0:
                continue

            px = price_usdt.get(a)
            if px is None:
                missing.add(a)
                continue

            total += q * float(px)

    if missing:
        logger.info("equity_usdt(mtm) warning: missing prices for assets=%s", ",".join(sorted(missing)))

    return total


def _total_usdt_equity(state: BotState) -> float:
    """
    Sum of USDT across all accounts (NOT mark-to-market).
    This is a "realized USDT delta" style metric, not true equity if you hold non-USDT assets.
    """
    total = 0.0
    for account in state.accounts.values():
        for asset, amt in (account.balances or {}).items():
            if _norm_asset_key(asset) == "USDT":
                total += float(amt or 0.0)
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
    """
    Show per-account balance changes. Keys are normalized to UPPER strings
    to avoid "config uses str, state uses enum" mismatch.
    """
    rows: List[List[str]] = []

    for (ex, inst), account in sorted(state.accounts.items(), key=lambda x: (x[0][0], x[0][1])):
        raw_init = config.paper_initial_balances.get((ex, inst), {}) or {}
        init = {_norm_asset_key(k): float(v or 0.0) for k, v in raw_init.items()}

        raw_cur = account.balances or {}
        cur = {_norm_asset_key(k): float(v or 0.0) for k, v in raw_cur.items()}

        assets = sorted(set(init.keys()) | set(cur.keys()))
        for asset in assets:
            init_v = float(init.get(asset, 0.0) or 0.0)
            cur_v = float(cur.get(asset, 0.0) or 0.0)
            delta = cur_v - init_v
            rows.append([ex, inst, asset, f"{init_v:.8f}", f"{cur_v:.8f}", f"{delta:.8f}"])

    headers = ["exchange", "instrument", "asset", "initial", "final", "delta"]
    return _format_table(headers, rows)


async def run_arbitrage_bot(
    config: BotConfig,
    symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    stop_event: Optional[asyncio.Event] = None,
    demo_binance_params: Optional[Dict[Tuple[str, str], BinanceDemoParams]] = None,
    demo_pancake_params: Optional[Dict[Tuple[str, str], PancakeDemoParams]] = None,
) -> None:
    # Ensure logging is configured even if this module wasn't imported first in some entrypoints.
    configure_market_data_client_logging()

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

    split_symbol = _build_symbol_splitter(_asset_universe_from_config(config))
    if demo_pancake_params is not None:
        for (ex, inst), p in list(demo_pancake_params.items()):
            if ex in ("PancakeSwapV2", "PancakeSwapV3") and inst == "swap":
                setattr(p, "symbol_splitter", split_symbol)

    exchange_clients = _build_exchange_clients(
        config=config,
        state=state,
        demo_binance=demo_binance_params,
        demo_pancake=demo_pancake_params,
    )

    # For final MTM summary (computed only on exit)
    initial_snapshot = _snapshot_balances(state)
    last_price_usdt: Dict[str, float] = {"USDT": 1.0, "USDC": 1.0}

    # Per-trade USDT-only delta (not MTM)
    total_realized_pnl_usdt = 0.0
    opportunities_executed = 0

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

            # Update price map opportunistically
            if opportunities:
                last_price_usdt = _derive_price_usdt_from_opps(opportunities, split_symbol)

            if not opportunities:
                await asyncio.sleep(config.scan_interval)
                continue

            opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)

            for opp in opportunities:
                equity_before_usdt = (
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

                opportunities_executed += 1

                new_trades = state.executed_trades[trades_before:]
                for t in new_trades:
                    _log_demo_trade(t)

                if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
                    equity_after_usdt = _total_usdt_equity(state)
                    pnl_usdt = equity_after_usdt - float(equity_before_usdt or 0.0)
                    total_realized_pnl_usdt += pnl_usdt

                    logger.info(
                        "trade_summary symbol=%s path=BUY %s(%s)->SELL %s(%s) theo_net=%s pnl_usdt=%s equity_usdt=%s",
                        opp.symbol,
                        opp.buy_exchange,
                        opp.buy_instrument,
                        opp.sell_exchange,
                        opp.sell_instrument,
                        f"{opp.net_profit_pct:.6f}",
                        f"{pnl_usdt:.6f}",
                        f"{equity_after_usdt:.6f}",
                    )

            await asyncio.sleep(config.scan_interval)

    finally:
        await client.stop()
        logger.info("MarketDataClient stopped")

        if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
            # MTM equity computed ONLY here (on exit)
            initial_equity_mtm = _total_equity_usdt_from_snapshot(initial_snapshot, last_price_usdt)
            final_equity_mtm = _total_equity_usdt(state, last_price_usdt)

            pnl_total = final_equity_mtm - initial_equity_mtm
            roi_pct = (pnl_total / initial_equity_mtm * 100.0) if initial_equity_mtm > 0 else 0.0

            trades_logged = len(state.executed_trades)

            logger.info(
                "final_summary mode=%s opportunities=%s trades_logged=%s",
                config.mode.value,
                opportunities_executed,
                trades_logged,
            )
            logger.info(
                "equity_usdt(mtm) initial=%s final=%s pnl=%s roi_pct=%s",
                f"{initial_equity_mtm:.6f}",
                f"{final_equity_mtm:.6f}",
                f"{pnl_total:.6f}",
                f"{roi_pct:.6f}",
            )
            logger.info(
                "realized_usdt_delta (non-mtm) opportunities=%s pnl_usdt=%s",
                opportunities_executed,
                f"{total_realized_pnl_usdt:.6f}",
            )

            table = _balances_table(state, config)
            logger.info("balances:\n%s", table)
