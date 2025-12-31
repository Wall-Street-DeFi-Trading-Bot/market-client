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

    for name, obj in logging.root.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name.startswith("market_data_client") and name != "market_data_client":
            obj.handlers = []
            obj.propagate = True
            obj.setLevel(level)

    _LOGGING_CONFIGURED = True


configure_market_data_client_logging()


def _norm_asset_key(a: Any) -> str:
    return str(a).upper()


def _avg(xs: List[Optional[float]]) -> Optional[float]:
    ys = [float(x) for x in xs if isinstance(x, (int, float))]
    return (sum(ys) / len(ys)) if ys else None


def _sum(xs: List[Optional[float]]) -> Optional[float]:
    ys = [float(x) for x in xs if isinstance(x, (int, float))]
    return sum(ys) if ys else None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "None"
    return f"{x * 100:.6f}%"


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "None"
    return f"{x:.10f}"


def _pick_num(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _dedupe_per_block_results(per: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    per_block_results may contain duplicated rows (same fork_block + same tx)
    and sometimes one of those duplicates has missing fields (None).
    Keep only the "best" row per key.
    """
    best: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

    for r in per:
        if not isinstance(r, dict):
            continue

        fb = r.get("fork_block")
        txh = r.get("tx_hash") or r.get("tx")

        key = (fb, txh)

        fee_q = _pick_num(r, ["fee_quote"])
        approve_g = _pick_num(r, ["approve_gas_cost_native", "approve_gas_native", "approve_gas_native_bnb"])
        total_g = _pick_num(r, ["total_gas_cost_native", "total_gas_native", "total_gas_native_bnb"])
        rvh_bps = _pick_num(r, ["return_vs_hint_bps", "slip_bps"])
        if rvh_bps is None:
            rvh = _pick_num(r, ["return_vs_hint"])
            if isinstance(rvh, (int, float)):
                rvh_bps = float(rvh) * 10000.0

        # Skip completely empty rows (prevents None spam)
        if fee_q is None and approve_g is None and total_g is None and rvh_bps is None:
            continue

        score = (
            (1 if fee_q is not None else 0)
            + (1 if approve_g is not None else 0)
            + (1 if total_g is not None else 0)
            + (1 if rvh_bps is not None else 0)
        )

        prev = best.get(key)
        if prev is None or score > prev.get("_score", -1):
            r2 = dict(r)
            r2["_score"] = score
            best[key] = r2

    out = list(best.values())
    out.sort(key=lambda x: (x.get("fork_block") is None, x.get("fork_block")))
    for r in out:
        r.pop("_score", None)
    return out


def _extract_dex_summary(p: Dict[str, Any]) -> Dict[str, Optional[float]]:
    raw_per = p.get("per_block_results") or []
    per = _dedupe_per_block_results(raw_per) if isinstance(raw_per, list) else []

    def pick_avg_any(keys: List[str]) -> Optional[float]:
        for k in keys:
            v = p.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        for k in keys:
            v = _avg([r.get(k) for r in per])
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def sum_per_any(keys: List[str]) -> Optional[float]:
        # Only sum from per-block rows (avoid misreading avg fields from p)
        for k in keys:
            v = _sum([r.get(k) for r in per])
            if isinstance(v, (int, float)):
                return float(v)
        return None

    fee_quote_avg = pick_avg_any(["avg_fee_quote", "fee_quote"])
    swap_pnl_quote_avg = pick_avg_any(["avg_swap_pnl_quote", "swap_pnl_quote"])

    slip_bps_avg = pick_avg_any(["avg_slip_bps"])
    if slip_bps_avg is None:
        rvh = pick_avg_any(["avg_return_vs_hint", "return_vs_hint"])
        if isinstance(rvh, (int, float)):
            slip_bps_avg = float(rvh) * 10000.0

    approve_gas_native_avg = pick_avg_any(
        ["avg_approve_gas_native", "approve_gas_cost_native", "approve_gas_native"]
    )
    total_gas_native_avg = pick_avg_any(
        ["avg_total_gas_native", "total_gas_cost_native", "total_gas_native"]
    )

    # Sum across 3 runs (fork blocks) from per-block rows
    total_gas_native_sum = sum_per_any(["total_gas_cost_native", "total_gas_native"])

    net_excl_gas = pick_avg_any(["avg_net_return_excl_gas", "net_return_excl_gas"])
    pnl_excl_gas = pick_avg_any(["avg_net_pnl_quote_excl_gas", "net_pnl_quote_excl_gas"])

    return {
        "fee_quote_avg": fee_quote_avg,
        "swap_pnl_quote_avg": swap_pnl_quote_avg,
        "slip_bps_avg": slip_bps_avg,
        "approve_gas_native_avg": approve_gas_native_avg,
        "total_gas_native_avg": total_gas_native_avg,
        "total_gas_native_sum": total_gas_native_sum,
        "net_return_excl_gas": net_excl_gas,
        "net_pnl_quote_excl_gas": pnl_excl_gas,
    }


def _extract_cex_summary(b: Dict[str, Any], side_val: str, qty: Optional[float]) -> Dict[str, Optional[float]]:
    theo = b.get("theoretical_price")
    execp = b.get("execution_price")
    fee_rate = b.get("fee_rate")
    if not (
        isinstance(theo, (int, float))
        and isinstance(execp, (int, float))
        and isinstance(fee_rate, (int, float))
        and isinstance(qty, (int, float))
    ):
        return {
            "fee_quote": None,
            "net_return_excl_gas": None,
            "net_pnl_quote_excl_gas": None,
            "slip_bps": b.get("slippage_bps"),
        }

    theo_f = float(theo)
    execp_f = float(execp)
    fee_rate_f = float(fee_rate)
    qty_f = float(qty)

    expected = theo_f * qty_f
    actual = execp_f * qty_f

    if side_val.upper() == "BUY":
        pnl_quote = expected - actual
    else:
        pnl_quote = actual - expected

    fee_quote = abs(actual) * fee_rate_f
    net_pnl = pnl_quote - fee_quote
    net_ret = (net_pnl / expected) if expected > 0 else None

    return {
        "fee_quote": fee_quote,
        "net_return_excl_gas": net_ret,
        "net_pnl_quote_excl_gas": net_pnl,
        "slip_bps": b.get("slippage_bps"),
    }


def _trade_cost_breakdown(trade: Any) -> Dict[str, float]:
    """
    Returns:
      - fee_quote (USDT quote)
      - approve_gas_avg_bnb (BNB, avg across 3 runs in demo)
      - gas_avg_bnb (BNB, avg total gas across 3 runs in demo)
      - gas_sum_3runs_bnb (BNB, sum total gas across 3 runs in demo)
    """
    meta = getattr(trade, "metadata", None) or {}

    out = {
        "fee_quote": 0.0,
        "approve_gas_avg_bnb": 0.0,
        "gas_avg_bnb": 0.0,
        "gas_sum_3runs_bnb": 0.0,
    }

    if "pancake_demo" in meta:
        d = _extract_dex_summary(meta["pancake_demo"])
        out["fee_quote"] = float(d.get("fee_quote_avg") or 0.0)
        out["approve_gas_avg_bnb"] = float(d.get("approve_gas_native_avg") or 0.0)
        out["gas_avg_bnb"] = float(d.get("total_gas_native_avg") or 0.0)
        out["gas_sum_3runs_bnb"] = float(d.get("total_gas_native_sum") or 0.0)
        return out

    if "binance_demo" in meta:
        side = getattr(trade, "side", "?")
        side_val = getattr(side, "value", str(side))
        qty = getattr(trade, "quantity", None)

        b = meta["binance_demo"]
        s = _extract_cex_summary(b, side_val, float(qty) if isinstance(qty, (int, float)) else None)
        out["fee_quote"] = float(s.get("fee_quote") or 0.0)
        return out

    return out


def _log_pancake_per_blocks(trade: Any) -> None:
    """
    Single place to print pancake_block lines.
    - Dedupes duplicated rows.
    - Skips rows that are all None (prevents None spam).
    """
    meta = getattr(trade, "metadata", None) or {}
    p = meta.get("pancake_demo")
    if not isinstance(p, dict):
        return

    raw_per = p.get("per_block_results") or []
    if not isinstance(raw_per, list) or not raw_per:
        return

    per = _dedupe_per_block_results(raw_per)
    if not per:
        return

    for r in per:
        fb = r.get("fork_block")
        status = r.get("status")
        txh = r.get("tx_hash") or r.get("tx")

        fee_q = _pick_num(r, ["fee_quote"])
        approve_g = _pick_num(r, ["approve_gas_cost_native", "approve_gas_native", "approve_gas_native_bnb"])
        total_g = _pick_num(r, ["total_gas_cost_native", "total_gas_native", "total_gas_native_bnb"])

        rvh_bps = _pick_num(r, ["return_vs_hint_bps", "slip_bps"])
        if rvh_bps is None:
            rvh = _pick_num(r, ["return_vs_hint"])
            if isinstance(rvh, (int, float)):
                rvh_bps = float(rvh) * 10000.0

        logger.info(
            "pancake_block fork_block=%s status=%s tx_hash=%s fee_quote=%s "
            "approve_gas_native_bnb=%s total_gas_native_bnb=%s return_vs_hint_bps=%s",
            fb,
            status,
            txh,
            f"{fee_q:.10f}" if isinstance(fee_q, (int, float)) else "None",
            f"{approve_g:.10f}" if isinstance(approve_g, (int, float)) else "None",
            f"{total_g:.10f}" if isinstance(total_g, (int, float)) else "None",
            f"{rvh_bps:.3f}" if isinstance(rvh_bps, (int, float)) else "None",
        )


def _trade_fee_gas_native(t: Any) -> Tuple[float, float]:
    """
    Returns (fee_quote, gas_native_bnb_avg_per_tx).

    - CEX: fee_quote from binance_demo, gas_native_bnb = 0
    - DEX(BSC Pancake): fee_quote(avg) + gas_native_bnb(avg)
      NOTE: fee_quote is in quote token units, gas is in BNB units (different unit).
    """
    meta = getattr(t, "metadata", None) or {}

    if "binance_demo" in meta:
        side = getattr(t, "side", "?")
        side_val = getattr(side, "value", str(side))
        qty = getattr(t, "quantity", None)

        b = meta["binance_demo"]
        s = _extract_cex_summary(b, side_val, float(qty) if isinstance(qty, (int, float)) else None)
        fee_q = float(s.get("fee_quote") or 0.0)
        return fee_q, 0.0

    if "pancake_demo" in meta:
        p = meta["pancake_demo"]
        d = _extract_dex_summary(p)
        fee_q = float(d.get("fee_quote_avg") or 0.0)
        gas_bnb = float(d.get("total_gas_native_avg") or 0.0)
        return fee_q, gas_bnb

    return 0.0, 0.0


def _log_demo_trade(trade: Any) -> None:
    meta = getattr(trade, "metadata", None) or {}

    ex = getattr(trade, "exchange", "?")
    inst = getattr(trade, "instrument", "?")
    sym = getattr(trade, "symbol", "?")
    side = getattr(trade, "side", "?")
    side_val = getattr(side, "value", str(side))
    qty = getattr(trade, "quantity", None)
    px = getattr(trade, "price", None)

    logger.info(
        "trade executed exchange=%s instrument=%s symbol=%s side=%s qty=%s price=%s",
        ex,
        inst,
        sym,
        side_val,
        f"{qty:.8f}" if isinstance(qty, (int, float)) else str(qty),
        f"{px:.8f}" if isinstance(px, (int, float)) else str(px),
    )

    if "binance_demo" in meta:
        b = meta["binance_demo"]
        s = _extract_cex_summary(b, side_val, float(qty) if isinstance(qty, (int, float)) else None)
        logger.info(
            "cex_metrics slip_bps=%s fee_quote=%s net_excl_gas=%s pnl_excl_gas=%s",
            s.get("slip_bps"),
            _fmt_money(s.get("fee_quote")),
            _fmt_pct(s.get("net_return_excl_gas")),
            _fmt_money(s.get("net_pnl_quote_excl_gas")),
        )

    if "pancake_demo" in meta:
        p = meta["pancake_demo"]
        d = _extract_dex_summary(p)

        logger.info(
            "dex_summary slip_bps=%s fee_quote(avg)=%s "
            "approve_gas_native(avg_bnb)=%s total_gas_native(avg_bnb)=%s total_gas_native(sum_3runs_bnb)=%s "
            "pnl_excl_gas=%s net_excl_gas=%s",
            (f"{d.get('slip_bps_avg'):.3f}" if isinstance(d.get("slip_bps_avg"), (int, float)) else "None"),
            _fmt_money(d.get("fee_quote_avg")),
            _fmt_money(d.get("approve_gas_native_avg")),
            _fmt_money(d.get("total_gas_native_avg")),
            _fmt_money(d.get("total_gas_native_sum")),
            _fmt_money(d.get("net_pnl_quote_excl_gas")),
            _fmt_pct(d.get("net_return_excl_gas")),
        )

        # NOTE: only print pancake_block here (deduped + None-safe)
        _log_pancake_per_blocks(trade)


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
    quote_list = sorted({q.upper() for q in quotes}, key=len, reverse=True)

    def split(symbol: str) -> tuple[str, str]:
        s = "".join(ch for ch in symbol.upper() if ch.isalnum())
        for q in quote_list:
            if s.endswith(q) and len(s) > len(q):
                return s[: -len(q)], q
        raise ValueError(f"Cannot split symbol={symbol}. quotes={quote_list}")

    return split


def _derive_price_usdt_from_opps(
    opps: Iterable[ArbitrageOpportunity],
    split_symbol,
) -> Dict[str, float]:
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
    configure_market_data_client_logging()

    logger.info(
        "Starting arbitrage bot mode=%s symbols=%s",
        config.mode.value,
        ",".join(config.symbols),
    )

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

    initial_snapshot = _snapshot_balances(state)
    last_price_usdt: Dict[str, float] = {"USDT": 1.0, "USDC": 1.0}

    total_realized_pnl_usdt = 0.0

    # Cumulative report (gas is tracked in native BNB)
    cum_fee_quote = 0.0
    cum_gas_avg_bnb = 0.0
    cum_gas_sum_3runs_bnb = 0.0
    cum_gross_pnl_quote = 0.0
    cum_net_after_fee_quote = 0.0
    cum_notional_quote = 0.0

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
                    # NOTE: pancake_block logs are printed ONLY inside _log_demo_trade now.
                    _log_demo_trade(t)

                if config.mode in (ExecutionMode.PAPER, ExecutionMode.DEMO):
                    equity_after_usdt = _total_usdt_equity(state)
                    pnl_usdt = equity_after_usdt - float(equity_before_usdt or 0.0)
                    total_realized_pnl_usdt += pnl_usdt

                    buy_trade = next(
                        (t for t in new_trades if getattr(getattr(t, "side", None), "value", None) == "BUY"),
                        None,
                    )
                    sell_trade = next(
                        (t for t in new_trades if getattr(getattr(t, "side", None), "value", None) == "SELL"),
                        None,
                    )

                    if buy_trade and sell_trade:
                        qty0 = float(getattr(buy_trade, "quantity", 0.0) or 0.0)
                        buy_px = float(getattr(buy_trade, "price", 0.0) or 0.0)
                        sell_px = float(getattr(sell_trade, "price", 0.0) or 0.0)

                        notional_quote = buy_px * qty0
                        swap_profit_quote = (sell_px - buy_px) * qty0

                        buy_cost = _trade_cost_breakdown(buy_trade)
                        sell_cost = _trade_cost_breakdown(sell_trade)

                        fee_buy = float(buy_cost["fee_quote"] or 0.0)
                        fee_sell = float(sell_cost["fee_quote"] or 0.0)
                        fee_total = fee_buy + fee_sell

                        approve_buy_avg = float(buy_cost["approve_gas_avg_bnb"] or 0.0)
                        approve_sell_avg = float(sell_cost["approve_gas_avg_bnb"] or 0.0)
                        approve_total_avg = approve_buy_avg + approve_sell_avg

                        gas_buy_avg = float(buy_cost["gas_avg_bnb"] or 0.0)
                        gas_sell_avg = float(sell_cost["gas_avg_bnb"] or 0.0)
                        gas_total_avg = gas_buy_avg + gas_sell_avg

                        gas_buy_sum = float(buy_cost["gas_sum_3runs_bnb"] or 0.0)
                        gas_sell_sum = float(sell_cost["gas_sum_3runs_bnb"] or 0.0)
                        gas_total_sum = gas_buy_sum + gas_sell_sum

                        net_after_fee_quote = swap_profit_quote - fee_total
                        ret_after_fee = (net_after_fee_quote / notional_quote) if notional_quote > 0 else 0.0

                        logger.info(
                            "final_return symbol=%s qty=%s notional_quote=%s "
                            "swap_profit_quote=%s "
                            "fee_quote buy=%s sell=%s total=%s "
                            "approve_gas_avg_bnb buy=%s sell=%s total=%s "
                            "gas_avg_bnb buy=%s sell=%s total=%s "
                            "gas_sum_3runs_bnb buy=%s sell=%s total=%s "
                            "net_after_fee_quote=%s ret_after_fee_pct=%s",
                            opp.symbol,
                            f"{qty0:.8f}",
                            f"{notional_quote:.10f}",
                            f"{swap_profit_quote:.10f}",
                            f"{fee_buy:.10f}",
                            f"{fee_sell:.10f}",
                            f"{fee_total:.10f}",
                            f"{approve_buy_avg:.10f}",
                            f"{approve_sell_avg:.10f}",
                            f"{approve_total_avg:.10f}",
                            f"{gas_buy_avg:.10f}",
                            f"{gas_sell_avg:.10f}",
                            f"{gas_total_avg:.10f}",
                            f"{gas_buy_sum:.10f}",
                            f"{gas_sell_sum:.10f}",
                            f"{gas_total_sum:.10f}",
                            f"{net_after_fee_quote:.10f}",
                            f"{ret_after_fee * 100.0:.6f}%",
                        )

                        # Cumulative
                        cum_fee_quote += fee_total
                        cum_gas_avg_bnb += gas_total_avg
                        cum_gas_sum_3runs_bnb += gas_total_sum
                        cum_gross_pnl_quote += swap_profit_quote
                        cum_net_after_fee_quote += net_after_fee_quote
                        cum_notional_quote += notional_quote

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

            ret_after_fee_total = (cum_net_after_fee_quote / cum_notional_quote) if cum_notional_quote > 0 else 0.0

            logger.info(
                "cumulative_report fee_total_quote=%s gas_total_avg_native_bnb=%s gas_total_sum_3runs_bnb=%s "
                "gross_pnl_quote=%s net_after_fee_quote=%s ret_after_fee_pct=%s notional_quote=%s",
                f"{cum_fee_quote:.10f}",
                f"{cum_gas_avg_bnb:.10f}",
                f"{cum_gas_sum_3runs_bnb:.10f}",
                f"{cum_gross_pnl_quote:.10f}",
                f"{cum_net_after_fee_quote:.10f}",
                f"{ret_after_fee_total * 100.0:.6f}%",
                f"{cum_notional_quote:.10f}",
            )

            logger.info("balances:\n%s", table)
