import asyncio
import csv
import os
import re
import time
import logging
from copy import deepcopy
from datetime import timedelta
from typing import Dict, Tuple, Optional, List, Iterable

import dataclasses

import nats
from nats.aio.client import Client as NATS
from nats.js import api as jsapi
from nats.js.errors import NotFoundError

import proto.market_pb2 as pb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# ========================= CSV (fsync) =========================
_csv_lock = asyncio.Lock()
_csv_writers: Dict[str, csv.DictWriter] = {}
_csv_files: Dict[str, any] = {}


def _ensure_writer(path: str, fieldnames: List[str]) -> csv.DictWriter:
    if path in _csv_writers:
        return _csv_writers[path]
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    
    if f.tell() == 0:
        w.writeheader()
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    
    _csv_writers[path] = w
    _csv_files[path] = f
    return w


async def _csv_write(path: str, fieldnames: List[str], row: dict):
    async with _csv_lock:
        try:
            w = _ensure_writer(path, fieldnames)
            w.writerow(row)
            f = _csv_files[path]
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        except Exception as e:
            logger.error(f"CSV write error for {path}: {e}", exc_info=True)


async def _csv_close_all():
    async with _csv_lock:
        for f in list(_csv_files.values()):
            try:
                f.flush()
                os.fsync(f.fileno())
                f.close()
            except Exception:
                pass
        _csv_writers.clear()
        _csv_files.clear()


# ========================= utils =========================
def _iso_from_ns(ns: int) -> str:
    if not ns:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ns / 1e9))


def _instrument_short(code: int) -> str:
    try:
        if code == pb.Instrument.INSTRUMENT_SPOT:
            return "spot"
        if code == pb.Instrument.INSTRUMENT_PERPETUAL:
            return "perpetual"
        if code == pb.Instrument.INSTRUMENT_SWAP:
            return "swap"
        return pb.Instrument.Name(code).replace("INSTRUMENT_", "").lower()
    except Exception:
        return "unknown"


def _norm_symbol(sym: str) -> str:
    # "ETH/USDT" | "eth-usdt" -> "ETHUSDT"
    return re.sub(r"[/\-\s]", "", sym).upper()


def _sanitize_for_durable(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)[:200]


# ========================= data caches =========================
@dataclasses.dataclass
class Tick:
    bid: float
    ask: float
    mid: float
    ts_ns: int
    instrument: str
    latency_ms: Optional[float] = None


@dataclasses.dataclass
class Funding:
    mark_price: float
    index_price: float
    est_settle_price: float
    funding_rate: float
    next_funding_time_ms: int
    ts_ns: int
    latency_ms: Optional[float] = None


@dataclasses.dataclass
class Fee:
    maker_rate: float
    taker_rate: float
    ts_ns: int
    latency_ms: Optional[float] = None


@dataclasses.dataclass
class Volume:
    volume0: float
    volume1: float
    high: float
    low: float
    trades: int
    ts_ns: int
    latency_ms: Optional[float] = None


@dataclasses.dataclass
class Slippage:
    impact_bps01: float
    impact_bps10: float
    block_number: int
    tx_hash: str
    ts_ns: int
    latency_ms: Optional[float] = None


@dataclasses.dataclass
class DexPair:
    exchange: str
    chain: str
    pair: str
    token0: str
    token1: str
    price01: float
    price10: float
    invert_for_quote: bool
    ts_ns: int
    latency_ms: Optional[float] = None


# ========================= config model =========================
@dataclasses.dataclass
class CexConfig:
    exchange: str
    symbols: List[str]
    instruments: List[str]
    want: Iterable[str] = ("tick", "funding", "fee", "volume")


@dataclasses.dataclass
class DexConfig:
    exchange: str
    chain: str
    pairs: List[str]
    want: Iterable[str] = ("tick", "slippage", "fee", "volume")


# ========================= client =========================
class MarketDataClient:
    """
    생성 시 전달한 (type, exchange, instrument/chain, symbol/PAIR)에 따라 구독을 자동 구성.
    - type="cex": exchange + instruments + symbols
    - type="dex": exchange + chain + pairs
    JetStream 또는 일반 NATS 모두 지원.
    """

    def __init__(
        self,
        nats_url: str = None,
        use_jetstream: bool = True,
        stream: Optional[str] = "MD",
        durable_prefix: str = "mdc",
        csv_dir: Optional[str] = "./csv",
        enable_csv: bool = True,
        cex: Optional[List[CexConfig]] = None,
        dex: Optional[List[DexConfig]] = None,
    ):
        self.nats_url = nats_url or os.getenv("NATS_URL", "nats://127.0.0.1:4222")
        self.use_js = use_jetstream
        self.stream = stream
        self.durable_prefix = durable_prefix
        self.csv_dir = csv_dir
        self.enable_csv = enable_csv

        self.cex_cfgs = cex or []
        self.dex_cfgs = dex or []

        self.nc: Optional[NATS] = None
        self.js = None
        self._subs = []

        # caches (완전한 Lock 보호)
        self.ticks: Dict[Tuple[str, str, str], Tick] = {}
        self.funding: Dict[Tuple[str, str], Funding] = {}
        self.fees: Dict[Tuple[str, str, str], Fee] = {}
        self.volume: Dict[Tuple[str, str, str], Volume] = {}
        self.slip: Dict[Tuple[str, str], Slippage] = {}
        self.dex_pairs: Dict[Tuple[str, str, str], DexPair] = {}

        self._lock = asyncio.Lock()
        self._listeners = []

    # --------------- lifecycle ---------------
    async def start(self):
        try:
            self.nc = await nats.connect(
                self.nats_url,
                connect_timeout=3,
                allow_reconnect=True,
                reconnect_time_wait=1,
                max_reconnect_attempts=-1,
            )
            logger.info(f"Connected to NATS: {self.nats_url}")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}", exc_info=True)
            raise

        if self.use_js:
            self.js = self.nc.jetstream(timeout=5)
            if self.stream:
                try:
                    await self.js.stream_info(self.stream)
                    logger.info(f"JetStream {self.stream} exists")
                except NotFoundError:
                    try:
                        await self.js.add_stream(jsapi.StreamConfig(
                            name=self.stream,
                            subjects=["md.>"],
                            storage=_enum(jsapi.StorageType, "File", "FILE"),
                            retention=_enum(jsapi.RetentionPolicy, "Limits", "LIMITS"),
                            discard=_enum(jsapi.DiscardPolicy, "Old", "OLD"),
                            max_msgs=-1,
                            max_bytes=-1,
                            max_age=0,
                            num_replicas=1,
                        ))
                        logger.info(f"Created JetStream {self.stream}")
                    except Exception as e:
                        logger.warning(f"Failed to create stream: {e}")

        for cfg in self.cex_cfgs:
            await self._subscribe_cex(cfg)
        for cfg in self.dex_cfgs:
            await self._subscribe_dex(cfg)

        logger.info("MarketDataClient started successfully")

    async def stop(self):
        await _csv_close_all()
        for _, sub in self._subs:
            try:
                await sub.drain()
            except Exception:
                pass
        if self.nc:
            await self.nc.drain()
        logger.info("MarketDataClient stopped")

    def add_listener(self, callback, **filters):
        """
        callback(ev: dict) 코루틴을 등록.
        filters: kind/exchange/symbol/instrument/pair 중 일부를 집합(set)이나 리스트로 전달 가능.
        """
        self._listeners.append((callback, filters or {}))
        logger.debug(f"Added listener with filters: {filters}")

    def remove_listener(self, callback):
        before = len(self._listeners)
        self._listeners = [(cb, f) for (cb, f) in self._listeners if cb is not callback]
        after = len(self._listeners)
        logger.debug(f"Removed {before - after} listener(s)")

    def _match_filters(self, ev, flt):
        def _ok(key):
            v = flt.get(key)
            return (not v) or (ev.get(key) in v)

        return _ok("kind") and _ok("exchange") and _ok("symbol") and _ok("instrument") and _ok("pair")

    async def _emit(self, ev: dict):
        for cb, flt in list(self._listeners):
            try:
                if self._match_filters(ev, flt):
                    await cb(ev)
            except Exception as e:
                logger.error(f"Listener {cb.__name__} error: {e}", exc_info=True)

    # --------------- subscribe builders ---------------
    async def _subscribe_cex(self, cfg: CexConfig):
        ex = cfg.exchange
        wants = set(cfg.want)
        for sym in map(_norm_symbol, cfg.symbols):
            for inst in cfg.instruments:
                inst = inst.lower()
                if "tick" in wants:
                    await self._sub(f"md.tick.cex.{ex}.{sym}.{inst}")
                if "fee" in wants:
                    await self._sub(f"md.fee.cex.{ex}.{sym}.{inst}")
                if "volume" in wants:
                    await self._sub(f"md.volume.cex.{ex}.{sym}.{inst}")
                if inst in ("perp", "perpetual") and "funding" in wants:
                    await self._sub(f"md.funding.cex.{ex}.{sym}.perpetual")

    async def _subscribe_dex(self, cfg: DexConfig):
        ex, chain = cfg.exchange, cfg.chain
        wants = set(cfg.want)
        for pair in cfg.pairs:
            if "tick" in wants:
                await self._sub(f"md.tick.dex.{ex}.{chain}.{pair}.swap")
            if "slippage" in wants:
                await self._sub(f"md.slippage.dex.{ex}.{chain}.{pair}.swap")
            if "fee" in wants:
                await self._sub(f"md.fee.dex.{ex}.{chain}.{pair}.swap")
            if "volume" in wants:
                await self._sub(f"md.volume.dex.{ex}.{chain}.{pair}.swap")

    async def _sub(self, subject: str):
        async def _cb(msg, _subject=subject):
            await self._on_msg(_subject, msg)

        if self.use_js:
            durable = _sanitize_for_durable(f"{self.durable_prefix}.{subject}")
            sub = await self.js.subscribe(
                subject,
                durable=durable,
                manual_ack=True,
                cb=_cb,
                config=jsapi.ConsumerConfig(
                    durable_name=durable,
                    ack_policy=jsapi.AckPolicy.EXPLICIT,
                    deliver_policy=jsapi.DeliverPolicy.NEW,
                    ack_wait=timedelta(seconds=60),
                    max_deliver=-1,
                ),
            )
        else:
            sub = await self.nc.subscribe(subject, cb=_cb)
        
        self._subs.append((subject, sub))
        logger.debug(f"Subscribed to {subject}")

    # --------------- message handler ---------------
    async def _on_msg(self, subject: str, msg):
        try:
            md = pb.MarketData()
            md.ParseFromString(msg.data)
            
            which = md.WhichOneof("data")
            node = getattr(md, which, None)
            
            if node is None:
                if self.use_js:
                    await msg.ack()
                return

            exch = md.header.exchange
            sym = md.header.symbol or ""
            inst = _instrument_short(md.header.instrument)
            ts_ns = md.header.ts_ns

            now_ns = time.time_ns()
            lat_ms = ((now_ns - ts_ns) / 1e6) if ts_ns else None

            if subject.startswith("md.tick.dex.") and which == "dexSwapL1":
                parts = subject.split(".")
                ex, chain, pair = (parts[3], parts[4], parts[5]) if len(parts) >= 7 else (exch, "BSC", sym)
                await self._h_dexswapl1(ex, chain, pair, node, ts_ns, lat_ms)
                await self._h_dex_as_tick(ex, pair, node, ts_ns, lat_ms)

            elif which == "tick":
                await self._h_tick(exch, sym, inst, node, ts_ns, lat_ms)
            elif which == "funding":
                await self._h_funding(exch, sym, node, ts_ns, lat_ms)
            elif which == "fee":
                await self._h_fee(exch, sym, inst, node, ts_ns, lat_ms)
            elif which == "volume":
                await self._h_volume(exch, sym, inst, node, ts_ns, lat_ms)
            elif which == "slippage":
                await self._h_slippage(exch, sym, node, ts_ns, lat_ms)
            else:
                logger.debug(f"Unknown message type: {which}")

            if self.use_js:
                await msg.ack()

        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
            if self.use_js:
                try:
                    await msg.nak()
                except Exception:
                    pass

    # --------------- handlers (cache + CSV + emit) ---------------
    async def _h_tick(self, exchange: str, symbol: str, instrument: str, t, ts_ns: int, lat_ms: Optional[float]):
        async with self._lock:
            self.ticks[(exchange, symbol, instrument)] = Tick(t.bid, t.ask, t.mid, ts_ns, instrument, latency_ms=lat_ms)

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "ticks.csv"),
                ["timestamp", "exchange", "symbol", "instrument", "bid", "ask", "mid", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "symbol": symbol,
                    "instrument": instrument,
                    "bid": t.bid,
                    "ask": t.ask,
                    "mid": t.mid,
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        await self._emit({
            "kind": "tick",
            "exchange": exchange,
            "symbol": symbol,
            "instrument": instrument,
            "mid": t.mid,
            "bid": t.bid,
            "ask": t.ask,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    async def _h_funding(self, exchange: str, symbol: str, f, ts_ns: int, lat_ms: Optional[float]):
        async with self._lock:
            self.funding[(exchange, symbol)] = Funding(
                f.mark_price,
                f.index_price,
                f.estimated_settle_price,
                f.funding_rate,
                f.next_funding_time,
                ts_ns,
                latency_ms=lat_ms
            )

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "funding.csv"),
                ["timestamp", "exchange", "symbol", "mark_price", "index_price", "est_settle_price", "funding_rate", "next_funding_time_ms", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "symbol": symbol,
                    "mark_price": f.mark_price,
                    "index_price": f.index_price,
                    "est_settle_price": f.estimated_settle_price,
                    "funding_rate": f.funding_rate,
                    "next_funding_time_ms": f.next_funding_time,
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        await self._emit({
            "kind": "funding",
            "exchange": exchange,
            "symbol": symbol,
            "rate": f.funding_rate,
            "next_ms": f.next_funding_time,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    async def _h_fee(self, exchange: str, symbol: str, instrument: str, f, ts_ns: int, lat_ms: Optional[float]):
        async with self._lock:
            self.fees[(exchange, symbol, instrument)] = Fee(f.maker_rate, f.taker_rate, ts_ns, latency_ms=lat_ms)

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "fees.csv"),
                ["timestamp", "exchange", "symbol", "instrument", "taker_bps", "maker_bps", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "symbol": symbol,
                    "instrument": instrument,
                    "taker_bps": f"{(f.taker_rate * 10000.0):.6f}",
                    "maker_bps": f"{(f.maker_rate * 10000.0):.6f}",
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        await self._emit({
            "kind": "fee",
            "exchange": exchange,
            "symbol": symbol,
            "instrument": instrument,
            "maker_rate": f.maker_rate,
            "taker_rate": f.taker_rate,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    async def _h_volume(self, exchange: str, symbol: str, instrument: str, v, ts_ns: int, lat_ms: Optional[float]):
        async with self._lock:
            self.volume[(exchange, symbol, instrument)] = Volume(v.volume0, v.volume1, v.high, v.low, v.trades, ts_ns, latency_ms=lat_ms)

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "volume.csv"),
                ["timestamp", "exchange", "symbol", "instrument", "volume0", "volume1", "high", "low", "trades", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "symbol": symbol,
                    "instrument": instrument,
                    "volume0": v.volume0,
                    "volume1": v.volume1,
                    "high": v.high,
                    "low": v.low,
                    "trades": v.trades,
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        await self._emit({
            "kind": "volume",
            "exchange": exchange,
            "symbol": symbol,
            "instrument": instrument,
            "volume0": v.volume0,
            "volume1": v.volume1,
            "high": v.high,
            "low": v.low,
            "trades": v.trades,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    async def _h_dexswapl1(self, exchange: str, chain: str, pair: str, ds, ts_ns: int, lat_ms: Optional[float]):
        token0 = getattr(ds, "token0", "")
        token1 = getattr(ds, "token1", "")
        invert = bool(getattr(ds, "invert_for_quote", False))
        price01 = float(ds.price01)
        price10 = float(ds.price10)
        price_qb = price10 if invert else price01

        async with self._lock:
            self.dex_pairs[(exchange, chain, pair)] = DexPair(
                exchange=exchange,
                chain=chain,
                pair=pair,
                token0=token0,
                token1=token1,
                price01=price01,
                price10=price10,
                invert_for_quote=invert,
                ts_ns=ts_ns,
                latency_ms=lat_ms
            )

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "swaps.csv"),
                ["timestamp", "exchange", "chain", "pair", "token0", "token1", "price", "price01", "price10", "invert_for_quote", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "chain": chain,
                    "pair": pair,
                    "token0": token0,
                    "token1": token1,
                    "price": price_qb,
                    "price01": price01,
                    "price10": price10,
                    "invert_for_quote": int(invert),
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        # 1) dexSwapL1 원본 이벤트
        await self._emit({
            "kind": "dexSwapL1",
            "exchange": exchange,
            "pair": pair,
            "token0": token0,
            "token1": token1,
            "price": price_qb,
            "price01": price01,
            "price10": price10,
            "invert_for_quote": invert,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

        # 2) swap 틱으로도 발행 (get_latest_price(...,"swap") 와 동일)
        await self._emit({
            "kind": "tick",
            "exchange": exchange,
            "symbol": pair,
            "instrument": "swap",
            "mid": price_qb,
            "bid": price_qb,
            "ask": price_qb,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    async def _h_dex_as_tick(self, exchange: str, pair: str, ds, ts_ns: int, lat_ms: Optional[float]):
        invert = bool(getattr(ds, "invert_for_quote", False))
        price01 = float(ds.price01)
        price10 = float(ds.price10)
        price_qb = price10 if invert else price01

        # DEX는 bid/ask가 없으니 mid로 동일 세팅
        async with self._lock:
            self.ticks[(exchange, pair, "swap")] = Tick(
                bid=price_qb,
                ask=price_qb,
                mid=price_qb,
                ts_ns=ts_ns,
                instrument="swap",
                latency_ms=lat_ms
            )

    async def _h_slippage(self, exchange: str, symbol: str, sl, ts_ns: int, lat_ms: Optional[float]):
        async with self._lock:
            self.slip[(exchange, symbol)] = Slippage(
                sl.impact_bps01,
                sl.impact_bps10,
                sl.block_number,
                sl.tx_hash,
                ts_ns,
                latency_ms=lat_ms
            )

        if self.enable_csv and self.csv_dir:
            await _csv_write(
                os.path.join(self.csv_dir, "slippage.csv"),
                ["timestamp", "exchange", "pair", "bps01", "bps10", "block", "tx", "lat_ms"],
                {
                    "timestamp": _iso_from_ns(ts_ns),
                    "exchange": exchange,
                    "pair": symbol,
                    "bps01": f"{sl.impact_bps01:.8f}",
                    "bps10": f"{sl.impact_bps10:.8f}",
                    "block": sl.block_number,
                    "tx": sl.tx_hash,
                    "lat_ms": f"{lat_ms:.2f}" if lat_ms is not None else ""
                }
            )

        await self._emit({
            "kind": "slippage",
            "exchange": exchange,
            "pair": symbol,
            "bps01": sl.impact_bps01,
            "bps10": sl.impact_bps10,
            "block": sl.block_number,
            "tx": sl.tx_hash,
            "ts_ns": ts_ns,
            "lat_ms": lat_ms,
        })

    # --------------- simple query APIs ---------------
    async def get_latest_price(self, symbol: str, exchange: str, instrument: str = "spot") -> Optional[float]:
        sym = _norm_symbol(symbol)
        inst = instrument.lower()
        async with self._lock:
            t = self.ticks.get((exchange, sym, inst))
        return t.mid if t else None

    async def get_funding_rate(self, symbol: str, exchange: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            f = self.funding.get((exchange, sym))
        return f.funding_rate if f else None

    async def get_fee(self, symbol: str, exchange: str, instrument: str) -> Optional[dict]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            f = self.fees.get((exchange, sym, instrument.lower()))
        if not f:
            return None
        return {
            "maker_rate": f.maker_rate,
            "taker_rate": f.taker_rate,
            "maker_bps": f.maker_rate * 10000.0,
            "taker_bps": f.taker_rate * 10000.0,
            "ts": f.ts_ns
        }

    async def get_volume(self, symbol: str, exchange: str, instrument: str) -> Optional[dict]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            v = self.volume.get((exchange, sym, instrument.lower()))
        if not v:
            return None
        return {
            "volume0": v.volume0,
            "volume1": v.volume1,
            "high": v.high,
            "low": v.low,
            "trades": v.trades,
            "ts": v.ts_ns
        }

    async def get_slippage_latest(self, pair: str, exchange: str) -> Optional[dict]:
        async with self._lock:
            s = self.slip.get((exchange, pair))
        if not s:
            return None
        return {
            "bps01": s.impact_bps01,
            "bps10": s.impact_bps10,
            "ts": s.ts_ns
        }

    async def get_latency_price(self, symbol: str, exchange: str, instrument: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            t = self.ticks.get((exchange, sym, instrument.lower()))
        return t.latency_ms if t else None

    async def get_latency_funding(self, symbol: str, exchange: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            f = self.funding.get((exchange, sym))
        return f.latency_ms if f else None

    async def get_latency_fee(self, symbol: str, exchange: str, instrument: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            fa = self.fees.get((exchange, sym, instrument.lower()))
        return fa.latency_ms if fa else None

    async def get_latency_volume(self, symbol: str, exchange: str, instrument: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        async with self._lock:
            v = self.volume.get((exchange, sym, instrument.lower()))
        return v.latency_ms if v else None

    async def get_latency_slippage(self, pair: str, exchange: str) -> Optional[float]:
        async with self._lock:
            s = self.slip.get((exchange, pair))
        return s.latency_ms if s else None

    async def get_tick(self, symbol: str, exchange: str, instrument: str = "spot"):
        """(mid, ts, latency)을 같은 스냅샷으로 한번에 반환"""
        sym = _norm_symbol(symbol)
        key = (exchange, sym, instrument.lower())
        async with self._lock:
            t = self.ticks.get(key)
            return None if not t else _dc_copy(t)

    async def get_funding_meta(self, symbol: str, exchange: str):
        sym = _norm_symbol(symbol)
        key = (exchange, sym)
        async with self._lock:
            f = self.funding.get(key)
            return None if not f else _dc_copy(f)

    async def get_fee_meta(self, symbol: str, exchange: str, instrument: str):
        sym = _norm_symbol(symbol)
        key = (exchange, sym, instrument.lower())
        async with self._lock:
            v = self.fees.get(key)
            return None if not v else _dc_copy(v)

    async def get_volume_meta(self, symbol: str, exchange: str, instrument: str):
        sym = _norm_symbol(symbol)
        key = (exchange, sym, instrument.lower())
        async with self._lock:
            v = self.volume.get(key)
            return None if not v else _dc_copy(v)

    async def get_slippage_meta(self, pair: str, exchange: str):
        key = (exchange, pair)
        async with self._lock:
            s = self.slip.get(key)
            return None if not s else _dc_copy(s)

    # 편의: 가격+지연을 한 번에
    async def get_latest_price_with_latency(self, symbol: str, exchange: str, instrument: str = "spot"):
        t = await self.get_tick(symbol, exchange, instrument)
        if not t:
            return None
        return {
            "mid": t.mid,
            "bid": t.bid,
            "ask": t.ask,
            "lat_ms": t.latency_ms,
            "ts_ns": t.ts_ns,
            "ts_iso": _iso_from_ns(t.ts_ns),
            "instrument": t.instrument,
            "exchange": exchange,
            "symbol": _norm_symbol(symbol),
        }

    async def get_dex_price_qb(self, pair: str, exchange: str, chain: str = "BSC"):
        key = (exchange, chain, pair)
        async with self._lock:
            dp = self.dex_pairs.get(key)
            if not dp:
                return None
            price_qb = dp.price10 if dp.invert_for_quote else dp.price01
            return {
                "price_qb": price_qb,
                "ts_ns": dp.ts_ns,
                "ts_iso": _iso_from_ns(dp.ts_ns),
                "lat_ms": dp.latency_ms
            }

    async def get_spot_usdt_from_dex(self, symbol_usdt: str, exchange: str, chain: str = "BSC", router_token: str = "WBNB"):
        """
        ETHUSDT 요청 시:
          router= WBNB
          price(ETH/USDT) = price(WBNB/USDT) / price(WBNB/ETH)
        둘 다 같은 락에서 읽어 원자적 스냅샷으로 계산.
        """
        sym = _norm_symbol(symbol_usdt)
        assert sym.endswith("USDT"), "symbol_usdt must end with USDT"
        base = sym[:-4]

        rq = f"{router_token}USDT"
        rb = f"{router_token}{base}"

        async with self._lock:
            dp_rq = self.dex_pairs.get((exchange, chain, rq))
            dp_rb = self.dex_pairs.get((exchange, chain, rb))
            if not dp_rq or not dp_rb:
                return None

            p_rq = dp_rq.price10 if dp_rq.invert_for_quote else dp_rq.price01
            p_rb = dp_rb.price10 if dp_rb.invert_for_quote else dp_rb.price01

            if not p_rq or not p_rb:
                return None

            price = p_rq / p_rb
            ts_ns = min(dp_rq.ts_ns, dp_rb.ts_ns)
            lat_ms = max(dp_rq.latency_ms or 0.0, dp_rb.latency_ms or 0.0)

            return {
                "price": price,
                "ts_ns": ts_ns,
                "ts_iso": _iso_from_ns(ts_ns),
                "lat_ms": lat_ms,
                "components": {
                    rq: {
                        "qb": p_rq,
                        "ts_ns": dp_rq.ts_ns,
                        "lat_ms": dp_rq.latency_ms
                    },
                    rb: {
                        "qb": p_rb,
                        "ts_ns": dp_rb.ts_ns,
                        "lat_ms": dp_rb.latency_ms
                    },
                }
            }


# ========================= helper builders =========================
def _dc_copy(obj):
    return dataclasses.replace(obj) if dataclasses.is_dataclass(obj) else deepcopy(obj)


def _enum(enum_cls, *candidates, default=None):
    for name in candidates:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    if default is not None:
        return default
    try:
        return list(enum_cls)[0]
    except Exception:
        return None


def make_client_simple(
    nats_url: str,
    use_jetstream: bool,
    cex_exchange: Optional[str] = None,
    cex_instruments: Optional[List[str]] = None,
    cex_symbols: Optional[List[str]] = None,
    dex_exchange: Optional[str] = None,
    dex_chain: Optional[str] = None,
    dex_pairs: Optional[List[str]] = None,
    csv_dir: str = "./csv",
    enable_csv: bool = True,
    durable_prefix: str = "mdc",
    stream: Optional[str] = "MD",
) -> MarketDataClient:
    cex_cfgs = []
    if cex_exchange and cex_symbols and cex_instruments:
        cex_cfgs.append(CexConfig(
            exchange=cex_exchange,
            symbols=[_norm_symbol(s) for s in cex_symbols],
            instruments=[i.lower() for i in cex_instruments],
            want=("tick", "funding", "fee", "volume"),
        ))
    
    dex_cfgs = []
    if dex_exchange and dex_chain and dex_pairs:
        dex_cfgs.append(DexConfig(
            exchange=dex_exchange,
            chain=dex_chain,
            pairs=dex_pairs,
            want=("tick", "slippage", "fee", "volume"),
        ))
    
    return MarketDataClient(
        nats_url=nats_url,
        use_jetstream=use_jetstream,
        stream=stream,
        durable_prefix=durable_prefix,
        csv_dir=csv_dir,
        enable_csv=enable_csv,
        cex=cex_cfgs,
        dex=dex_cfgs,
    )
