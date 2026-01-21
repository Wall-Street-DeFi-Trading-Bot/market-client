"""
Cross-Exchange Arbitrage Detector

This module detects arbitrage opportunities between different exchanges
by comparing prices and calculating potential profit after fees and slippage.

Arbitrage Strategy:
- Buy on exchange A (lower price) and sell on exchange B (higher price)
- Profit = (sell_price - buy_price) - fees - slippage
- Only shows opportunities where profit > minimum_threshold

Author: Market Data Client
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from market_data_client import CexConfig, DexConfig, MarketDataClient

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# Known token suffixes (longest first to avoid partial matches)
_KNOWN_SUFFIXES = (
    "BTCB",
    "WBNB",
    "WETH",
    "USDT",
    "USDC",
    "BUSD",
    "BNB",
    "ETH",
)

# Token alias map: treat wrapped/native variants as the same asset
_TOKEN_ALIAS = {
    "WBNB": "BNB",
    "BNB": "BNB",
    "WETH": "ETH",
    "ETH": "ETH",
    "BTCB": "BTC",
    "BTC": "BTC",
    "USDT": "USDT",
    "USDC": "USDC",
    "BUSD": "BUSD",
}


def _normalize_token(sym: str) -> str:
    """
    Normalize token symbol so that wrapped/native variants collapse
    to the same key (e.g. WBNB → BNB).
    """
    sym = sym.upper()
    return _TOKEN_ALIAS.get(sym, sym)


def _split_symbol_pair(sym: str) -> tuple[str, str]:
    """
    Roughly split a symbol into (base, quote) using known suffixes.

    Examples:
        "BNBUSDT"  -> ("BNB", "USDT")
        "USDTWBNB" -> ("USDT", "WBNB")
        "CAKEWBNB" -> ("CAKE", "WBNB")
        "USDTBTCB" -> ("USDT", "BTCB")
    """
    if not sym:
        return "", ""
    s = sym.upper()
    for suf in _KNOWN_SUFFIXES:
        if s.endswith(suf):
            base = s[: -len(suf)]
            quote = suf
            if not base:
                # Fallback – if we can't split, just return as-is
                return s, ""
            return base, quote
    # Fallback: unknown structure
    return s, ""


def _map_pair_price_to_canonical(
    canonical_symbol: str,
    pair_symbol: str,
    pair_price_qb: float,
) -> float:
    """
    Convert a DEX pair's quote/base price into the canonical symbol's quote/base
    price if they represent the same underlying assets.

    - canonical_symbol: e.g. "BNBUSDT"
    - pair_symbol: e.g. "USDTWBNB"
    - pair_price_qb: quote/base for *pair_symbol* (what DEX publishes)

    Returns:
        price_qb_canonical: quote/base for *canonical_symbol*.
    """
    if pair_price_qb <= 0.0:
        return 0.0

    c_base, c_quote = _split_symbol_pair(canonical_symbol)
    p_base, p_quote = _split_symbol_pair(pair_symbol)

    # Normalize to collapse aliases (BNB/WBNB, WETH/ETH, BTC/BTCB, …)
    cb, cq = _normalize_token(c_base), _normalize_token(c_quote)
    pb, pq = _normalize_token(p_base), _normalize_token(p_quote)

    # Case 1: same orientation: canonical (Base, Quote) == pair (Base, Quote)
    #   canonical price_qb = Quote/Base = pair_price_qb (already correct)
    if cb == pb and cq == pq:
        return pair_price_qb

    # Case 2: reversed orientation: canonical (Base, Quote) == pair (Quote, Base)
    #   canonical price_qb = Quote/Base = 1 / (pair Quote/Base)
    if cb == pq and cq == pb:
        return 1.0 / pair_price_qb

    # Anything else is suspicious; fall back and just log.
    logging.getLogger(__name__).warning(
        "Could not confidently map DEX pair %s (%s/%s) to canonical %s (%s/%s); "
        "using raw pair price.",
        pair_symbol,
        pb,
        pq,
        canonical_symbol,
        cb,
        cq,
    )
    return pair_price_qb

@dataclass
class ArbitrageOpportunity:
    """
    Represents a potential arbitrage opportunity between two exchanges.
    
    Attributes:
        symbol: Trading pair symbol (e.g., "ETHUSDT")
        buy_exchange: Exchange to buy from (lower price)
        sell_exchange: Exchange to sell to (higher price)
        buy_price: Price to buy at (ask price on buy_exchange)
        sell_price: Price to sell at (bid price on sell_exchange)
        buy_instrument: Instrument type for buy (spot/perpetual/swap)
        sell_instrument: Instrument type for sell (spot/perpetual/swap)
        price_diff: Raw price difference (sell_price - buy_price)
        price_diff_pct: Price difference as percentage
        buy_fee: Fee rate for buying (taker fee)
        sell_fee: Fee rate for selling (taker fee)
        buy_slippage: Slippage cost for buying (in bps, 0 for CEX)
        sell_slippage: Slippage cost for selling (in bps, 0 for CEX)
        net_profit_pct: Net profit percentage after all costs
        net_profit_per_unit: Net profit per unit of asset
        buy_latency_ms: Data latency for buy exchange
        sell_latency_ms: Data latency for sell exchange
        timestamp: When this opportunity was detected
    """
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    buy_instrument: str
    sell_instrument: str
    price_diff: float
    price_diff_pct: float
    buy_fee: float
    sell_fee: float
    buy_slippage: float
    sell_slippage: float
    net_profit_pct: float
    net_profit_per_unit: float
    buy_latency_ms: Optional[float]
    sell_latency_ms: Optional[float]
    timestamp: datetime

    def __str__(self) -> str:
        """Format opportunity as a readable string."""
        buy_lat = f"{self.buy_latency_ms:.1f}" if self.buy_latency_ms else "N/A"
        sell_lat = f"{self.sell_latency_ms:.1f}" if self.sell_latency_ms else "N/A"
        
        return (
            f"Arbitrage: {self.symbol}\n"
            f"  Buy  {self.buy_exchange} ({self.buy_instrument}) @ {self.buy_price:.4f} (latency: {buy_lat}ms)\n"
            f"  Sell {self.sell_exchange} ({self.sell_instrument}) @ {self.sell_price:.4f} (latency: {sell_lat}ms)\n"
            f"  Price diff: {self.price_diff:.4f} ({self.price_diff_pct:.2f}%)\n"
            f"  Fees: {self.buy_fee*100:.3f}% (buy) + {self.sell_fee*100:.3f}% (sell) = {(self.buy_fee + self.sell_fee)*100:.3f}% total\n"
            f"  Slippage: {self.buy_slippage:.2f}bps (buy) + {self.sell_slippage:.2f}bps (sell) = {self.buy_slippage + self.sell_slippage:.2f}bps total\n"
            f"  Net Profit: {self.net_profit_per_unit:.4f} per unit ({self.net_profit_pct:.2f}%)\n"
            f"  Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )


class ArbitrageDetector:
    """
    Detects arbitrage opportunities across multiple exchanges.
    
    This class monitors prices from different exchanges and calculates
    potential profit opportunities after accounting for fees and slippage.
    
    Note: DEX pairs may use different naming than CEX symbols.
    For example, "ETHUSDT" on CEX might be "USDTWBNB" or "WBNBETH" on DEX.
    Check CSV files or use the diagnostic output to see available pairs.
    """
    
    def __init__(
        self,
        client,
        min_profit_pct: float = 0.1,
        symbols: Optional[List[str]] = None,
        exchanges: Optional[List[Tuple[str, str]]] = None,
        symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None,
        fee_overrides: Optional[Dict[Tuple[str, str], float]] = None
    ):
        """
        Initialize the arbitrage detector.
        
        Args:
            client: MarketDataClient instance (already started)
            min_profit_pct: Minimum profit percentage to consider (default: 0.1%)
            symbols: List of symbols to monitor (e.g., ["ETHUSDT", "BTCUSDT"])
                     If None, uses symbols from client configuration
            exchanges: List of (exchange, instrument) tuples to compare
                      e.g., [("Binance", "spot"), ("PancakeSwapV2", "swap")]
                      If None, uses all available exchanges from client
            symbol_mapping: Optional mapping of CEX symbol to DEX pair names
                           e.g., {"ETHUSDT": {"PancakeSwapV2": "USDTWBNB"}}
                           If not provided, will try to auto-discover
        """
        self.client = client
        self.min_profit_pct = min_profit_pct
        self.symbols = symbols or []
        self.exchanges = exchanges or []
        self.symbol_mapping = symbol_mapping or {}
        
        # Track opportunities
        self.opportunities: List[ArbitrageOpportunity] = []
        self.opportunity_history: List[ArbitrageOpportunity] = []

        self.fee_overrides = fee_overrides or {
            ("Binance", "spot"): 0.0004,
            ("Binance", "perpetual"): 0.0004,
            ("PancakeSwapV2", "swap"): 0.0025,
            ("PancakeSwapV3", "swap"): 0.0001,
        }
    
    def get_available_dex_pairs(self, exchange: str, chain: str = "BSC") -> List[str]:
        """
        Get list of available DEX pairs from the client's cache.
        
        Args:
            exchange: DEX exchange name
            chain: Chain name (default: BSC)
            
        Returns:
            List of available pair names
        """
        available_pairs = []
        # Check the client's dex_pairs cache
        for (ex, ch, pair) in self.client.dex_pairs.keys():
            if ex == exchange and ch == chain:
                available_pairs.append(pair)
        return available_pairs
    
    async def get_price_data(
        self, 
        symbol: str, 
        exchange: str, 
        instrument: str
    ) -> Optional[Dict]:
        """
        Get price data for a symbol on a specific exchange.
        
        Args:
            symbol: Trading pair symbol (canonical symbol, e.g. "BNBUSDT")
            exchange: Exchange name
            instrument: Instrument type (spot/perpetual/swap)
            
        Returns:
            Dictionary with price data (canonical quote/base) or None if not available
        """
        logger.debug(f"Fetching price data: {symbol} on {exchange} ({instrument})")
        
        # DEX exchanges: use get_dex_price_qb and map pair price to canonical symbol
        if instrument == "swap" and exchange in ["PancakeSwapV2", "PancakeSwapV3", "UniswapV2", "UniswapV3"]:
            mapped_pair: Optional[str] = None
            using_mapping = False
            pair_used: Optional[str] = None
            dex_price: Optional[Dict] = None

            # 1) Try explicit symbol mapping first
            if symbol in self.symbol_mapping and exchange in self.symbol_mapping[symbol]:
                mapped_pair = self.symbol_mapping[symbol][exchange]
                logger.info(f"  Using symbol mapping: {symbol} -> {mapped_pair} on {exchange}")
                dex_price = await self.client.get_dex_price_qb(mapped_pair, exchange)
                using_mapping = True
                if dex_price:
                    pair_used = mapped_pair
            else:
                # 2) Try canonical symbol as-is
                logger.debug(f"  No mapping found for {symbol} on {exchange}, trying original symbol")
                dex_price = await self.client.get_dex_price_qb(symbol, exchange)
                if dex_price:
                    pair_used = symbol
            
            # 3) If no price yet and no mapping, try common DEX pair naming variations
            if not dex_price and not using_mapping:
                variations = [symbol]
                if symbol.startswith("ETH"):
                    variations.append(f"W{symbol}")   # ETHUSDT -> WETHUSDT
                    variations.append("WBNBETH")      # common BSC pair
                if symbol.startswith("BTC"):
                    variations.append(f"W{symbol}")   # BTCUSDT -> WBTCUSDT
                if "USDT" in symbol:
                    if symbol != "USDTWBNB":
                        variations.append("USDTWBNB")  # common BSC USDT/BNB pair
                
                for var in variations[1:]:  # first element is already tried (canonical)
                    logger.debug(f"  Trying DEX pair variation: {var} for {symbol}")
                    dp = await self.client.get_dex_price_qb(var, exchange)
                    if dp:
                        dex_price = dp
                        pair_used = var
                        logger.info(f"  ✓ Found price using variation: {var} (requested: {symbol})")
                        logger.info(f"     Tip: Add mapping: symbol_mapping={{'{symbol}': {{'{exchange}': '{var}'}}}}")
                        break
            
            # 4) If still no price, log available pairs and return None
            if not dex_price:
                available_pairs = self.get_available_dex_pairs(exchange)
                if using_mapping and mapped_pair:
                    logger.warning(
                        f"  ✗ {exchange} ({instrument}): No price data for mapped pair "
                        f"'{mapped_pair}' (original symbol: '{symbol}')"
                    )
                    logger.warning(
                        "     The symbol mapping may be incorrect, or the pair "
                        f"'{mapped_pair}' is not being published"
                    )
                else:
                    logger.warning(f"  ✗ {exchange} ({instrument}): No price data for '{symbol}'")
                
                if available_pairs:
                    logger.info(f"     Available pairs on {exchange}: {', '.join(available_pairs)}")
                    if using_mapping and mapped_pair and mapped_pair not in available_pairs:
                        logger.warning(f"     ⚠️  Mapped pair '{mapped_pair}' is not in available pairs!")
                        logger.info("     Tip: Check your symbol_mapping configuration or verify the pair name in CSV files")
                    else:
                        logger.info("     Tip: Use one of the available pairs, or check CSV files for actual pair names")
                else:
                    logger.warning("     No DEX pairs found in cache. Check if market data publisher is running")
                    logger.warning("     and publishing DEX data, or check CSV files in ./csv/ directory")
                return None

            # 5) We have a DEX price: map from DEX pair quote/base → canonical quote/base
            pair_symbol = pair_used or mapped_pair or symbol
            raw_price_qb = dex_price["price_qb"]
            price_qb_canonical = _map_pair_price_to_canonical(
                canonical_symbol=symbol,
                pair_symbol=pair_symbol,
                pair_price_qb=raw_price_qb,
            )

            lat_str = f"{dex_price.get('lat_ms', 0):.1f}ms" if dex_price.get('lat_ms') else "N/A"
            logger.info(
                f"  ✓ {exchange} ({instrument}): "
                f"Price(canonical {symbol})={price_qb_canonical:.4f}, "
                f"pair={pair_symbol}, raw_pair_price={raw_price_qb:.6f}, "
                f"Latency={lat_str}"
            )

            lat_ms = (
                dex_price.get("lat_ms")
                or dex_price.get("lat_source_ms")
                or dex_price.get("lat_publish_ms")
            )
            # Return canonical quote/base price
            return {
                "mid": price_qb_canonical,
                "bid": price_qb_canonical,   # Treat DEX as a single executable price
                "ask": price_qb_canonical,
                "lat_ms": lat_ms,
                "ts_iso": dex_price.get("ts_iso"),
                "instrument": "swap",
                "exchange": exchange,
                "symbol": symbol,           # keep canonical symbol here
            }
        
        # CEX exchanges: use standard latest price API
        price_data = await self.client.get_latest_price_with_latency(
            symbol, exchange, instrument
        )
        if price_data:
            lat_val = (
                price_data.get("lat_ms")
                or price_data.get("lat_source_ms")
                or price_data.get("lat_publish_ms")
            )
            lat_str = f"{lat_val:.1f}ms" if lat_val else "N/A"
            logger.info(
                f"  ✓ {exchange} ({instrument}): "
                f"Bid={price_data['bid']:.4f}, Ask={price_data['ask']:.4f}, "
                f"Mid={price_data['mid']:.4f}, Latency={lat_str}"
            )
        else:
            logger.warning(f"  ✗ {exchange} ({instrument}): No price data available")
        return price_data
    
    async def get_fee_rate(self, symbol: str, exchange: str, instrument: str) -> float:
        """
        Get taker fee rate.
        Returns a fraction (e.g. 0.001 for 0.1%).
        """
        symbol_for_fee = symbol

        # For DEX, query meta by mapped pair when present
        if instrument == "swap" and exchange in ["PancakeSwapV2", "PancakeSwapV3", "UniswapV2", "UniswapV3"]:
            if symbol in self.symbol_mapping and exchange in self.symbol_mapping[symbol]:
                symbol_for_fee = self.symbol_mapping[symbol][exchange]

        fee_meta = await self.client.get_fee_meta(symbol_for_fee, exchange, instrument)
        if fee_meta:
            return float(fee_meta.taker_rate)

        # override fallback
        override = self.fee_overrides.get((exchange, instrument))
        if override is not None:
            logger.warning(
                "  Fee %s (%s) for %s: Using override %.4f%% (meta not available)",
                exchange,
                instrument,
                symbol_for_fee,
                override * 100,
            )
            return float(override)

        # last resort
        default_fee = 0.001
        logger.warning(
            "  Fee %s (%s) for %s: Using default %.3f%% (meta not available)",
            exchange,
            instrument,
            symbol_for_fee,
            default_fee * 100,
        )
        return default_fee

    
    async def get_slippage_bps(
        self,
        symbol: str,
        exchange: str,
        trade_size_pct: float = 1.0,
    ) -> float:
        """
        Get slippage in basis points for a trade.

        - For CEX: return 0 bps (slippage 거의 무시)
        - For DEX: use slippage_meta(impact_bps01, impact_bps10) + trade_size_pct 로 보간
        """
        # CEX: 그냥 0으로 가정
        if exchange not in ["PancakeSwapV2", "PancakeSwapV3", "UniswapV2", "UniswapV3"]:
            logger.debug(f"  Slippage {exchange}: 0bps (CEX)")
            return 0.0

        # --- DEX: symbol_mapping 적용해서 실제 pair 이름으로 조회 ---
        dex_symbol = symbol
        if symbol in self.symbol_mapping and exchange in self.symbol_mapping[symbol]:
            dex_symbol = self.symbol_mapping[symbol][exchange]
            logger.debug(
                "  Slippage mapping: %s -> %s on %s",
                symbol,
                dex_symbol,
                exchange,
            )

        slippage_meta = await self.client.get_slippage_meta(dex_symbol, exchange)

        if not slippage_meta:
            logger.warning(
                "  Slippage %s: No slippage_meta for %s, assuming 0bps",
                exchange,
                dex_symbol,
            )
            return 0.0

        # trade_size_pct: 0.01 = 1%, 0.10 = 10%
        if trade_size_pct <= 0.01:
            slippage = slippage_meta.impact_bps01
        elif trade_size_pct >= 0.10:
            slippage = slippage_meta.impact_bps10
        else:
            ratio = (trade_size_pct - 0.01) / 0.09
            slippage = slippage_meta.impact_bps01 + ratio * (
                slippage_meta.impact_bps10 - slippage_meta.impact_bps01
            )

        logger.debug(
            "  Slippage %s: %.2f bps (1%%=%.2f, 10%%=%.2f, trade_size=%.2f%%)",
            exchange,
            slippage,
            slippage_meta.impact_bps01,
            slippage_meta.impact_bps10,
            trade_size_pct * 100,
        )
        return slippage

    
    def calculate_arbitrage(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price_data: Dict,
        sell_price_data: Dict,
        buy_instrument: str,
        sell_instrument: str,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Calculate arbitrage opportunity between two exchanges.
        
        Args:
            symbol: Trading pair symbol
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell to
            buy_price_data: Price data from buy exchange
            sell_price_data: Price data from sell exchange
            buy_instrument: Instrument type for buy
            sell_instrument: Instrument type for sell
            
        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        # Get ask price (buy price) and bid price (sell price)
        buy_price = buy_price_data["ask"]  # We buy at ask price
        sell_price = sell_price_data["bid"]  # We sell at bid price
        
        # Calculate raw price difference
        price_diff = sell_price - buy_price
        price_diff_pct = (price_diff / buy_price) * 100 if buy_price > 0 else 0
        
        # If sell price is not higher, no arbitrage
        if price_diff <= 0:
            return None
        
        # Start with zero; real 수수료/슬리피지는 후단에서 덮어씀
        buy_fee = 0.0
        sell_fee = 0.0
        
        # Gross profit percentage (before fees and slippage)
        gross_profit_pct = price_diff_pct
        
        # Default slippage assumptions (will be overwritten later with real values)
        buy_slippage = 0.0
        sell_slippage = 0.0
        
        # Net profit = gross profit - fees - slippage
        total_fee_pct = (buy_fee + sell_fee) * 100
        total_slippage_bps = buy_slippage + sell_slippage
        total_slippage_pct = total_slippage_bps / 100
        
        net_profit_pct = gross_profit_pct - total_fee_pct - total_slippage_pct
        net_profit_per_unit = buy_price * (net_profit_pct / 100)
        
        return ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_instrument=buy_instrument,
            sell_instrument=sell_instrument,
            price_diff=price_diff,
            price_diff_pct=price_diff_pct,
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            buy_slippage=buy_slippage,
            sell_slippage=sell_slippage,
            net_profit_pct=net_profit_pct,
            net_profit_per_unit=net_profit_per_unit,
            buy_latency_ms=buy_price_data.get("lat_ms"),
            sell_latency_ms=sell_price_data.get("lat_ms"),
            timestamp=datetime.now(),
        )
    
    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities across all configured exchanges.
        
        This method compares prices between all exchange pairs and
        identifies profitable arbitrage opportunities.
        
        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities: List[ArbitrageOpportunity] = []
        
        # Default symbols if not configured
        if not self.symbols:
            self.symbols = ["ETHUSDT", "BTCUSDT"]
        
        # Default exchanges if not configured
        if not self.exchanges:
            self.exchanges = [
                ("Binance", "spot"),
                ("Binance", "perpetual"),
                ("PancakeSwapV2", "swap"),
                ("PancakeSwapV3", "swap"),
            ]
        
        logger.info("Scanning for arbitrage opportunities:")
        logger.info(f"  Symbols: {', '.join(self.symbols)}")
        logger.info(f"  Exchanges: {', '.join([f'{ex}({inst})' for ex, inst in self.exchanges])}")
        logger.info(f"  Minimum profit threshold: {self.min_profit_pct}%")
        
        # Compare all exchange pairs symbol by symbol
        for symbol in self.symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Analyzing {symbol}")
            logger.info(f"{'='*80}")
            
            # Get prices from all exchanges
            prices: Dict[Tuple[str, str], Dict] = {}
            logger.info("\nFetching prices from all exchanges:")
            for exchange, instrument in self.exchanges:
                price_data = await self.get_price_data(symbol, exchange, instrument)
                if price_data:
                    prices[(exchange, instrument)] = price_data
            
            # Display all prices found
            if prices:
                logger.info(f"\n📊 Current Prices for {symbol}:")
                logger.info(f"{'Exchange':<20} {'Instrument':<12} {'Bid':<12} {'Ask':<12} {'Mid':<12} {'Latency':<10}")
                logger.info(f"{'-'*80}")
                for (ex, inst), data in prices.items():
                    bid = data.get('bid', data.get('mid', 0))
                    ask = data.get('ask', data.get('mid', 0))
                    mid = data.get('mid', 0)
                    lat = f"{data.get('lat_ms', 0):.1f}ms" if data.get('lat_ms') else "N/A"
                    logger.info(f"{ex:<20} {inst:<12} {bid:<12.4f} {ask:<12.4f} {mid:<12.4f} {lat:<10}")
            else:
                logger.warning(f"  ⚠️  No price data available for {symbol} on any exchange")
                continue
            
            # Get fees and slippage for all exchanges
            logger.info(f"\n💰 Fees and Slippage for {symbol}:")
            logger.info(f"{'Exchange':<20} {'Instrument':<12} {'Taker Fee':<12} {'Slippage':<12}")
            logger.info(f"{'-'*60}")
            exchange_costs: Dict[Tuple[str, str], Dict[str, float]] = {}
            for exchange, instrument in self.exchanges:
                if (exchange, instrument) in prices:
                    fee = await self.get_fee_rate(symbol, exchange, instrument)
                    slippage = await self.get_slippage_bps(symbol, exchange, trade_size_pct=0.01)
                    exchange_costs[(exchange, instrument)] = {'fee': fee, 'slippage': slippage}
                    logger.info(f"{exchange:<20} {instrument:<12} {fee*100:<11.3f}% {slippage:<11.6f}bps")
            
            # Compare all exchange pairs for this symbol
            logger.info("\n🔍 Comparing all exchange pairs:")
            comparison_count = 0
            for (buy_ex, buy_inst), buy_data in prices.items():
                for (sell_ex, sell_inst), sell_data in prices.items():
                    # Skip if same exchange + instrument
                    if buy_ex == sell_ex and buy_inst == sell_inst:
                        continue
                    
                    comparison_count += 1
                    buy_price = buy_data["ask"]
                    sell_price = sell_data["bid"]
                    price_diff = sell_price - buy_price
                    price_diff_pct = (price_diff / buy_price) * 100 if buy_price > 0 else 0
                    
                    logger.debug(
                        f"  Comparing: {buy_ex}({buy_inst}) @ {buy_price:.4f} "
                        f"vs {sell_ex}({sell_inst}) @ {sell_price:.4f} "
                        f"(diff: {price_diff_pct:.2f}%)"
                    )
                    
                    # First, compute a quick arbitrage with default fees/slippage
                    opp = self.calculate_arbitrage(
                        symbol=symbol,
                        buy_exchange=buy_ex,
                        sell_exchange=sell_ex,
                        buy_price_data=buy_data,
                        sell_price_data=sell_data,
                        buy_instrument=buy_inst,
                        sell_instrument=sell_inst,
                    )
                    
                    if opp:
                        # Overwrite with actual fees
                        opp.buy_fee = exchange_costs.get(
                            (buy_ex, buy_inst),
                            {}
                        ).get('fee', await self.get_fee_rate(symbol, buy_ex, buy_inst))
                        opp.sell_fee = exchange_costs.get(
                            (sell_ex, sell_inst),
                            {}
                        ).get('fee', await self.get_fee_rate(symbol, sell_ex, sell_inst))
                        
                        # Overwrite with actual slippage
                        opp.buy_slippage = exchange_costs.get((buy_ex, buy_inst), {}).get(
                            'slippage',
                            await self.get_slippage_bps(symbol, buy_ex, trade_size_pct=0.01),
                        )

                        opp.sell_slippage = exchange_costs.get((sell_ex, sell_inst), {}).get(
                            'slippage',
                            await self.get_slippage_bps(symbol, sell_ex, trade_size_pct=0.01),
                        )

                        
                        # Recalculate net profit with actual costs
                        total_fee_pct = (opp.buy_fee + opp.sell_fee) * 100
                        total_slippage_pct = (opp.buy_slippage + opp.sell_slippage) / 100
                        opp.net_profit_pct = opp.price_diff_pct - total_fee_pct - total_slippage_pct
                        opp.net_profit_per_unit = opp.buy_price * (opp.net_profit_pct / 100)
                        
                        logger.info(
                            f"  ✓ Potential opportunity: {buy_ex}({buy_inst}) → "
                            f"{sell_ex}({sell_inst}): Gross={price_diff_pct:.2f}%, "
                            f"Net={opp.net_profit_pct:.2f}%"
                        )
                        
                        # Only keep if still above threshold after real costs
                        if opp.net_profit_pct >= self.min_profit_pct:
                            opportunities.append(opp)
                            logger.info(
                                f"    ✅ PROFITABLE! Net profit: {opp.net_profit_pct:.2f}% "
                                f"({opp.net_profit_per_unit:.4f} per unit)"
                            )
                        else:
                            logger.debug(
                                f"    ❌ Below threshold ({opp.net_profit_pct:.2f}% "
                                f"< {self.min_profit_pct}%)"
                            )
            
            logger.info(f"\n  Total comparisons: {comparison_count}")
        
        self.opportunities = opportunities
        self.opportunity_history.extend(opportunities)
        
        return opportunities
    
    def print_opportunities(self, opportunities: Optional[List[ArbitrageOpportunity]] = None):
        """
        Print arbitrage opportunities in a formatted way.
        
        Args:
            opportunities: List of opportunities to print. If None, uses self.opportunities
        """
        opps = opportunities or self.opportunities
        
        if not opps:
            logger.info("No arbitrage opportunities found.")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Found {len(opps)} Arbitrage Opportunity(ies)")
        logger.info(f"{'='*80}\n")
        
        # Sort by profit percentage (highest first)
        opps_sorted = sorted(opps, key=lambda x: x.net_profit_pct, reverse=True)
        
        for i, opp in enumerate(opps_sorted, 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"Opportunity #{i}:")
            logger.info(f"{'─'*80}")
            
            logger.info(f"\n Symbol: {opp.symbol}")
            logger.info("\n Prices:")
            buy_lat_str = f" (latency: {opp.buy_latency_ms:.1f}ms)" if opp.buy_latency_ms else ""
            sell_lat_str = f" (latency: {opp.sell_latency_ms:.1f}ms)" if opp.sell_latency_ms else ""
            logger.info(f"  Buy:  {opp.buy_exchange:20} ({opp.buy_instrument:12}) @ {opp.buy_price:.4f}{buy_lat_str}")
            logger.info(f"  Sell: {opp.sell_exchange:20} ({opp.sell_instrument:12}) @ {opp.sell_price:.4f}{sell_lat_str}")
            
            logger.info("\n Price Analysis:")
            logger.info(f"  Raw price difference: {opp.price_diff:.4f} ({opp.price_diff_pct:.2f}%)")
            
            logger.info("\n Costs:")
            logger.info(f"  Buy fee:  {opp.buy_fee*100:.3f}% ({opp.buy_fee*opp.buy_price:.4f} per unit)")
            logger.info(f"  Sell fee: {opp.sell_fee*100:.3f}% ({opp.sell_fee*opp.sell_price:.4f} per unit)")
            logger.info(
                f"  Total fees: {(opp.buy_fee + opp.sell_fee)*100:.3f}% "
                f"({(opp.buy_fee*opp.buy_price + opp.sell_fee*opp.sell_price):.4f} per unit)"
            )
            
            logger.info("\n Slippage:")
            logger.info(f"  Buy slippage:  {opp.buy_slippage:.2f} bps ({opp.buy_slippage/100*opp.buy_price:.4f} per unit)")
            logger.info(f"  Sell slippage: {opp.sell_slippage:.2f} bps ({opp.sell_slippage/100*opp.sell_price:.4f} per unit)")
            logger.info(
                f"  Total slippage: {opp.buy_slippage + opp.sell_slippage:.2f} bps "
                f"({(opp.buy_slippage/100*opp.buy_price + opp.sell_slippage/100*opp.sell_price):.4f} per unit)"
            )
            
            total_costs_pct = (opp.buy_fee + opp.sell_fee)*100 + (opp.buy_slippage + opp.sell_slippage)/100
            logger.info(f"\n Total Costs: {total_costs_pct:.3f}%")
            
            logger.info("\n✅ Net Profit:")
            logger.info(f"  Per unit: {opp.net_profit_per_unit:.4f}")
            logger.info(f"  Percentage: {opp.net_profit_pct:.2f}%")
            logger.info(f"  Timestamp: {opp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            logger.info(f"\n{'─'*80}\n")

async def run_arbitrage_detector(
    symbols: List[str] = None,
    exchanges: List[Tuple[str, str]] = None,
    min_profit_pct: float = 0.1,
    scan_interval: float = 5.0,
    nats_url: str = "nats://127.0.0.1:4222",
    symbol_mapping: Optional[Dict[str, Dict[str, str]]] = None,
):
    """
    Run the arbitrage detector continuously.
    
    This is the main entry point for running arbitrage detection.
    It sets up the market data client, starts monitoring, and scans
    for opportunities at regular intervals.
    """
    # Default symbols
    if symbols is None:
        symbols = ["ETHUSDT"]
    
    # Default exchanges - all supported combinations
    if exchanges is None:
        exchanges = [
            ("Binance", "spot"),
            ("Binance", "perpetual"),
            ("PancakeSwapV2", "swap"),
            ("PancakeSwapV3", "swap"),
        ]
    
    cex_configs: List[CexConfig] = []
    dex_configs: List[DexConfig] = []
    
    # Group CEX exchanges by exchange name
    cex_by_exchange: Dict[str, Dict[str, set]] = {}
    for exchange, instrument in exchanges:
        if instrument != "swap":
            if exchange not in cex_by_exchange:
                cex_by_exchange[exchange] = {"instruments": set(), "symbols": set(symbols)}
            cex_by_exchange[exchange]["instruments"].add(instrument)
    
    # Create CEX configs
    for exchange, data in cex_by_exchange.items():
        cex_configs.append(CexConfig(
            exchange=exchange,
            symbols=list(data["symbols"]),
            instruments=list(data["instruments"]),
            want=("tick", "funding", "fee", "volume"),
        ))
        logger.info(f"  CEX Config: {exchange} - symbols: {data['symbols']}, instruments: {data['instruments']}")
    
    # Group DEX exchanges, apply symbol_mapping to get actual pair names
    dex_by_exchange: Dict[Tuple[str, str], Dict[str, set]] = {}
    for exchange, instrument in exchanges:
        if instrument == "swap":
            key = (exchange, "BSC")  # default chain
            if key not in dex_by_exchange:
                dex_by_exchange[key] = {"pairs": set()}
            
            for symbol in symbols:
                if symbol_mapping and symbol in symbol_mapping and exchange in symbol_mapping[symbol]:
                    mapped_pair = symbol_mapping[symbol][exchange]
                    dex_by_exchange[key]["pairs"].add(mapped_pair)
                    logger.debug(f"  Symbol mapping: {symbol} -> {mapped_pair} for {exchange}")
                else:
                    dex_by_exchange[key]["pairs"].add(symbol)
    
    # Create DEX configs
    for (exchange, chain), data in dex_by_exchange.items():
        dex_configs.append(DexConfig(
            exchange=exchange,
            chain=chain,
            pairs=list(data["pairs"]),
            want=("tick", "slippage", "fee", "volume"),
        ))
        logger.info(f"  DEX Config: {exchange} ({chain}) - pairs: {data['pairs']}")
    
    # Create market data client
    client = MarketDataClient(
        nats_url=nats_url,
        use_jetstream=False,
        cex=cex_configs if cex_configs else None,
        dex=dex_configs if dex_configs else None,
        enable_csv=True,
    )
    
    # Start client
    await client.start()
    logger.info("="*80)
    logger.info("✅ Market data client started")
    logger.info("="*80)
    logger.info(f"   Monitoring symbols: {', '.join(symbols)}")
    logger.info(f"   Exchanges: {', '.join([f'{ex}({inst})' for ex, inst in exchanges])}")
    logger.info(f"   Total exchange pairs: {len(exchanges)}")
    logger.info(f"   Total comparisons per symbol: {len(exchanges) * (len(exchanges) - 1)}")
    logger.info(f"   Minimum profit threshold: {min_profit_pct}%")
    logger.info(f"   Scan interval: {scan_interval}s")
    logger.info("="*80)
    
    # Log subscription details
    logger.info("\n📡 Subscriptions:")
    if cex_configs:
        for cfg in cex_configs:
            logger.info(f"   CEX: {cfg.exchange} - {len(cfg.symbols)} symbols, {len(cfg.instruments)} instruments")
    if dex_configs:
        for cfg in dex_configs:
            logger.info(f"   DEX: {cfg.exchange} ({cfg.chain}) - {len(cfg.pairs)} pairs")
    
    # Wait for initial data
    logger.info("\n⏳ Waiting for market data to arrive...")
    await asyncio.sleep(5)
    
    # Diagnostic: Check what DEX data is actually available
    if dex_configs:
        logger.info("\n🔍 Checking available DEX price data:")
        for cfg in dex_configs:
            available_pairs = []
            for (ex, ch, pair) in client.dex_pairs.keys():
                if ex == cfg.exchange and ch == cfg.chain:
                    available_pairs.append(pair)
            
            if available_pairs:
                logger.info(f"   Available pairs on {cfg.exchange} ({cfg.chain}): {', '.join(available_pairs)}")
            else:
                logger.warning(f"   No pairs found in cache for {cfg.exchange} ({cfg.chain})")
                logger.info("      Check CSV files (./csv/swaps.csv) to see what pairs are being published")
            
            # Try to get prices for requested pairs
            for pair in cfg.pairs:
                price = await client.get_dex_price_qb(pair, cfg.exchange, cfg.chain)
                if price:
                    logger.info(f"   ✓ {cfg.exchange}: {pair} = {price['price_qb']:.4f}")
                else:
                    logger.warning(f"   ✗ {cfg.exchange}: {pair} - No data")
                    if available_pairs:
                        logger.info(f"      Available pairs: {', '.join(available_pairs)}")
                        logger.info(f"      Tip: Use one of the available pairs instead of '{pair}'")
                    else:
                        logger.info("      Check ./csv/swaps.csv to see what pairs are actually published")
    
    logger.info("\n✅ Ready to scan for opportunities\n")
    
    # Create detector
    detector = ArbitrageDetector(
        client=client,
        min_profit_pct=min_profit_pct,
        symbols=symbols,
        exchanges=exchanges,
        symbol_mapping=symbol_mapping or {},
    )
    
    try:
        scan_count = 0
        while True:
            scan_count += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 Scan #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            opportunities = await detector.scan_opportunities()
            
            if opportunities:
                detector.print_opportunities(opportunities)
            else:
                logger.info("\n  ❌ No profitable opportunities found above threshold.")
            
            logger.info(f"\n⏳ Next scan in {scan_interval} seconds...")
            await asyncio.sleep(scan_interval)
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Stopping arbitrage detector...")
    finally:
        await client.stop()
        logger.info("✅ Market data client stopped")
        
        # Print summary
        if detector.opportunity_history:
            logger.info(f"\n{'='*80}")
            logger.info("📊 Summary")
            logger.info(f"{'='*80}")
            logger.info(f"  Total opportunities found: {len(detector.opportunity_history)}")
            best = max(detector.opportunity_history, key=lambda x: x.net_profit_pct)
            logger.info(f"  Best opportunity: {best.net_profit_pct:.2f}% profit on {best.symbol}")
            logger.info(f"    Buy: {best.buy_exchange} ({best.buy_instrument}) @ {best.buy_price:.4f}")
            logger.info(f"    Sell: {best.sell_exchange} ({best.sell_instrument}) @ {best.sell_price:.4f}")
            logger.info(f"{'='*80}")


async def main():
    """
    Main entry point for the arbitrage detector.
    """
    SYMBOLS = ["ETHUSDT"]
    EXCHANGES = [
        ("Binance", "spot"),
        ("Binance", "perpetual"),
        ("PancakeSwapV2", "swap"),
        ("PancakeSwapV3", "swap"),
    ]
    MIN_PROFIT_PCT = 0.1  # Minimum 0.1% profit to report
    SCAN_INTERVAL = 5.0   # Scan every 5 seconds
    
    await run_arbitrage_detector(
        symbols=SYMBOLS,
        exchanges=EXCHANGES,
        min_profit_pct=MIN_PROFIT_PCT,
        scan_interval=SCAN_INTERVAL,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
