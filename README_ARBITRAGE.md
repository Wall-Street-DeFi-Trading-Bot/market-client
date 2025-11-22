# Arbitrage Detector Documentation

## Overview

The Arbitrage Detector is a tool that identifies cross-exchange arbitrage opportunities by comparing prices across multiple exchanges (both CEX and DEX) and calculating potential profit after accounting for fees and slippage.

### What is Arbitrage?

Arbitrage is the practice of buying an asset on one exchange at a lower price and simultaneously selling it on another exchange at a higher price to profit from the price difference.

**Example:**

- Buy ETHUSDT on Binance spot at $2,500
- Sell ETHUSDT on PancakeSwapV2 at $2,505
- Gross profit: $5 per ETH
- After fees (0.1% each side) and slippage: Net profit ~$0.50 per ETH

## Features

- ✅ **Multi-Exchange Support**: Compare prices across CEX (Binance, etc.) and DEX (PancakeSwapV2/V3, UniswapV2/V3)
- ✅ **Real-Time Monitoring**: Continuously scans for opportunities at configurable intervals
- ✅ **Cost-Aware**: Accounts for trading fees and slippage in profit calculations
- ✅ **Symbol Mapping**: Map CEX symbols to DEX pair names (e.g., "BNBUSDT" → "USDTWBNB")
- ✅ **Detailed Logging**: Comprehensive logs showing prices, fees, slippage, and profit calculations
- ✅ **Configurable Thresholds**: Set minimum profit percentage to filter opportunities

## Installation

The arbitrage detector is part of the `market-client` package. Ensure you have:

1. Python 3.7+
2. Dependencies installed: `pip install -r requirements.txt`
3. NATS server running (for market data)
4. Market data publisher running and publishing data to NATS

## Quick Start

### Basic Usage

```python
import asyncio
from arbitrage_detector import run_arbitrage_detector

async def main():
    await run_arbitrage_detector(
        symbols=["ETHUSDT"],
        exchanges=[
            ("Binance", "spot"),
            ("PancakeSwapV2", "swap"),
        ],
        min_profit_pct=0.1,  # 0.1% minimum profit
        scan_interval=5.0,   # Scan every 5 seconds
    )

asyncio.run(main())
```

### Using the Example Script

The easiest way to get started is using `arbitrage_example.py`:

```bash
python arbitrage_example.py
```

Edit the configuration in `arbitrage_example.py` to customize:

- Symbols to monitor
- Exchanges to compare
- Minimum profit threshold
- Scan interval

## Configuration

### Symbols

List of trading pairs to monitor:

```python
SYMBOLS = ["ETHUSDT", "BTCUSDT", "BNBUSDT"]
```

### Exchanges

List of (exchange, instrument) tuples:

```python
EXCHANGES = [
    ("Binance", "spot"),        # Binance spot market
    ("Binance", "perpetual"),   # Binance perpetual futures
    ("PancakeSwapV2", "swap"),  # PancakeSwap V2 DEX
    ("PancakeSwapV3", "swap"),  # PancakeSwap V3 DEX
]
```

**Supported Exchanges:**

- **CEX**: Binance (spot, perpetual)
- **DEX**: PancakeSwapV2, PancakeSwapV3, UniswapV2, UniswapV3

**Instrument Types:**

- `spot`: Spot trading
- `perpetual`: Perpetual futures
- `swap`: DEX swap pools

### Symbol Mapping

DEX pairs often use different naming conventions than CEX symbols. Use `SYMBOL_MAPPING` to map CEX symbols to DEX pair names:

```python
SYMBOL_MAPPING = {
    "BNBUSDT": {
        "PancakeSwapV2": "USDTWBNB",  # BNBUSDT on CEX = USDTWBNB on DEX
        "PancakeSwapV3": "USDTWBNB",
    },
    "ETHUSDT": {
        "PancakeSwapV2": "WBNBETH",   # Example: ETHUSDT = WBNBETH pair
        "PancakeSwapV3": "WBNBETH",
    },
}
```

**Why is this needed?**

- CEX uses symbols like "BNBUSDT", "ETHUSDT"
- DEX uses pair names like "USDTWBNB" (Wrapped BNB), "WBNBETH"
- The mapping tells the detector which DEX pair corresponds to which CEX symbol

**How to find the correct DEX pair names:**

1. Check CSV files: `./csv/swaps.csv` shows what pairs are being published
2. Check the diagnostic output when running the detector
3. Look at `strategy_live_example.py` for examples

### Minimum Profit Threshold

Set the minimum net profit percentage required to report an opportunity:

```python
MIN_PROFIT_PCT = 0.1  # 0.1% minimum profit
```

**Recommended Values:**

- `0.05` (0.05%): Very sensitive, more opportunities (may include unprofitable ones)
- `0.1` (0.1%): Default, balanced
- `0.2` (0.2%): Conservative, fewer but higher-quality opportunities
- `0.5` (0.5%): Very conservative, only high-profit opportunities

### Scan Interval

How often to scan for opportunities (in seconds):

```python
SCAN_INTERVAL = 5.0  # Scan every 5 seconds
```

**Recommendations:**

- `1.0-3.0`: Fast scanning, more CPU usage
- `5.0-10.0`: Balanced (recommended)
- `30.0+`: Slow scanning, less CPU usage

## How It Works

### 1. Price Comparison

For each symbol, the detector:

1. Fetches current prices from all configured exchanges
2. Compares bid/ask prices between all exchange pairs
3. Identifies where `sell_price > buy_price`

### 2. Cost Calculation

For each potential opportunity, it calculates:

**Gross Profit:**

```
gross_profit = sell_price - buy_price
gross_profit_pct = (gross_profit / buy_price) * 100
```

**Fees:**

- Fetches taker fees from market data
- Default: 0.1% if data unavailable
- Applied to both buy and sell sides

**Slippage:**

- **DEX**: Uses real slippage data (`bps01` and `bps10`) with interpolation
- **CEX**: Assumes 0 bps (minimal slippage for market orders)
- Measured in basis points (1 bps = 0.01%)

**Net Profit:**

```
net_profit_pct = gross_profit_pct - total_fees_pct - total_slippage_pct
```

### 3. Filtering

Only opportunities where `net_profit_pct >= min_profit_pct` are reported.

## Output Format

### Opportunity Details

Each opportunity shows:

```
📈 Symbol: ETHUSDT

💵 Prices:
  Buy:  Binance            (spot       ) @ 2500.0000 (latency: 12.5ms)
  Sell: PancakeSwapV2      (swap       ) @ 2505.0000 (latency: 15.2ms)

📊 Price Analysis:
  Raw price difference: 5.0000 (0.20%)

💰 Costs:
  Buy fee:  0.100% (2.5000 per unit)
  Sell fee: 0.100% (2.5050 per unit)
  Total fees: 0.200% (5.0050 per unit)

📉 Slippage:
  Buy slippage:  0.00 bps (0.0000 per unit)
  Sell slippage: 2.20 bps (0.0551 per unit)
  Total slippage: 2.20 bps (0.0551 per unit)

💎 Net Profit:
  Net profit: 4.9399 per unit (0.20%)
```

### Log Levels

- **INFO**: Opportunities found, prices, fees, profit calculations
- **DEBUG**: Detailed price fetching, mapping usage, comparisons
- **WARNING**: Missing data, mapping issues, unavailable pairs

## API Reference

### `run_arbitrage_detector()`

Main entry point for running the arbitrage detector.

**Parameters:**

- `symbols` (List[str]): Trading pairs to monitor (default: `["ETHUSDT"]`)
- `exchanges` (List[Tuple[str, str]]): List of (exchange, instrument) tuples
- `min_profit_pct` (float): Minimum profit percentage (default: `0.1`)
- `scan_interval` (float): Scan interval in seconds (default: `5.0`)
- `nats_url` (str): NATS server URL (default: `"nats://127.0.0.1:4222"`)
- `symbol_mapping` (Dict): Optional symbol mapping (default: `None`)

**Example:**

```python
await run_arbitrage_detector(
    symbols=["ETHUSDT", "BTCUSDT"],
    exchanges=[
        ("Binance", "spot"),
        ("PancakeSwapV2", "swap"),
    ],
    min_profit_pct=0.2,
    scan_interval=3.0,
    symbol_mapping={
        "ETHUSDT": {"PancakeSwapV2": "WBNBETH"}
    }
)
```

### `ArbitrageDetector` Class

For programmatic use:

```python
from arbitrage_detector import ArbitrageDetector

detector = ArbitrageDetector(
    client=market_data_client,
    min_profit_pct=0.1,
    symbols=["ETHUSDT"],
    exchanges=[("Binance", "spot"), ("PancakeSwapV2", "swap")],
    symbol_mapping={"ETHUSDT": {"PancakeSwapV2": "WBNBETH"}}
)

# Scan once
opportunities = await detector.scan_opportunities()

# Print opportunities
detector.print_opportunities(opportunities)
```

**Methods:**

- `scan_opportunities()`: Scan for opportunities once, returns list of `ArbitrageOpportunity`
- `print_opportunities(opportunities)`: Print formatted opportunity details
- `get_price_data(symbol, exchange, instrument)`: Get price data for a symbol
- `get_fee_rate(symbol, exchange, instrument)`: Get fee rate
- `get_slippage_bps(symbol, exchange, trade_size_pct)`: Get slippage in bps

### `ArbitrageOpportunity` Dataclass

Represents a single arbitrage opportunity:

```python
@dataclass
class ArbitrageOpportunity:
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
```

## Troubleshooting

### No Price Data for DEX

**Problem:** `✗ PancakeSwapV2 (swap): No price data available for 'ETHUSDT'`

**Solutions:**

1. Check if the pair exists: Look at `./csv/swaps.csv` to see available pairs
2. Use symbol mapping: Map CEX symbol to correct DEX pair name
3. Verify market data publisher is running and publishing DEX data
4. Check NATS connection: Ensure NATS server is running

### Symbol Mapping Not Working

**Problem:** Mapping is configured but still getting "No price data"

**Solutions:**

1. Verify mapping syntax:
   ```python
   SYMBOL_MAPPING = {
       "BNBUSDT": {
           "PancakeSwapV2": "USDTWBNB",  # Correct
       },
   }
   ```
2. Check exchange name spelling: Must match exactly (e.g., "PancakeSwapV2", not "PancakeswapV2")
3. Verify the mapped pair exists in CSV files
4. Check logs for mapping usage: Should see "Using symbol mapping: ..."

### No Opportunities Found

**Possible Reasons:**

1. **Threshold too high**: Lower `min_profit_pct` (e.g., try 0.05)
2. **No price differences**: Markets may be efficient (no arbitrage)
3. **Fees too high**: After fees and slippage, profit may be negative
4. **Data latency**: Stale data may show false opportunities
5. **Missing data**: Some exchanges may not have price data

### High CPU Usage

**Solutions:**

1. Increase `scan_interval` (e.g., 10.0 or 30.0 seconds)
2. Reduce number of symbols or exchanges
3. Check if market data client is logging too much (adjust log level)

### NATS Connection Issues

**Problem:** Cannot connect to NATS server

**Solutions:**

1. Verify NATS is running: `docker ps` (if using Docker)
2. Check NATS URL: Default is `nats://127.0.0.1:4222`
3. Check firewall/network settings
4. See `TESTING.md` for NATS setup instructions

## Examples

### Example 1: Basic CEX-to-CEX Arbitrage

```python
await run_arbitrage_detector(
    symbols=["ETHUSDT"],
    exchanges=[
        ("Binance", "spot"),
        ("Binance", "perpetual"),
    ],
    min_profit_pct=0.1,
)
```

### Example 2: CEX-to-DEX with Symbol Mapping

```python
await run_arbitrage_detector(
    symbols=["BNBUSDT"],
    exchanges=[
        ("Binance", "spot"),
        ("PancakeSwapV2", "swap"),
        ("PancakeSwapV3", "swap"),
    ],
    min_profit_pct=0.15,
    symbol_mapping={
        "BNBUSDT": {
            "PancakeSwapV2": "USDTWBNB",
            "PancakeSwapV3": "USDTWBNB",
        }
    }
)
```

### Example 3: Multiple Symbols

```python
await run_arbitrage_detector(
    symbols=["ETHUSDT", "BTCUSDT", "BNBUSDT"],
    exchanges=[
        ("Binance", "spot"),
        ("Binance", "perpetual"),
        ("PancakeSwapV2", "swap"),
    ],
    min_profit_pct=0.2,
    scan_interval=10.0,
)
```

### Example 4: Programmatic Use

```python
from arbitrage_detector import ArbitrageDetector
from market_data_client import MarketDataClient, CexConfig, DexConfig

# Create client
client = MarketDataClient(
    nats_url="nats://127.0.0.1:4222",
    cex=[CexConfig(exchange="Binance", symbols=["ETHUSDT"], instruments=["spot"])],
    dex=[DexConfig(exchange="PancakeSwapV2", chain="BSC", pairs=["USDTWBNB"])],
)
await client.start()

# Create detector
detector = ArbitrageDetector(
    client=client,
    min_profit_pct=0.1,
    symbols=["ETHUSDT"],
    exchanges=[("Binance", "spot"), ("PancakeSwapV2", "swap")],
)

# Scan once
opportunities = await detector.scan_opportunities()
if opportunities:
    detector.print_opportunities(opportunities)
else:
    print("No opportunities found")

await client.stop()
```

## Understanding the Output

### Profit Calculation Breakdown

For each opportunity, the detector shows:

1. **Raw Price Difference**: The difference between sell and buy prices
2. **Fees**: Trading fees on both sides (typically 0.1% each)
3. **Slippage**: Price impact (DEX only, typically 0-10 bps)
4. **Net Profit**: Final profit after all costs

**Example:**

- Buy at $2,500, Sell at $2,505
- Gross profit: $5 (0.20%)
- Fees: $2.50 + $2.51 = $5.01 (0.20%)
- Slippage: 2.2 bps = $0.06 (0.0022%)
- **Net profit: -$0.07 (-0.0022%)** ❌ Not profitable!

### When Opportunities Are Profitable

An opportunity is profitable when:

```
net_profit_pct = gross_profit_pct - fees_pct - slippage_pct >= min_profit_pct
```

**Example:**

- Buy at $2,500, Sell at $2,510
- Gross profit: $10 (0.40%)
- Fees: $5.01 (0.20%)
- Slippage: $0.06 (0.0022%)
- **Net profit: $4.93 (0.20%)** ✅ Profitable if threshold is ≤0.2%!

## Best Practices

1. **Start Conservative**: Begin with higher `min_profit_pct` (0.2-0.5%) to see realistic opportunities
2. **Verify Data**: Check CSV files to ensure market data is being published
3. **Use Symbol Mapping**: Always map CEX symbols to correct DEX pairs
4. **Monitor Latency**: High latency can make opportunities stale
5. **Consider Execution**: The detector shows opportunities, but execution requires:
   - Fast execution (opportunities may disappear quickly)
   - Sufficient balance on both exchanges
   - Gas fees (for DEX transactions)
   - Network fees (for blockchain transactions)

## Limitations

1. **Theoretical Only**: The detector shows opportunities but doesn't execute trades
2. **Data Latency**: Stale data may show false opportunities
3. **Execution Costs**: Additional costs (gas fees, network fees) not included
4. **Liquidity**: Assumes sufficient liquidity for execution
5. **Timing**: Opportunities may disappear before execution
6. **Regulatory**: Ensure compliance with local regulations

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review CSV files for available data
3. Check NATS connection and market data publisher
4. Review logs for detailed error messages

## License

Part of the market-client project.
