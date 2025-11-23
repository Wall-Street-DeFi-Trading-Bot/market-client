"""
Quick Start Example for Arbitrage Detector

This is a simple example showing how to use the arbitrage detector.
Modify the configuration below to match your needs.
"""

import asyncio

from market_data_client.arbitrage.arbitrage_detector import run_arbitrage_detector


async def main():
    """
    Example: Monitor ETHUSDT for arbitrage between Binance spot and perpetual.
    
    This will:
    1. Connect to NATS and subscribe to market data
    2. Monitor ETHUSDT prices on Binance spot and perpetual markets
    3. Scan every 5 seconds for opportunities
    4. Report any opportunities with >0.1% profit
    """
    
    # Configuration
    SYMBOLS = ["BNBUSDT"]  # Trading pairs to monitor
    
    EXCHANGES = [
        ("Binance", "spot"),      # Binance spot market
        ("Binance", "perpetual"), # Binance perpetual futures
        ("PancakeSwapV2", "swap"), # PancakeSwap V2 DEX
        ("PancakeSwapV3", "swap"), # PancakeSwap V3 DEX
        # Add more exchanges as they become available:
        # ("Coinbase", "spot"),       # Another CEX (if available)
        # ("UniswapV2", "swap"),      # Uniswap V2 (if available)
        # ("UniswapV3", "swap"),      # Uniswap V3 (if available)
    ]
    
    MIN_PROFIT_PCT = 0.1   # Minimum 0.1% profit to report
    SCAN_INTERVAL = 5.0    # Scan every 5 seconds
    
    print("=" * 80)
    print("Cross-Exchange Arbitrage Detector - Quick Start")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Exchanges: {', '.join([f'{ex}({inst})' for ex, inst in EXCHANGES])}")
    print(f"  Min Profit: {MIN_PROFIT_PCT}%")
    print(f"  Scan Interval: {SCAN_INTERVAL}s")
    print("\nPress Ctrl+C to stop\n")
    
    # Optional: Map CEX symbols to DEX pair names
    # This allows you to use "ETHUSDT" for CEX but "USDTWBNB" for DEX
    # Check CSV files (./csv/swaps.csv) to see what pairs are actually available
    SYMBOL_MAPPING = {
        # Uncomment and adjust based on your actual DEX pairs:
        "BNBUSDT": {
            "PancakeSwapV2": "USDTWBNB",  # Use USDTWBNB on PancakeSwapV2
            "PancakeSwapV3": "USDTWBNB",  # Use USDTWBNB on PancakeSwapV3
        },
    }
    
    # Run the detector
    await run_arbitrage_detector(
        symbols=SYMBOLS,
        exchanges=EXCHANGES,
        min_profit_pct=MIN_PROFIT_PCT,
        scan_interval=SCAN_INTERVAL,
        nats_url="nats://127.0.0.1:4222",
        symbol_mapping=SYMBOL_MAPPING,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped by user")

