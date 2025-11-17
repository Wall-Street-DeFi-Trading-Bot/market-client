import asyncio
from market_data_client import MarketDataClient, CexConfig, DexConfig

def ms(v): return f"{v:.2f} ms" if v is not None else "n/a"

async def main():
    client = MarketDataClient(
        nats_url="nats://127.0.0.1:4222",
        use_jetstream=False,
        enable_csv=True,
        cex=[CexConfig(
            exchange="Binance",
            symbols=["BNBUSDT"],
            instruments=["spot", "perpetual"],
            want=("tick","funding","fee","volume"),
        )],
        dex=[
            DexConfig(exchange="PancakeSwapV2", chain="BSC", pairs=["USDTWBNB","CAKEUSDT","CAKEWBNB","TWTWBNB","USDTWBNB","USDTBTCB","SFPWBNB"]),
            DexConfig(exchange="PancakeSwapV3", chain="BSC", pairs=["USDTWBNB","CAKEUSDT","CAKEWBNB","TWTWBNB","USDTWBNB","USDTBTCB","SFPWBNB"]),
        ],
    )
    await client.start()
    print("✅ live, printing as messages arrive")

    async def on_ev(ev):
        k = ev["kind"]
        lat_src = ev.get("lat_source_ms")
        lat_pub = ev.get("lat_publish_ms")

        if k == "tick":
            print(
                f"[TICK] {ev['exchange']} {ev.get('symbol','')} {ev.get('instrument','')} "
                f"mid={ev.get('mid')} "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )
        elif k == "funding":
            print(
                f"[FUND] {ev['exchange']} {ev['symbol']} "
                f"rate={ev['rate']} next={ev['next_ms']} "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )
        elif k == "fee":
            print(
                f"[FEE ] {ev['exchange']} {ev['symbol']} {ev.get('instrument','')} "
                f"maker={ev['maker_rate']} taker={ev['taker_rate']} "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )
        elif k == "volume":
            print(
                f"[VOL ] {ev['exchange']} {ev['symbol']} {ev.get('instrument','')} "
                f"v0={ev['volume0']} v1={ev['volume1']} trades={ev['trades']} "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )
        elif k == "slippage":
            print(
                f"[SLIP] {ev['exchange']} {ev['pair']} "
                f"bps01={ev['bps01']} bps10={ev['bps10']} "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )
        elif k == "dexSwapL1":
            print(
                f"[DEX ] {ev['exchange']} {ev['pair']} price={ev['price']} "
                f"(p01={ev['price01']} p10={ev['price10']} inv={int(ev['invert_for_quote'])}) "
                f"lat_src={ms(lat_src)} lat_pub={ms(lat_pub)}"
            )

    # Binance + PancakeSwapV2 + V3 둘 다 들음
    client.add_listener(on_ev, exchange={"Binance","PancakeSwapV2","PancakeSwapV3"})

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
