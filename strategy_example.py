# example_queries_with_latency.py
import asyncio
from market_data_client import make_client_simple, _iso_from_ns
import json

async def main():
    client = make_client_simple(
        nats_url="nats://127.0.0.1:4222",
        use_jetstream=False,
        cex_exchange="Binance",
        cex_instruments=["spot","perpetual"],
        cex_symbols=["ETHUSDT"],
        dex_exchange="PancakeSwapV2",
        dex_chain="BSC",
        dex_pairs=["WBNBUSDT"],
    )
    await client.start()
    print("✅ ready")
    await asyncio.sleep(3)
    # ----- Binance  -----
    t_spot = await client.get_latest_price_with_latency("ETHUSDT", "Binance", "spot")
    t_perp = await client.get_latest_price_with_latency("ETHUSDT", "Binance", "perpetual")
    f_meta = await client.get_funding_meta("ETHUSDT", "Binance")
    v_spot = await client.get_volume_meta("ETHUSDT", "Binance", "spot")
    v_perp = await client.get_volume_meta("ETHUSDT", "Binance", "perpetual")
    fee_sp  = await client.get_fee_meta("ETHUSDT", "Binance", "spot")
    fee_pp  = await client.get_fee_meta("ETHUSDT", "Binance", "perpetual")

    # ----- PancakeSwapV2  -----
    dex_price  = await client.get_dex_price_qb("WBNBUSDT", "PancakeSwapV2")
    dex_vol    = await client.get_volume_meta("WBNBUSDT", "PancakeSwapV2", "swap")
    dex_fee    = await client.get_fee_meta("WBNBUSDT", "PancakeSwapV2", "swap")
    dex_slip   = await client.get_slippage_meta("WBNBUSDT", "PancakeSwapV2")

    # ----- 출력 -----
    print("\n--- Binance ---")
    if t_spot:
        print(f"Spot mid     : {t_spot['mid']} (lat={t_spot['lat_ms']:.2f} ms @ {t_spot['ts_iso']})")
    else:
        print("Spot mid     : None")
    if t_perp:
        print(f"Perp mid     : {t_perp['mid']} (lat={t_perp['lat_ms']:.2f} ms @ {t_perp['ts_iso']})")
    else:
        print("Perp mid     : None")
    if f_meta:
        print(f"Funding rate : {f_meta.funding_rate} (next={_iso_from_ns(f_meta.next_funding_time_ms*1_000_000)} lat={f_meta.latency_ms:.2f} ms)")
    else:
        print("Funding rate : None")
    print(f"Volume spot  : {vars(v_spot) if v_spot else None}")
    print(f"Volume perp  : {vars(v_perp) if v_perp else None}")
    print(f"Fee spot     : {vars(fee_sp) if fee_sp else None}")
    print(f"Fee perp     : {vars(fee_pp) if fee_pp else None}")

    print("\n--- PancakeSwapV2 (BSC) ---")
    print("Price (WBNBUSDT):", dex_price["price_qb"])
    print(f"Volume (WBNBUSDT): {vars(dex_vol) if dex_vol else None}")
    print(f"Fee    (WBNBUSDT): {vars(dex_fee) if dex_fee else None}")
    print(f"Slip   (WBNBUSDT): {vars(dex_slip) if dex_slip else None}")

    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
