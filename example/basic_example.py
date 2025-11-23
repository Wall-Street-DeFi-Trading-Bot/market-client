import asyncio
from market_data_client.market_data_client import make_client_simple, _iso_from_ns

def ms(v: float) -> str:
    return f"{v:.2f} ms" if v is not None else "n/a"

async def main():
    client = make_client_simple(
        nats_url="nats://127.0.0.1:4222",
        use_jetstream=False,
        cex_exchange="Binance",
        cex_instruments=["spot","perpetual"],
        cex_symbols=["BNBUSDT"],
        dex_exchange="PancakeSwapV3",
        dex_chain="BSC",
        dex_pairs=["USDTWBNB"],
    )
    await client.start()
    print("✅ ready")
    await asyncio.sleep(3)

    # ----- Binance  -----
    t_spot = await client.get_latest_price_with_latency("BNBUSDT", "Binance", "spot")
    t_perp = await client.get_latest_price_with_latency("BNBUSDT", "Binance", "perpetual")
    f_meta = await client.get_funding_meta("BNBUSDT", "Binance")
    v_spot = await client.get_volume_meta("BNBUSDT", "Binance", "spot")
    v_perp = await client.get_volume_meta("BNBUSDT", "Binance", "perpetual")
    fee_sp  = await client.get_fee_meta("BNBUSDT", "Binance", "spot")
    fee_pp  = await client.get_fee_meta("BNBUSDT", "Binance", "perpetual")

    # ----- PancakeSwapV3  -----
    dex_price  = await client.get_dex_price_qb("USDTWBNB", "PancakeSwapV3")
    dex_vol    = await client.get_volume_meta("USDTWBNB", "PancakeSwapV3", "swap")
    dex_fee    = await client.get_fee_meta("USDTWBNB", "PancakeSwapV3", "swap")
    dex_slip   = await client.get_slippage_meta("USDTWBNB", "PancakeSwapV3")

    # ----- 출력 -----
    print("\n--- Binance ---")
    if t_spot:
        print(
            f"Spot mid     : {t_spot['mid']} "
            f"(src={t_spot['source_ts_iso']} pub={t_spot['publish_ts_iso']} "
            f"lat_src={ms(t_spot['lat_source_ms'])} lat_pub={ms(t_spot['lat_publish_ms'])})"
        )
    else:
        print("Spot mid     : None")

    if t_perp:
        print(
            f"Perp mid     : {t_perp['mid']} "
            f"(src={t_perp['source_ts_iso']} pub={t_perp['publish_ts_iso']} "
            f"lat_src={ms(t_perp['lat_source_ms'])} lat_pub={ms(t_perp['lat_publish_ms'])})"
        )
    else:
        print("Perp mid     : None")

    if f_meta:
        print(
            f"Funding rate : {f_meta.funding_rate} "
            f"(next={_iso_from_ns(f_meta.next_funding_time_ms * 1_000_000)} "
            f"lat_src={ms(f_meta.lat_source_ms)} lat_pub={ms(f_meta.lat_publish_ms)})"
        )
    else:
        print("Funding rate : None")

    print(f"Volume spot  : {vars(v_spot) if v_spot else None}")
    print(f"Volume perp  : {vars(v_perp) if v_perp else None}")
    print(f"Fee spot     : {vars(fee_sp) if fee_sp else None}")
    print(f"Fee perp     : {vars(fee_pp) if fee_pp else None}")

    print("\n--- PancakeSwapV3 (BSC) ---")
    if dex_price:
        print(
            f"Price (USDTWBNB): {dex_price['price_qb']} "
            f"(src={dex_price['source_ts_iso']} pub={dex_price['publish_ts_iso']} "
            f"lat_src={ms(dex_price['lat_source_ms'])} lat_pub={ms(dex_price['lat_publish_ms'])})"
        )
    else:
        print("Price (USDTWBNB): None")
    print(f"Volume (USDTWBNB): {vars(dex_vol) if dex_vol else None}")
    print(f"Fee    (USDTWBNB): {vars(dex_fee) if dex_fee else None}")
    print(f"Slip   (USDTWBNB): {vars(dex_slip) if dex_slip else None}")

    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
