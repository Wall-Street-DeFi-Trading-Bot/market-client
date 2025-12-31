import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from web3 import HTTPProvider, Web3
from eth_account import Account
from market_data_client.market_data_client import make_client_simple

import sys
import asyncio

# Ensure "src" is on sys.path so that "market_data_client" can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ABI_DIR = ROOT / "contracts/abi"

ERC20_ABI_PATH = ABI_DIR / "ERC20.json"
PERMIT2_ABI_PATH = ABI_DIR / "Permit2.json"

ERC20_ABI = json.loads(ERC20_ABI_PATH.read_text())
PERMIT2_ABI = json.loads(PERMIT2_ABI_PATH.read_text())

V3_SWAP_EXACT_IN = 0x00

from web3.exceptions import ABIFunctionNotFound  # noqa: F401 (kept for future use)

from market_data_client.arbitrage.exchange import (
    OrderSide,
    PancakeDemoParams,
    PancakeSwapDemoExchangeClient,
)
from market_data_client.arbitrage.state import BotState

# Disable extraData length/format validation for PoA-style chains such as BSC
try:
    import web3._utils.method_formatters as mf

    if hasattr(mf, "BLOCK_FORMATTERS") and "extraData" in mf.BLOCK_FORMATTERS:
        def _keep_extra_data(value: Any) -> Any:
            """Return extraData as-is without validation."""
            return value

        mf.BLOCK_FORMATTERS["extraData"] = _keep_extra_data
except Exception:
    # Best-effort; if this fails we still proceed
    pass

# Simple token symbol -> address map for BSC mainnet
TOKEN_ADDRESS_MAP: Dict[str, str] = {
    "WBNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "USDT": "0x55d398326f99059fF775485246999027B3197955",
    "WETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
    "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
    "BTCB": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",
    "TWT": "0x4B0F1812e5Df2A09796481Ff14017e6005508003",
    "SFP": "0xD41FDb03Ba84762dD66a0af1a6C8540FF1ba5dfb",
}

# Environment variables required for this test
REQUIRED_ENVS = [
    "DEMO_PANCAKE_FORK_RPC_URL",
    "DEMO_PANCAKE_UPSTREAM_RPC_URL",
    "DEMO_PANCAKE_ROUTER_ADDRESS",
    "DEMO_PANCAKE_ROUTER_ABI_JSON",
    "DEMO_PANCAKE_SWAP_PATH",
    "DEMO_PANCAKE_PERMIT2_ADDRESS",
]

MISSING_ENVS = [name for name in REQUIRED_ENVS if not os.getenv(name)]

# Debug print so you can see what pytest actually sees
print("[Pancake demo env check]")
for name in REQUIRED_ENVS:
    print(f"  {name} = {os.getenv(name)!r}")


def build_swap_tx_router(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
) -> Dict[str, Any]:
    """
    Build a Universal Router v2 `execute()` transaction for a single V3 exact-in swap.

    This function:
      - Resolves token addresses from DEMO_PANCAKE_SWAP_PATH
      - Uses `side` to decide path direction:
            BUY  → path as given (token_in = first, token_out = last)
            SELL → reversed path (token_in = last, token_out = first)
      - Ensures the trader has enough token_in in test mode (for WBNB)
      - Ensures ERC20 -> Permit2 and Permit2 -> router allowances
      - Encodes a single V3_SWAP_EXACT_IN command for Universal Router
      - Returns a tx dict that will be signed and sent by the client
    """
    router_address = Web3.to_checksum_address(
        os.environ["DEMO_PANCAKE_ROUTER_ADDRESS"].strip()
    )
    permit2_address = os.environ["DEMO_PANCAKE_PERMIT2_ADDRESS"].strip()

    abi_path = Path(os.environ["DEMO_PANCAKE_ROUTER_ABI_JSON"])
    abi = json.loads(abi_path.read_text())
    router = web3.eth.contract(address=router_address, abi=abi)

    path_env = os.environ["DEMO_PANCAKE_SWAP_PATH"]
    raw_tokens: List[str] = [p.strip() for p in path_env.split(",") if p.strip()]

    if len(raw_tokens) < 2:
        raise ValueError(
            f"DEMO_PANCAKE_SWAP_PATH must contain at least 2 tokens, got: {raw_tokens}"
        )

    def resolve_token(token: str) -> str:
        """
        Resolve a token symbol or address string to a checksum address.

        - If the string already looks like an address (starts with 0x), just checksum it.
        - Otherwise, look it up in TOKEN_ADDRESS_MAP.
        """
        if token.startswith("0x") or token.startswith("0X"):
            return Web3.to_checksum_address(token)
        addr = TOKEN_ADDRESS_MAP.get(token.upper())
        if not addr:
            raise ValueError(f"Unknown token symbol in DEMO_PANCAKE_SWAP_PATH: {token}")
        return Web3.to_checksum_address(addr)

    path_addresses: List[str] = [resolve_token(t) for t in raw_tokens]

    side_upper = side.upper()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported side for build_swap_tx_router: {side}")

    # For BUY: use path as-is (token_in = first, token_out = last)
    # For SELL: reverse direction (token_in = last, token_out = first)
    if side_upper == "BUY":
        token_in = path_addresses[0]
        token_out = path_addresses[-1]
        path_for_log = raw_tokens
    else:  # SELL
        token_in = path_addresses[-1]
        token_out = path_addresses[0]
        path_for_log = list(reversed(raw_tokens))

    print(
        f"[build_swap_tx_router] side={side_upper} "
        f"token_in={token_in} token_out={token_out} "
        f"path={path_for_log}"
    )

    # Trader is derived from the local private key, not impersonation.
    priv_key = os.environ["DEMO_PANCAKE_TRADER_PRIVATE_KEY"].strip()
    acct = Account.from_key(priv_key)
    trader = Web3.to_checksum_address(acct.address)

    token_in_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
    token_out_contract = web3.eth.contract(address=token_out, abi=ERC20_ABI)

    try:
        decimals_in = token_in_contract.functions.decimals().call()
    except Exception:
        # Fallback for tokens that do not expose decimals() properly
        decimals_in = int(os.getenv("DEMO_PANCAKE_DECIMALS_IN", "18") or "18")

    amount_in = int(quantity * (10**decimals_in))
    amount_out_min = 0  # no slippage protection in this demo
    deadline = int(time.time()) + 600

    gas_limit = int(os.getenv("DEMO_PANCAKE_GAS_LIMIT", "500000"))
    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    # Optionally fund the trader with WBNB in test mode.
    _fund_test_tokens(
        web3=web3,
        trader=trader,
        token_in=token_in,
        amount_in=amount_in,
        private_key=priv_key,
    )

    bal_in_raw = token_in_contract.functions.balanceOf(trader).call()
    bal_out_raw = token_out_contract.functions.balanceOf(trader).call()
    native_wei = web3.eth.get_balance(trader)

    # Ensure ERC20->Permit2 and Permit2->Router allowances are large enough.
    _ensure_allowance(
        web3=web3,
        trader=trader,
        token_in=token_in,
        permit2_address=permit2_address,
        router_address=router_address,
        amount_in=amount_in,
        private_key=priv_key,
    )

    print(
        f"[pancake-debug] trader={trader} native_wei={native_wei} "
        f"token_in={token_in} bal_in_raw={bal_in_raw} "
        f"token_out={token_out} bal_out_raw={bal_out_raw} "
        f"amount_in={amount_in}"
    )

    fee = int(os.getenv("DEMO_PANCAKE_V3_FEE", "2500"))
    if fee < 0 or fee > 2**24 - 1:
        raise ValueError(f"Invalid V3 fee: {fee}")
    fee_bytes = fee.to_bytes(3, byteorder="big")

    # Packed V3 path: tokenIn (20 bytes) | fee (3 bytes) | tokenOut (20 bytes)
    path_bytes = bytes.fromhex(token_in[2:]) + fee_bytes + bytes.fromhex(token_out[2:])

    # For a single hop exact-in, we keep sqrtPriceLimitX96 = 0 (unused with Universal Router encoding)
    sqrt_price_limit_x96 = 0  # noqa: F841

    # V3SwapExactInParams:
    #   address recipient;
    #   uint256 amountIn;
    #   uint256 amountOutMinimum;
    #   bytes path;
    #   bool payerIsUser;
    params_types = [
        "address",
        "uint256",
        "uint256",
        "bytes",
        "bool",
    ]
    params_values = [
        trader,
        amount_in,
        amount_out_min,
        path_bytes,
        True,  # payerIsUser: msg.sender pays via Permit2
    ]

    # Encode the struct as ABI bytes for Universal Router `inputs[0]`.
    input_bytes = web3.codec.encode(params_types, params_values)

    # Single command: V3_SWAP_EXACT_IN
    commands = bytes([V3_SWAP_EXACT_IN])

    deadline = int(time.time()) + 600

    fn = router.functions.execute(
        commands,
        [input_bytes],
        deadline,
    )
    data = fn._encode_transaction_data()

    tx: Dict[str, Any] = {
        "to": router_address,
        "from": trader,
        "data": data,
        "nonce": web3.eth.get_transaction_count(trader),
        "gas": gas_limit,
        "gasPrice": gas_price,
        "value": 0,
    }

    # Demo-only metadata so the client can compute per-block PnL.
    # These keys are removed in PancakeSwapDemoExchangeClient before sending on-chain.
    tx["_demo_token_in"] = token_in
    tx["_demo_token_out"] = token_out

    return tx


def _fund_test_tokens(
    web3: Web3,
    trader: str,
    token_in: str,
    amount_in: int,
    private_key: str,
) -> None:
    """
    Fund token_in balance for the trader only in test mode.

    - If token_in is WBNB:
        Wrap native BNB into WBNB via deposit().
    - If token_in is USDT:
        Impersonate DEMO_PANCAKE_USDT_WHALE on the fork and transfer USDT to the trader.
    """

    provider = web3.provider
    if provider is None:
        return

    # Ensure the trader has enough native BNB to pay for gas (and wrapping).
    try:
        native_balance = web3.eth.get_balance(trader)
    except Exception:
        native_balance = 0

    target_native_bnb = float(os.getenv("DEMO_PANCAKE_NATIVE_BALANCE_BNB", "1"))
    target_native_wei = web3.to_wei(target_native_bnb, "ether")

    if native_balance < target_native_wei:
        # Best-effort: support both Anvil and Hardhat forks.
        try:
            provider.make_request("anvil_setBalance", [trader, hex(target_native_wei)])
        except Exception:
            try:
                provider.make_request("hardhat_setBalance", [trader, hex(target_native_wei)])
            except Exception:
                pass

    wbnb_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["WBNB"])
    usdt_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["USDT"])

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    # 1) WBNB funding: wrap native BNB via deposit().
    if token_in == wbnb_addr:
        token_in_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
        try:
            bal_in_raw = token_in_contract.functions.balanceOf(trader).call()
        except Exception:
            bal_in_raw = 0

        if bal_in_raw >= amount_in:
            return

        wrap_amount = amount_in - bal_in_raw
        nonce = web3.eth.get_transaction_count(trader)

        deposit_tx = {
            "to": token_in,
            "from": trader,
            "value": wrap_amount,
            "data": "0xd0e30db0",  # deposit()
            "gas": 200000,
            "gasPrice": gas_price,
            "nonce": nonce,
        }

        signed = Account.sign_transaction(deposit_tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)
        return

    # 2) USDT funding: impersonate a rich USDT holder on the fork and transfer.
    if token_in == usdt_addr:
        usdt_whale = os.getenv("DEMO_PANCAKE_USDT_WHALE")
        if not usdt_whale:
            # If no whale is configured, we cannot fund USDT in test mode.
            print(
                "[pancake-fund] DEMO_PANCAKE_USDT_WHALE is not set; "
                "skipping USDT funding for demo."
            )
            return

        whale = Web3.to_checksum_address(usdt_whale)

        # Ensure the whale has enough native BNB for gas on the fork.
        try:
            provider.make_request("anvil_setBalance", [whale, hex(target_native_wei)])
        except Exception:
            try:
                provider.make_request("hardhat_setBalance", [whale, hex(target_native_wei)])
            except Exception:
                pass

        token_in_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
        try:
            bal_in_raw = token_in_contract.functions.balanceOf(trader).call()
        except Exception:
            bal_in_raw = 0

        if bal_in_raw >= amount_in:
            return

        transfer_amount = amount_in - bal_in_raw
        nonce_whale = web3.eth.get_transaction_count(whale)

        fork_engine = os.getenv("DEMO_PANCAKE_FORK_ENGINE", "anvil")
        # Impersonate the whale account on the local fork.
        try:
            if fork_engine == "anvil":
                provider.make_request("anvil_impersonateAccount", [whale])
            elif fork_engine == "hardhat":
                provider.make_request("hardhat_impersonateAccount", [whale])
        except Exception:
            # If impersonation fails, we just log and skip; transfer will likely revert.
            print("[pancake-fund] Failed to impersonate USDT whale on fork.")

        transfer_tx = token_in_contract.functions.transfer(
            trader,
            transfer_amount,
        ).build_transaction(
            {
                "from": whale,
                "gas": 200000,
                "gasPrice": gas_price,
                "nonce": nonce_whale,
            }
        )

        tx_hash = web3.eth.send_transaction(transfer_tx)
        web3.eth.wait_for_transaction_receipt(tx_hash)
        return

    # For any other token_in we do nothing in test funding.
    return


def _ensure_allowance(
    web3: Web3,
    trader: str,
    token_in: str,
    permit2_address: str,
    router_address: str,
    amount_in: int,
    private_key: str,
) -> None:
    """
    Ensure allowances for Universal Router using Permit2.

    This sets:
      1) ERC20 allowance: owner(trader) -> spender(Permit2)
      2) Permit2 allowance: owner(trader), token(token_in) -> spender(router)
    """

    permit2_addr = Web3.to_checksum_address(permit2_address)
    router_addr = Web3.to_checksum_address(router_address)
    trader_addr = Web3.to_checksum_address(trader)
    token_addr = Web3.to_checksum_address(token_in)

    token_contract = web3.eth.contract(address=token_addr, abi=ERC20_ABI)
    permit2 = web3.eth.contract(address=permit2_addr, abi=PERMIT2_ABI)

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    # Use chain time to avoid issues on fork reset / block timestamp changes.
    now_ts = int(web3.eth.get_block("latest")["timestamp"])

    max_uint256 = (1 << 256) - 1
    max_uint160 = (1 << 160) - 1
    max_expiration_uint48 = (1 << 48) - 1  # uint48 max

    # Best-effort: verify router's Permit2 address matches (if ABI exposes it).
    try:
        router_abi_path = Path(os.environ["DEMO_PANCAKE_ROUTER_ABI_JSON"])
        router_abi = json.loads(router_abi_path.read_text(encoding="utf-8"))
        router_contract = web3.eth.contract(address=router_addr, abi=router_abi)

        # Some router ABIs expose PERMIT2() as a view function.
        if hasattr(router_contract.functions, "PERMIT2"):
            router_permit2 = router_contract.functions.PERMIT2().call()
            if Web3.to_checksum_address(router_permit2) != permit2_addr:
                raise ValueError(
                    f"Permit2 mismatch: router={router_permit2}, env={permit2_addr}"
                )
    except Exception:
        # Non-fatal; continue.
        pass

    nonce = web3.eth.get_transaction_count(trader_addr)

    # 1) ERC20 approve: trader -> Permit2
    try:
        erc20_allowance = int(token_contract.functions.allowance(trader_addr, permit2_addr).call())
    except Exception:
        erc20_allowance = 0

    if erc20_allowance < amount_in:
        # Some tokens (e.g., USDT-like) may require resetting allowance to 0 first.
        if erc20_allowance != 0:
            try:
                tx0 = token_contract.functions.approve(permit2_addr, 0).build_transaction(
                    {
                        "from": trader_addr,
                        "gas": 200_000,
                        "gasPrice": gas_price,
                        "nonce": nonce,
                    }
                )
                signed0 = web3.eth.account.sign_transaction(tx0, private_key)
                h0 = web3.eth.send_raw_transaction(signed0.raw_transaction)
                web3.eth.wait_for_transaction_receipt(h0)
                nonce += 1
            except Exception:
                # Best-effort; continue to the next approve.
                pass

        tx1 = token_contract.functions.approve(permit2_addr, max_uint256).build_transaction(
            {
                "from": trader_addr,
                "gas": 200_000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )
        signed1 = web3.eth.account.sign_transaction(tx1, private_key)
        h1 = web3.eth.send_raw_transaction(signed1.raw_transaction)
        web3.eth.wait_for_transaction_receipt(h1)
        nonce += 1

    # 2) Permit2 allowance: (owner=trader, token=token_in) -> spender(router)
    try:
        res = permit2.functions.allowance(trader_addr, token_addr, router_addr).call()
        allowed_amount = int(res[0])  # uint160
        expiration = int(res[1])      # uint48
        # res[2] is nonce in Permit2 struct; not needed here
    except Exception:
        allowed_amount, expiration = 0, 0

    need_permit2 = (allowed_amount < amount_in) or (expiration <= now_ts)

    if need_permit2:
        tx2 = permit2.functions.approve(
            token_addr,
            router_addr,
            max_uint160,
            max_expiration_uint48,  # never expires (uint48 max)
        ).build_transaction(
            {
                "from": trader_addr,
                "gas": 200_000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )
        signed2 = web3.eth.account.sign_transaction(tx2, private_key)
        h2 = web3.eth.send_raw_transaction(signed2.raw_transaction)
        web3.eth.wait_for_transaction_receipt(h2)


async def _get_price_hint_from_market_data(side: str) -> Optional[float]:
    """
    Try to fetch a reference price from MarketDataClient.

    If NATS is down, cannot connect, or no price is received within the timeout,
    this function returns None so the caller can fall back to an internal
    strategy default price.
    """
    exchange = os.getenv("MDC_HINT_EXCHANGE")
    symbol = os.getenv("MDC_HINT_SYMBOL")
    instrument = os.getenv("MDC_HINT_INSTRUMENT", "perpetual").lower()

    if not exchange or not symbol:
        print(
            "[mdc-hint] MDC_HINT_EXCHANGE / MDC_HINT_SYMBOL not set, "
            "using internal fallback price (no market data)."
        )
        return None

    raw_nats_url = os.getenv("NATS_URL")
    nats_url = raw_nats_url or "nats://127.0.0.1:4222"
    use_js = os.getenv("MDC_HINT_USE_JS", "1") != "0"
    timeout_sec = float(os.getenv("MDC_HINT_TIMEOUT_SEC", "5"))
    connect_timeout_sec = float(os.getenv("MDC_HINT_CONNECT_TIMEOUT_SEC", "3"))

    print(
        f"[mdc-hint-debug] EX= {exchange} SYM= {symbol} INST= {instrument} "
        f"TO= {timeout_sec} JS= {os.getenv('MDC_HINT_USE_JS')} "
        f"NATS_URL_ENV={raw_nats_url} NATS_URL_EFFECTIVE={nats_url}"
    )

    print(
        f"[mdc-hint] Waiting up to {timeout_sec}s for market price "
        f"{exchange} {instrument} {symbol} via {nats_url} "
        f"(JetStream={use_js})"
    )

    try:
        client = make_client_simple(
            nats_url=nats_url,
            use_jetstream=use_js,
            cex_exchange=exchange if instrument != "swap" else None,
            cex_instruments=[instrument] if instrument != "swap" else None,
            cex_symbols=[symbol] if instrument != "swap" else None,
            dex_exchange=exchange if instrument == "swap" else None,
            dex_chain=os.getenv("MDC_HINT_DEX_CHAIN", "BSC") if instrument == "swap" else None,
            dex_pairs=[symbol] if instrument == "swap" else None,
            enable_csv=False,
        )
    except Exception as e:
        print(
            f"[mdc-hint] Failed to build MarketDataClient ({e}); "
            "using internal fallback price (no market data)."
        )
        return None

    started = False
    try:
        # Short timeout for connecting to NATS
        try:
            await asyncio.wait_for(client.start(), timeout=connect_timeout_sec)
            started = True
        except asyncio.TimeoutError:
            print(
                f"[mdc-hint] Timed out connecting to NATS at {nats_url} "
                f"after {connect_timeout_sec}s; using internal fallback price."
            )
            return None
        except Exception as e:
            print(
                f"[mdc-hint] Failed to connect to NATS at {nats_url}: {e}; "
                "using internal fallback price."
            )
            return None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        price: Optional[float] = None

        # DEX path: use get_dex_price_qb for swap instruments
        if instrument == "swap" and exchange in ("PancakeSwapV2", "PancakeSwapV3"):
            pair = symbol  # e.g. "USDTWBNB"

            while loop.time() < deadline:
                try:
                    dp = await client.get_dex_price_qb(pair, exchange)
                except Exception as e:
                    print(
                        f"[mdc-hint] Error in get_dex_price_qb for {exchange} {pair}: {e}; "
                        "will retry until timeout."
                    )
                    dp = None

                if dp and dp.get("price_qb") is not None:
                    raw_price_qb = float(dp["price_qb"])

                    if raw_price_qb > 0:
                        price = 1.0 / raw_price_qb

                        print(
                            "[mdc-hint] Using DEX price_qb from market data: "
                            f"exchange={exchange}, pair={pair}, "
                            f"raw_price_qb={raw_price_qb}, side={side}, "
                            f"price_for_hint={price}"
                        )
                    else:
                        print(
                            f"[mdc-hint] Got non-positive price_qb={raw_price_qb} "
                            f"for {exchange} {pair}"
                        )

                    break

                await asyncio.sleep(0.1)

        # CEX path: use mid-price
        else:
            while loop.time() < deadline:
                try:
                    pd = await client.get_latest_price_with_latency(symbol, exchange, instrument)
                except Exception as e:
                    print(
                        f"[mdc-hint] Error in get_latest_price_with_latency for "
                        f"{exchange} {instrument} {symbol}: {e}; "
                        "will retry until timeout."
                    )
                    pd = None

                if pd and pd.get("mid") is not None:
                    price = float(pd["mid"])
                    print(
                        "[mdc-hint] Using CEX mid price from market data: "
                        f"exchange={exchange}, instrument={instrument}, "
                        f"symbol={symbol}, mid={price}"
                    )
                    break

                await asyncio.sleep(0.1)

        if price is None:
            print(
                f"[mdc-hint] No market price for {exchange} {instrument} {symbol} "
                f"within {timeout_sec}s; using internal fallback price."
            )

        return price

    except Exception as e:
        print(
            f"[mdc-hint] Unexpected error while fetching market data: {e}; "
            "using internal fallback price."
        )
        return None

    finally:
        if started:
            try:
                await client.stop()
            except Exception:
                pass



def _inspect_trade(label: str, trade) -> None:
    """
    Helper to inspect and log pancake_demo metadata for a single trade.
    """
    meta = trade.metadata or {}
    assert "pancake_demo" in meta

    demo_meta = meta["pancake_demo"]
    block_offsets = demo_meta["block_offsets"]
    per_block_results = demo_meta["per_block_results"]

    assert len(per_block_results) == len(block_offsets)

    print(f"\n=== {label} leg: per-block results ===")

    fill_prices: List[float] = []
    returns: List[float] = []

    for result in per_block_results:
        assert "fork_block" in result
        assert "tx_hash" in result
        assert "status" in result
        assert result["status"] in (0, 1)

        delta_in = result.get("delta_in")
        delta_out = result.get("delta_out")
        fill_price = result.get("fill_price")
        block_ret = result.get("return_vs_hint")

        print(
            f"[pancake-per-block-{label}] fork_block={result['fork_block']} "
            f"status={result['status']} "
            f"delta_in={delta_in} delta_out={delta_out} "
            f"fill_price={fill_price} return_vs_hint={block_ret}"
        )

        if fill_price is not None:
            fill_prices.append(float(fill_price))
        if block_ret is not None:
            returns.append(float(block_ret))

    if fill_prices:
        avg_fill_price_meta = demo_meta.get("avg_fill_price")
        if avg_fill_price_meta is None:
            avg_fill_price_meta = sum(fill_prices) / len(fill_prices)
        print(f"[pancake-summary-{label}] avg_fill_price={avg_fill_price_meta}")

    if returns:
        avg_return_meta = demo_meta.get("avg_return_vs_hint")
        if avg_return_meta is None:
            avg_return_meta = sum(returns) / len(returns)
        print(f"[pancake-summary-{label}] avg_return_vs_hint={avg_return_meta}")

    print(f"\n=== Pancake demo metadata ({label} leg) ===")
    meta_for_log = dict(demo_meta)
    meta_for_log.pop("account_snapshot", None)
    print(json.dumps(meta_for_log, indent=2))


async def _run_pancake_demo_test() -> None:
    """
    Dynamic check for PancakeSwapDemoExchangeClient.

    It should:
        - Use a forked BSC node (Hardhat or Anvil)
        - Reset to base_block + (1, 2, 3)
        - Sign locally for the trader account or impersonate
        - Send a swap transaction on each fork
        - Expose per-block metadata for analysis
        - Run both BUY and SELL legs so side and path direction can be tested
    """
    fork_rpc_url = os.environ["DEMO_PANCAKE_FORK_RPC_URL"]
    upstream_rpc_url = os.environ["DEMO_PANCAKE_UPSTREAM_RPC_URL"]
    fork_engine = os.getenv("DEMO_PANCAKE_FORK_ENGINE", "anvil")

    web3 = Web3(HTTPProvider(fork_rpc_url))
    state = BotState()

    # Use local private key mode
    private_key = os.environ["DEMO_PANCAKE_TRADER_PRIVATE_KEY"].strip()

    params = PancakeDemoParams(
        web3=web3,
        account_address="",  # derived from private_key in client
        build_swap_tx=build_swap_tx_router,
        private_key=private_key,
        block_offsets=(1, 2, 3),
        default_fee_rate=0.0005,
    )
    params.upstream_rpc_url = upstream_rpc_url  # type: ignore[attr-defined]
    params.fork_engine = fork_engine  # type: ignore[attr-defined]

    # On-chain trader and token contracts for balance checks
    acct = Account.from_key(private_key)
    trader_onchain = Web3.to_checksum_address(acct.address)

    wbnb_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["WBNB"])
    usdt_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["USDT"])

    wbnb = web3.eth.contract(address=wbnb_addr, abi=ERC20_ABI)
    usdt = web3.eth.contract(address=usdt_addr, abi=ERC20_ABI)

    # On-chain balances before any swaps
    wbnb_before = wbnb.functions.balanceOf(trader_onchain).call()
    usdt_before_token = usdt.functions.balanceOf(trader_onchain).call()
    print(
        f"[pancake-onchain-before] trader={trader_onchain} "
        f"wbnb={wbnb_before} usdt={usdt_before_token}"
    )

    client = PancakeSwapDemoExchangeClient(
        exchange_name="PancakeSwapV3",
        instrument="swap",
        state=state,
        params=params,
    )

    # Internal bot account snapshot (off-chain accounting)
    account = state.get_or_create_account("PancakeSwapV3", "swap")
    account.deposit("USDT", 1000.0)

    symbol = "DEMO_WBNBUSDT"
    quantity = float(os.getenv("DEMO_PANCAKE_TEST_QUANTITY", "0.01"))

    # --- BUY leg: use first snapshot price as hint ---
    fallback_price_buy = 100.0  # internal demo-only default
    mdc_hint_buy = await _get_price_hint_from_market_data("BUY")
    price_hint_buy = mdc_hint_buy if mdc_hint_buy is not None else fallback_price_buy

    print(
        f"[demo-price-hint-BUY] price_hint_buy={price_hint_buy} "
        f"(from_market_data={mdc_hint_buy}, fallback={fallback_price_buy})"
    )


    print("\n=== BUY leg ===")
    trade_buy = await client.create_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price_hint_buy,
    )

    assert trade_buy.success is True
    assert trade_buy.exchange == "PancakeSwapV3"
    assert trade_buy.symbol == symbol

    _inspect_trade("BUY", trade_buy)

    base_asset = symbol.replace("USDT", "")  # "DEMO_WBNB" for "DEMO_WBNBUSDT"
    base_balance = account.balances.get(base_asset, 0.0)
    sell_qty = min(base_balance, quantity)

    if sell_qty <= 0:
        raise AssertionError(
            f"No base asset balance to sell after BUY leg (asset={base_asset}, balance={base_balance})"
        )

    # --- SELL leg: refresh hint with a new real-time snapshot ---
    fallback_price_sell = 100.0  # internal demo-only default
    mdc_hint_sell = await _get_price_hint_from_market_data("SELL")
    price_hint_sell = mdc_hint_sell if mdc_hint_sell is not None else fallback_price_sell

    print(
        f"[demo-price-hint-SELL] price_hint_sell={price_hint_sell} "
        f"(from_market_data={mdc_hint_sell}, fallback={fallback_price_sell})"
    )

    trade_sell = await client.create_market_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=sell_qty,
        price=price_hint_sell,
    )



    assert trade_sell.success is True
    assert trade_sell.exchange == "PancakeSwapV3"
    assert trade_sell.symbol == symbol

    _inspect_trade("SELL", trade_sell)

    # On-chain balances after both legs
    wbnb_after = wbnb.functions.balanceOf(trader_onchain).call()
    usdt_after_token = usdt.functions.balanceOf(trader_onchain).call()
    print(
        f"[pancake-onchain-after] trader={trader_onchain} "
        f"wbnb={wbnb_after} usdt={usdt_after_token}"
    )

    print("\n=== Final BotState balances ===")
    print(account.balances)


@pytest.mark.skipif(
    bool(MISSING_ENVS),
    reason=(
        "Pancake demo environment variables are not fully configured. "
        f"Missing: {', '.join(MISSING_ENVS)}"
    ),
)
def test_pancake_demo_runs_swaps_on_three_forked_blocks_bidirectional() -> None:
    asyncio.run(_run_pancake_demo_test())


if __name__ == "__main__":
    asyncio.run(_run_pancake_demo_test())
