import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest
from web3 import HTTPProvider, Web3
from eth_account import Account
from eth_abi import encode as abi_encode

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


import pytest
from web3 import HTTPProvider, Web3
from web3.exceptions import ABIFunctionNotFound

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
    "DEMO_PANCAKE_PERMIT2_ADDRESS"
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
      - Resolves token addresses from DEMO_PANCAKE_SWAP_PATH (e.g. WBNB,USDT)
      - Ensures the trader has enough token_in in test mode
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
    token_in = path_addresses[0]
    token_out = path_addresses[-1]

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

    amount_in = int(quantity * (10 ** decimals_in))
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

    # For a single hop exact-in, we keep sqrtPriceLimitX96 = 0
    sqrt_price_limit_x96 = 0

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
    """Fund WBNB balance for the trader only in test mode."""
    if os.getenv("DEMO_PANCAKE_TEST_FUND", "0") != "1":
        return

    provider = web3.provider
    if provider is None:
        return

    try:
        native_balance = web3.eth.get_balance(trader)
    except Exception:
        native_balance = 0

    target_native_bnb = float(os.getenv("DEMO_PANCAKE_NATIVE_BALANCE_BNB", "1"))
    target_native_wei = web3.to_wei(target_native_bnb, "ether")

    if native_balance < target_native_wei:
        try:
            provider.make_request("anvil_setBalance", [trader, hex(target_native_wei)])
        except Exception:
            try:
                provider.make_request("hardhat_setBalance", [trader, hex(target_native_wei)])
            except Exception:
                pass

    wbnb_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["WBNB"])
    if token_in != wbnb_addr:
        return

    token_in_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
    try:
        bal_in_raw = token_in_contract.functions.balanceOf(trader).call()
    except Exception:
        bal_in_raw = 0

    if bal_in_raw >= amount_in:
        return

    wrap_amount = amount_in - bal_in_raw
    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")
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



def _ensure_allowance(
    web3: Web3,
    trader: str,
    token_in: str,
    permit2_address: str,
    router_address: str,
    amount_in: int,
    private_key: str,
) -> None:
    """Ensure ERC20 -> Permit2 and Permit2 -> router allowances for Universal Router."""
    permit2_addr = Web3.to_checksum_address(permit2_address)
    router_addr = Web3.to_checksum_address(router_address)

    token_contract = web3.eth.contract(address=token_in, abi=ERC20_ABI)
    permit2 = web3.eth.contract(address=permit2_addr, abi=PERMIT2_ABI)

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")
    nonce = web3.eth.get_transaction_count(trader)

    max_uint256 = 2**256 - 1
    max_uint160 = 2**160 - 1
    now_ts = int(time.time())
    max_expiration = 2**48 - 1
    desired_exp = min(max_expiration, now_ts + 10 * 365 * 24 * 60 * 60)

    # 1) ERC20 allowance trader -> Permit2
    try:
        erc20_allowance = token_contract.functions.allowance(trader, permit2_addr).call()
    except Exception:
        erc20_allowance = 0

    if erc20_allowance < amount_in:
        approve_tx = token_contract.functions.approve(
            permit2_addr,
            max_uint256,
        ).build_transaction(
            {
                "from": trader,
                "gas": 200000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )
        nonce += 1
        signed = web3.eth.account.sign_transaction(approve_tx, private_key)

        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)

    # 2) Permit2 allowance trader -> router
    try:
        res = permit2.functions.allowance(trader, token_in, router_addr).call()
        # struct PermitDetails { uint160 amount; uint48 expiration; uint48 nonce; } in first slot
        if isinstance(res, (list, tuple)):
            allowed_amount = int(res[0])
            expiration = int(res[1])
        else:
            allowed_amount = 0
            expiration = 0
    except Exception:
        allowed_amount = 0
        expiration = 0

    need_permit2 = allowed_amount < amount_in or expiration <= now_ts

    if need_permit2:
        permit2_tx = permit2.functions.approve(
            token_in,
            router_addr,
            max_uint160,
            desired_exp,
        ).build_transaction(
            {
                "from": trader,
                "gas": 200000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )
        signed = web3.eth.account.sign_transaction(permit2_tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)





async def _run_pancake_demo_test() -> None:
    """
    Dynamic check for PancakeSwapDemoExchangeClient.

    It should:
        - Use a forked BSC node (Hardhat or Anvil)
        - Reset to base_block + (1, 2, 3)
        - Impersonate or locally sign for the trader account
        - Send a swap transaction on each fork
        - Expose per-block metadata for analysis
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

    # On-chain balances before swap
    wbnb_before = wbnb.functions.balanceOf(trader_onchain).call()
    usdt_before_token = usdt.functions.balanceOf(trader_onchain).call()
    print(
        f"[pancake-onchain-before] trader={trader_onchain} "
        f"wbnb={wbnb_before} usdt={usdt_before_token}"
    )

    client = PancakeSwapDemoExchangeClient(
        exchange_name="PancakeSwapV2",
        instrument="swap",
        state=state,
        params=params,
    )

    # Internal bot account snapshot (off-chain accounting)
    account = state.get_or_create_account("PancakeSwapV2", "swap")
    account.deposit("USDT", 1000.0)

    symbol = "DEMO_WBNBUSDT"
    quantity = float(os.getenv("DEMO_PANCAKE_TEST_QUANTITY", "0.01"))
    price_hint = float(os.getenv("DEMO_PANCAKE_TEST_PRICE_HINT", "100.0"))

    usdt_before = account.balances.get("USDT", 0.0)

    trade = await client.create_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price_hint,
    )

    assert trade.success is True
    assert trade.exchange == "PancakeSwapV2"
    assert trade.symbol == symbol

    meta = trade.metadata or {}
    assert "pancake_demo" in meta

    demo_meta = meta["pancake_demo"]
    block_offsets = demo_meta["block_offsets"]
    per_block_results = demo_meta["per_block_results"]

    assert block_offsets == [1, 2, 3]
    assert len(per_block_results) == 3

    for result in per_block_results:
        assert "fork_block" in result
        assert "tx_hash" in result
        assert "status" in result
        assert result["status"] in (0, 1)

    # Per-block balance changes and returns
    fill_prices: List[float] = []
    returns: List[float] = []

    for result in per_block_results:
        delta_in = result.get("delta_in")
        delta_out = result.get("delta_out")
        fill_price = result.get("fill_price")
        block_ret = result.get("return_vs_hint")

        if delta_in is None or delta_out is None or fill_price is None:
            continue

        print(
            f"[pancake-per-block] fork_block={result['fork_block']} "
            f"delta_in={delta_in} delta_out={delta_out} "
            f"fill_price={fill_price} return_vs_hint={block_ret}"
        )

        fill_prices.append(float(fill_price))
        if block_ret is not None:
            returns.append(float(block_ret))

    if fill_prices:
        avg_fill_price_meta = demo_meta.get("avg_fill_price")
        if avg_fill_price_meta is None:
            avg_fill_price_meta = sum(fill_prices) / len(fill_prices)
        print(f"[pancake-summary] avg_fill_price={avg_fill_price_meta}")

    if returns:
        avg_return_meta = demo_meta.get("avg_return_vs_hint")
        if avg_return_meta is None:
            avg_return_meta = sum(returns) / len(returns)
        print(f"[pancake-summary] avg_return_vs_hint={avg_return_meta}")

    # On-chain balances after swap
    wbnb_after = wbnb.functions.balanceOf(trader_onchain).call()
    usdt_after_token = usdt.functions.balanceOf(trader_onchain).call()
    print(
        f"[pancake-onchain-after] trader={trader_onchain} "
        f"wbnb={wbnb_after} usdt={usdt_after_token}"
    )

    print("\n=== Pancake demo metadata ===")
    meta_for_log = dict(demo_meta)
    meta_for_log.pop("account_snapshot", None)
    print(json.dumps(meta_for_log, indent=2))





@pytest.mark.skipif(
    bool(MISSING_ENVS),
    reason=(
        "Pancake demo environment variables are not fully configured. "
        f"Missing: {', '.join(MISSING_ENVS)}"
    ),
)
def test_pancake_demo_runs_swaps_on_three_forked_blocks() -> None:
    asyncio.run(_run_pancake_demo_test())


if __name__ == "__main__":
    asyncio.run(_run_pancake_demo_test())
