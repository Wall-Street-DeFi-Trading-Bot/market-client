from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eth_account import Account
from web3 import Web3
from web3.types import TxParams


V3_SWAP_EXACT_IN = 0x00

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


def build_swap_tx_router(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
) -> Dict[str, Any]:
    router_address = Web3.to_checksum_address(os.environ["DEMO_PANCAKE_ROUTER_ADDRESS"].strip())
    permit2_address = Web3.to_checksum_address(os.environ["DEMO_PANCAKE_PERMIT2_ADDRESS"].strip())

    erc20_abi, permit2_abi, router_abi = _load_abis()

    raw_path = os.environ["DEMO_PANCAKE_SWAP_PATH"]
    raw_tokens: List[str] = [p.strip() for p in raw_path.split(",") if p.strip()]
    if len(raw_tokens) < 2:
        raise ValueError(f"DEMO_PANCAKE_SWAP_PATH must contain at least 2 tokens, got: {raw_tokens}")

    side_upper = side.upper()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported side: {side}")

    path_addresses = [_resolve_token(t) for t in raw_tokens]
    if side_upper == "BUY":
        token_in = path_addresses[0]
        token_out = path_addresses[-1]
    else:
        token_in = path_addresses[-1]
        token_out = path_addresses[0]

    priv_key = os.environ["DEMO_PANCAKE_TRADER_PRIVATE_KEY"].strip()
    trader = Web3.to_checksum_address(Account.from_key(priv_key).address)

    token_in_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
    try:
        decimals_in = int(token_in_contract.functions.decimals().call())
    except Exception:
        decimals_in = int(os.getenv("DEMO_PANCAKE_DECIMALS_IN", "18") or "18")

    amount_in = int(float(quantity) * (10**decimals_in))

    gas_limit = int(os.getenv("DEMO_PANCAKE_GAS_LIMIT", "500000"))
    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    _fund_test_tokens(
        web3=web3,
        trader=trader,
        token_in=token_in,
        amount_in=amount_in,
        private_key=priv_key,
        erc20_abi=erc20_abi,
    )

    _ensure_allowance(
        web3=web3,
        trader=trader,
        token_in=token_in,
        permit2_address=permit2_address,
        router_address=router_address,
        amount_in=amount_in,
        private_key=priv_key,
        erc20_abi=erc20_abi,
        permit2_abi=permit2_abi,
    )

    fee = int(os.getenv("DEMO_PANCAKE_V3_FEE", "2500"))
    if fee < 0 or fee > 2**24 - 1:
        raise ValueError(f"Invalid V3 fee: {fee}")
    fee_bytes = fee.to_bytes(3, byteorder="big")

    path_bytes = bytes.fromhex(token_in[2:]) + fee_bytes + bytes.fromhex(token_out[2:])

    amount_out_min = 0
    deadline = int(time.time()) + int(os.getenv("DEMO_PANCAKE_DEADLINE_SEC", "600"))

    params_types = ["address", "uint256", "uint256", "bytes", "bool"]
    params_values = [trader, amount_in, amount_out_min, path_bytes, True]
    input_bytes = web3.codec.encode(params_types, params_values)

    commands = bytes([V3_SWAP_EXACT_IN])

    router = web3.eth.contract(address=router_address, abi=router_abi)
    fn = router.functions.execute(commands, [input_bytes], deadline)
    data = fn._encode_transaction_data()

    tx: TxParams = {
        "to": router_address,
        "from": trader,
        "data": data,
        "nonce": web3.eth.get_transaction_count(trader),
        "gas": gas_limit,
        "gasPrice": gas_price,
        "value": 0,
    }

    out: Dict[str, Any] = dict(tx)
    out["_demo_token_in"] = token_in
    out["_demo_token_out"] = token_out
    out["_demo_amount_in_wei"] = amount_in
    out["_demo_spenders"] = [permit2_address, router_address]
    return out


def _resolve_token(token: str) -> str:
    if token.startswith("0x") or token.startswith("0X"):
        return Web3.to_checksum_address(token)

    addr = TOKEN_ADDRESS_MAP.get(token.upper())
    if not addr:
        raise ValueError(f"Unknown token symbol: {token}")
    return Web3.to_checksum_address(addr)


def _load_abis() -> tuple[list, list, list]:
    abi_dir = _find_abi_dir()

    erc20_path = abi_dir / "ERC20.json"
    permit2_path = abi_dir / "Permit2.json"
    universal_router_path = abi_dir / "UniversalRouter.json"

    if not erc20_path.exists():
        raise FileNotFoundError(f"Missing ABI: {erc20_path}")
    if not permit2_path.exists():
        raise FileNotFoundError(f"Missing ABI: {permit2_path}")
    if not universal_router_path.exists():
        raise FileNotFoundError(f"Missing ABI: {universal_router_path}")

    erc20_abi = json.loads(erc20_path.read_text(encoding="utf-8"))
    permit2_abi = json.loads(permit2_path.read_text(encoding="utf-8"))
    router_abi = json.loads(universal_router_path.read_text(encoding="utf-8"))

    return erc20_abi, permit2_abi, router_abi


def _find_abi_dir() -> Path:
    env_dir = os.getenv("DEMO_PANCAKE_ABI")
    if env_dir:
        p = Path(env_dir).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    found = _search_upwards_for_abi(Path.cwd())
    if found is not None:
        return found

    found = _search_upwards_for_abi(Path(__file__).resolve())
    if found is not None:
        return found

    raise FileNotFoundError(
        "Could not locate contracts/abi. "
        "Set DEMO_PANCAKE_ROUTER_ABI_JSON=contracts/abi (or an absolute path)."
    )


def _search_upwards_for_abi(start: Path) -> Optional[Path]:
    cur = start if start.is_dir() else start.parent
    for p in [cur] + list(cur.parents):
        cand = p / "contracts" / "abi"
        if cand.is_dir():
            return cand
    return None


def _fund_test_tokens(
    web3: Web3,
    trader: str,
    token_in: str,
    amount_in: int,
    private_key: str,
    erc20_abi: list,
) -> None:
    provider = web3.provider
    if provider is None:
        return

    target_native_bnb = float(os.getenv("DEMO_PANCAKE_NATIVE_BALANCE_BNB", "1"))
    target_native_wei = web3.to_wei(target_native_bnb, "ether")

    try:
        native_balance = web3.eth.get_balance(trader)
    except Exception:
        native_balance = 0

    if native_balance < target_native_wei:
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

    if token_in == wbnb_addr:
        token_in_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
        try:
            bal_in_raw = int(token_in_contract.functions.balanceOf(trader).call())
        except Exception:
            bal_in_raw = 0

        if bal_in_raw >= amount_in:
            return

        wrap_amount = amount_in - bal_in_raw
        nonce = web3.eth.get_transaction_count(trader)

        deposit_tx: TxParams = {
            "to": token_in,
            "from": trader,
            "value": wrap_amount,
            "data": "0xd0e30db0",
            "gas": 200000,
            "gasPrice": gas_price,
            "nonce": nonce,
        }

        signed = Account.sign_transaction(deposit_tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)
        return

    if token_in == usdt_addr:
        usdt_whale = os.getenv("DEMO_PANCAKE_USDT_WHALE")
        if not usdt_whale:
            return

        whale = Web3.to_checksum_address(usdt_whale)

        try:
            provider.make_request("anvil_setBalance", [whale, hex(target_native_wei)])
        except Exception:
            try:
                provider.make_request("hardhat_setBalance", [whale, hex(target_native_wei)])
            except Exception:
                pass

        token_in_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
        try:
            bal_in_raw = int(token_in_contract.functions.balanceOf(trader).call())
        except Exception:
            bal_in_raw = 0

        if bal_in_raw >= amount_in:
            return

        transfer_amount = amount_in - bal_in_raw
        nonce_whale = web3.eth.get_transaction_count(whale)

        fork_engine = os.getenv("DEMO_PANCAKE_FORK_ENGINE", "anvil")
        try:
            if fork_engine == "anvil":
                provider.make_request("anvil_impersonateAccount", [whale])
            elif fork_engine == "hardhat":
                provider.make_request("hardhat_impersonateAccount", [whale])
        except Exception:
            return

        transfer_tx = token_in_contract.functions.transfer(trader, transfer_amount).build_transaction(
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


def _ensure_allowance(
    web3: Web3,
    trader: str,
    token_in: str,
    permit2_address: str,
    router_address: str,
    amount_in: int,
    private_key: str,
    erc20_abi: list,
    permit2_abi: list,
) -> None:
    token_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
    permit2 = web3.eth.contract(address=permit2_address, abi=permit2_abi)

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    nonce = web3.eth.get_transaction_count(trader)
    max_uint256 = 2**256 - 1
    max_uint160 = 2**160 - 1

    now_ts = int(time.time())
    max_expiration = 2**48 - 1
    desired_exp = min(max_expiration, now_ts + 10 * 365 * 24 * 60 * 60)

    try:
        allowance_permit2 = int(token_contract.functions.allowance(trader, permit2_address).call())
    except Exception:
        allowance_permit2 = 0

    if allowance_permit2 < amount_in:
        tx = token_contract.functions.approve(permit2_address, max_uint256).build_transaction(
            {"from": trader, "gas": 200000, "gasPrice": gas_price, "nonce": nonce}
        )
        nonce += 1
        signed = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)

    try:
        allowance_router = int(token_contract.functions.allowance(trader, router_address).call())
    except Exception:
        allowance_router = 0

    if allowance_router < amount_in:
        tx = token_contract.functions.approve(router_address, max_uint256).build_transaction(
            {"from": trader, "gas": 200000, "gasPrice": gas_price, "nonce": nonce}
        )
        nonce += 1
        signed = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)

    try:
        res = permit2.functions.allowance(trader, token_in, router_address).call()
        allowed_amount = int(res[0])
        expiration = int(res[1])
    except Exception:
        allowed_amount = 0
        expiration = 0

    need_permit2 = allowed_amount < amount_in or expiration <= now_ts
    if need_permit2:
        tx = permit2.functions.approve(token_in, router_address, max_uint160, desired_exp).build_transaction(
            {"from": trader, "gas": 200000, "gasPrice": gas_price, "nonce": nonce}
        )
        signed = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)
