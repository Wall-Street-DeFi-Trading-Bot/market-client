from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from eth_account import Account
from web3 import Web3
from web3.types import TxParams

from market_data_client.arbitrage.pairs import resolve_base_quote


# Universal Router command bytes
V3_SWAP_EXACT_IN = 0x00
V2_SWAP_EXACT_IN = 0x08


TOKEN_ADDRESS_MAP: Dict[str, str] = {
    # Wrapped/native aliases
    "WBNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "BNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "WETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    "ETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    # Stablecoins
    "USDT": "0x55d398326f99059fF775485246999027B3197955",
    "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
    # Others
    "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
    "BTCB": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",
    "TWT": "0x4B0F1812e5Df2A09796481Ff14017e6005508003",
    "SFP": "0xD41FDb03Ba84762dD66a0af1a6C8540FF1ba5dfb",
}


def build_swap_tx_router_v3(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
    price: Optional[float] = None,
    quote_asset: str = "USDT",
    swap_kind: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return _build_swap_tx_universal_router(
        web3=web3,
        symbol=symbol,
        quantity=quantity,
        side=side,
        price=price,
        quote_asset=quote_asset,
        router_mode="v3",
    )


def build_swap_tx_router_v2(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
    price: Optional[float] = None,
    quote_asset: str = "USDT",
    swap_kind: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return _build_swap_tx_universal_router(
        web3=web3,
        symbol=symbol,
        quantity=quantity,
        side=side,
        price=price,
        quote_asset=quote_asset,
        router_mode="v2",
    )


# Backward compatible name (keeps old imports working)
def build_swap_tx_router(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
    price: Optional[float] = None,
    quote_asset: str = "USDT",
    swap_kind: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Dispatcher that selects v2/v3 by swap_kind.
    Accepts extra kwargs for compatibility with caller.
    """
    mode = "v3"
    if swap_kind is not None:
        sk = str(swap_kind).lower()
        if "v2" in sk or sk in ("2", "uni_v2", "pancake_v2"):
            mode = "v2"
        elif "v3" in sk or sk in ("3", "uni_v3", "pancake_v3"):
            mode = "v3"

    return _build_swap_tx_universal_router(
        web3=web3,
        symbol=symbol,
        quantity=quantity,
        side=side,
        price=price,
        quote_asset=quote_asset,
        router_mode=mode,
    )



def _build_swap_tx_universal_router(
    web3: Web3,
    symbol: str,
    quantity: float,
    side: str,
    price: Optional[float],
    quote_asset: str,
    router_mode: str,  # "v2" | "v3"
) -> Dict[str, Any]:
    router_address = Web3.to_checksum_address(os.environ["DEMO_PANCAKE_ROUTER_ADDRESS"].strip())
    priv_key = os.environ["DEMO_PANCAKE_TRADER_PRIVATE_KEY"].strip()
    trader = Web3.to_checksum_address(Account.from_key(priv_key).address)

    erc20_abi, permit2_abi, router_abi = _load_abis()
    router = web3.eth.contract(address=router_address, abi=router_abi)

    permit2_address = _read_permit2_address(router)
    if permit2_address is None:
        permit2_address = Web3.to_checksum_address(os.environ["DEMO_PANCAKE_PERMIT2_ADDRESS"].strip())

    # Quick runtime env sanity (helps catch "I set it but it's not loaded" cases)
    # print("[env] DEMO_PANCAKE_USDT_WHALE =", os.getenv("DEMO_PANCAKE_USDT_WHALE"))

    print(f"[pancake-debug] router={router_address} permit2={permit2_address} mode={router_mode}")

    side_upper = side.upper()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported side: {side}")

    clean_symbol = _strip_demo_prefix(symbol)
    base_sym, quote_sym = resolve_base_quote(clean_symbol)
    base_sym = _strip_demo_prefix(base_sym)
    quote_sym = _strip_demo_prefix(quote_sym) if quote_sym else _strip_demo_prefix(quote_asset)

    # BUY: spend quote -> receive base
    # SELL: spend base  -> receive quote
    if side_upper == "BUY":
        token_in = _resolve_token(quote_sym)
        token_out = _resolve_token(base_sym)
    else:
        token_in = _resolve_token(base_sym)
        token_out = _resolve_token(quote_sym)

    token_in_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
    token_out_contract = web3.eth.contract(address=token_out, abi=erc20_abi)

    decimals_in = _read_decimals_safe(token_in_contract)
    decimals_out = _read_decimals_safe(token_out_contract)

    if side_upper == "BUY":
        if price is None or float(price) <= 0:
            raise ValueError("BUY requires `price` (quote per base).")
        quote_amount = float(quantity) * float(price)
        amount_in = int(quote_amount * (10 ** decimals_in))
    else:
        amount_in = int(float(quantity) * (10 ** decimals_in))

    gas_limit = int(os.getenv("DEMO_PANCAKE_GAS_LIMIT", "500000"))
    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    # Ensure fork has enough gas + input tokens
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

    amount_out_min = 0
    chain_ts = int(web3.eth.get_block("latest")["timestamp"])
    deadline = chain_ts + int(os.getenv("DEMO_PANCAKE_DEADLINE_SEC", "600"))

    if router_mode == "v3":
        fee = int(os.getenv("DEMO_PANCAKE_V3_FEE", "2500"))
        if fee < 0 or fee > 2**24 - 1:
            raise ValueError(f"Invalid V3 fee: {fee}")
        fee_bytes = fee.to_bytes(3, byteorder="big")
        path_bytes = bytes.fromhex(token_in[2:]) + fee_bytes + bytes.fromhex(token_out[2:])

        params_types = ["address", "uint256", "uint256", "bytes", "bool"]
        params_values = [trader, amount_in, amount_out_min, path_bytes, True]
        input_bytes = web3.codec.encode(params_types, params_values)
        commands = bytes([V3_SWAP_EXACT_IN])

    elif router_mode == "v2":
        path = [token_in, token_out]  # address[]
        params_types = ["address", "uint256", "uint256", "address[]", "bool"]
        params_values = [trader, amount_in, amount_out_min, path, True]
        input_bytes = web3.codec.encode(params_types, params_values)
        commands = bytes([V2_SWAP_EXACT_IN])

    else:
        raise ValueError(f"Unknown router_mode: {router_mode}")

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
    out["_demo_permit2"] = permit2_address
    out["_demo_decimals_in"] = decimals_in
    out["_demo_decimals_out"] = decimals_out
    out["_demo_router_mode"] = router_mode
    return out


def _read_permit2_address(router_contract: Any) -> Optional[str]:
    """
    Read Permit2 address from Universal Router.
    Tries common function names: PERMIT2() / permit2().
    """
    for fn_name in ("PERMIT2", "permit2"):
        try:
            fn = getattr(router_contract.functions, fn_name)
            addr = fn().call()
            return Web3.to_checksum_address(addr)
        except Exception:
            continue
    return None

def _read_decimals_safe(token_contract: Any) -> int:
    try:
        return int(token_contract.functions.decimals().call())
    except Exception:
        return int(os.getenv("DEMO_PANCAKE_DECIMALS_FALLBACK", "18") or "18")


def _resolve_token(token: str) -> str:
    t = _strip_demo_prefix(token)
    if t.startswith("0x") or t.startswith("0X"):
        return Web3.to_checksum_address(t)

    addr = TOKEN_ADDRESS_MAP.get(t.upper())
    if not addr:
        raise ValueError(f"Unknown token symbol: {token}")
    return Web3.to_checksum_address(addr)


def _load_abi_json(path: Path) -> List[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "abi" in raw:
        raw = raw["abi"]
    if not isinstance(raw, list):
        raise TypeError(f"ABI json must be a list or {{'abi': list}}: {path}")
    return raw


def _load_abis() -> Tuple[List[Any], List[Any], List[Any]]:
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

    erc20_abi = _load_abi_json(erc20_path)
    permit2_abi = _load_abi_json(permit2_path)
    router_abi = _load_abi_json(universal_router_path)
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
        "Set DEMO_PANCAKE_ABI=contracts/abi (or an absolute path)."
    )

def _strip_demo_prefix(s: str) -> str:
    s2 = (s or "").strip()
    if s2.upper().startswith("DEMO_"):
        return s2[5:]
    return s2


def _search_upwards_for_abi(start: Path) -> Optional[Path]:
    cur = start if start.is_dir() else start.parent
    for p in [cur] + list(cur.parents):
        cand = p / "contracts" / "abi"
        if cand.is_dir():
            return cand
    return None


def _send_signed_and_wait(web3: Web3, tx: TxParams, private_key: str) -> None:
    """
    Send a locally-signed tx and ensure receipt.status == 1.
    """
    signed = Account.sign_transaction(tx, private_key)
    raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
    tx_hash = web3.eth.send_raw_transaction(raw_tx)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    status = int(receipt.get("status", 0)) if isinstance(receipt, dict) else int(getattr(receipt, "status", 0))
    if status != 1:
        raise RuntimeError(f"tx reverted: hash={tx_hash.hex()}")


def _reverse_lookup_symbol_by_address(token_addr: str) -> Optional[str]:
    token_addr = Web3.to_checksum_address(token_addr)
    for sym, addr in TOKEN_ADDRESS_MAP.items():
        try:
            if Web3.to_checksum_address(addr) == token_addr:
                return sym.upper()
        except Exception:
            continue
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

    trader = Web3.to_checksum_address(trader)
    token_in = Web3.to_checksum_address(token_in)

    # Ensure native BNB for gas
    target_native_bnb = float(os.getenv("DEMO_PANCAKE_NATIVE_BALANCE_BNB", "1"))
    target_native_wei = web3.to_wei(target_native_bnb, "ether")

    try:
        native_balance = web3.eth.get_balance(trader)
    except Exception:
        native_balance = 0

    if native_balance < target_native_wei:
        for method in ("anvil_setBalance", "hardhat_setBalance"):
            try:
                provider.make_request(method, [trader, hex(target_native_wei)])
                break
            except Exception:
                pass

    wbnb_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["WBNB"])
    usdt_addr = Web3.to_checksum_address(TOKEN_ADDRESS_MAP["USDT"])

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    token_in_contract = web3.eth.contract(address=token_in, abi=erc20_abi)

    # ----- WBNB: wrap native BNB -> WBNB
    if token_in == wbnb_addr:
        bal_in_raw = int(token_in_contract.functions.balanceOf(trader).call())
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
        raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
        tx_hash = web3.eth.send_raw_transaction(raw_tx)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if int(receipt.get("status", 0)) != 1:
            raise RuntimeError("WBNB deposit() reverted")
        return

    # ----- ERC20 funding via whale impersonation
    # USDT uses DEMO_PANCAKE_USDT_WHALE, others use DEMO_PANCAKE_<SYMBOL>_WHALE
    if token_in == usdt_addr:
        whale_env_name = "DEMO_PANCAKE_USDT_WHALE"
    else:
        sym = _reverse_lookup_symbol_by_address(token_in)
        whale_env_name = f"DEMO_PANCAKE_{sym}_WHALE" if sym else "DEMO_PANCAKE_TOKEN_WHALE"

    whale_raw = (os.getenv(whale_env_name) or "").strip()
    if not whale_raw:
        raise RuntimeError(
            f"Token-in funding requires a whale on fork. Set {whale_env_name}=0x... "
            f"(token_in={token_in})."
        )

    whale = Web3.to_checksum_address(whale_raw)

    # Give whale gas
    for method in ("anvil_setBalance", "hardhat_setBalance"):
        try:
            provider.make_request(method, [whale, hex(target_native_wei)])
            break
        except Exception:
            pass

    bal_trader = int(token_in_contract.functions.balanceOf(trader).call())
    if bal_trader >= amount_in:
        return

    need = amount_in - bal_trader

    # Fail fast: ensure whale has enough token
    bal_whale = int(token_in_contract.functions.balanceOf(whale).call())
    if bal_whale < need:
        raise RuntimeError(
            f"Whale has insufficient token balance: env={whale_env_name} whale={whale} "
            f"have={bal_whale} need={need} token={token_in}"
        )

    # impersonate
    impersonated = False
    for method in ("anvil_impersonateAccount", "hardhat_impersonateAccount"):
        try:
            provider.make_request(method, [whale])
            impersonated = True
            break
        except Exception:
            pass

    if not impersonated:
        raise RuntimeError(f"Failed to impersonate whale on fork: {whale}")

    try:
        nonce_whale = web3.eth.get_transaction_count(whale)
        transfer_tx = token_in_contract.functions.transfer(trader, need).build_transaction(
            {"from": whale, "gas": 250000, "gasPrice": gas_price, "nonce": nonce_whale}
        )

        tx_hash = web3.eth.send_transaction(transfer_tx)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if int(receipt.get("status", 0)) != 1:
            raise RuntimeError("Whale transfer reverted")

        # Verify
        bal2 = int(token_in_contract.functions.balanceOf(trader).call())
        if bal2 < amount_in:
            raise RuntimeError(f"Funding insufficient after transfer: have={bal2} need={amount_in}")

    finally:
        for method in ("anvil_stopImpersonatingAccount", "hardhat_stopImpersonatingAccount"):
            try:
                provider.make_request(method, [whale])
                break
            except Exception:
                pass


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
    """
    Ensure:
      1) ERC20 allowance(trader -> Permit2) >= amount_in
      2) Permit2 allowance(trader, token_in, router) is set and not expired

    If Permit2 allowance is missing/expired, Universal Router swap reverts with:
      Permit2.AllowanceExpired(uint256) selector 0xd81b2f2e
    """

    # --- Normalize all addresses (this matters a lot) ---
    trader = Web3.to_checksum_address(trader)
    token_in = Web3.to_checksum_address(token_in)
    permit2_address = Web3.to_checksum_address(permit2_address)
    router_address = Web3.to_checksum_address(router_address)

    token_contract = web3.eth.contract(address=token_in, abi=erc20_abi)
    permit2 = web3.eth.contract(address=permit2_address, abi=permit2_abi)

    gas_price_gwei = int(os.getenv("DEMO_PANCAKE_GAS_PRICE_GWEI", "3"))
    gas_price = web3.to_wei(gas_price_gwei, "gwei")

    # Use pending nonce to be safe under forks
    def _nonce() -> int:
        try:
            return int(web3.eth.get_transaction_count(trader, "pending"))
        except Exception:
            return int(web3.eth.get_transaction_count(trader))

    max_uint256 = (1 << 256) - 1
    max_uint160 = (1 << 160) - 1
    max_uint48 = (1 << 48) - 1  # uint48 max

    chain_ts = int(web3.eth.get_block("latest")["timestamp"])
    desired_exp = max_uint48  # "never" style (works well on forks)

    # -------- (A) ERC20 approve -> Permit2 --------
    try:
        allowance_to_permit2 = int(token_contract.functions.allowance(trader, permit2_address).call())
    except Exception:
        allowance_to_permit2 = 0

    if allowance_to_permit2 < amount_in:
        # Some tokens require approve(0) first (USDT-like)
        if allowance_to_permit2 != 0:
            try:
                tx0 = token_contract.functions.approve(permit2_address, 0).build_transaction(
                    {"from": trader, "gas": 200000, "gasPrice": gas_price, "nonce": _nonce()}
                )
                _send_signed_and_wait(web3, tx0, private_key)
            except Exception:
                pass

        tx1 = token_contract.functions.approve(permit2_address, max_uint256).build_transaction(
            {"from": trader, "gas": 200000, "gasPrice": gas_price, "nonce": _nonce()}
        )
        _send_signed_and_wait(web3, tx1, private_key)

        # Re-check ERC20 allowance
        allowance_to_permit2 = int(token_contract.functions.allowance(trader, permit2_address).call())

    # -------- (B) Permit2 approve(token, spender=router, amount, expiration) --------
    try:
        res = permit2.functions.allowance(trader, token_in, router_address).call()
        allowed_amount = int(res[0])
        expiration = int(res[1])
    except Exception:
        allowed_amount = 0
        expiration = 0

    print(
        "[permit2-before] "
        f"token={token_in} router={router_address} "
        f"allowed={allowed_amount} exp={expiration} now={chain_ts} "
        f"amount_in={amount_in} erc20_allow_to_permit2={allowance_to_permit2}"
    )

    need_permit2 = (allowed_amount < amount_in) or (expiration <= chain_ts)

    if need_permit2:
        tx2 = permit2.functions.approve(token_in, router_address, max_uint160, desired_exp).build_transaction(
            {"from": trader, "gas": 250000, "gasPrice": gas_price, "nonce": _nonce()}
        )
        _send_signed_and_wait(web3, tx2, private_key)

        # Re-check and fail fast
        res2 = permit2.functions.allowance(trader, token_in, router_address).call()
        allowed2 = int(res2[0])
        exp2 = int(res2[1])
        now2 = int(web3.eth.get_block("latest")["timestamp"])

        print(
            "[permit2-after] "
            f"token={token_in} router={router_address} "
            f"allowed={allowed2} exp={exp2} now={now2} amount_in={amount_in}"
        )

        if allowed2 < amount_in or exp2 <= now2:
            raise RuntimeError(
                f"Permit2 allowance invalid after approve: allowed={allowed2} exp={exp2} now={now2} amount_in={amount_in}"
            )
