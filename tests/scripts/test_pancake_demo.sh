#!/usr/bin/env bash
set -euo pipefail

# Move to project root directory (from tests/scripts -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "[Pancake demo test runner]"
echo "  ROOT_DIR                        = ${ROOT_DIR}"

# Load .env if present (for secrets / overrides)
if [ -f "${ROOT_DIR}/.env" ]; then
  echo "[env] loading .env from ${ROOT_DIR}/.env"
  set -a
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.env"
  set +a
fi

# Forked BSC node (Anvil / Hardhat)
export DEMO_PANCAKE_FORK_RPC_URL="${DEMO_PANCAKE_FORK_RPC_URL:-http://127.0.0.1:8545}"
export DEMO_PANCAKE_UPSTREAM_RPC_URL="${DEMO_PANCAKE_UPSTREAM_RPC_URL:-https://bsc-dataseed.binance.org}"

# Universal Router + Permit2
export DEMO_PANCAKE_ROUTER_ADDRESS="${DEMO_PANCAKE_ROUTER_ADDRESS:-0xd9C500DfF816a1Da21A48A732d3498Bf09dc9AEB}"
export DEMO_PANCAKE_ROUTER_ABI_JSON="${DEMO_PANCAKE_ROUTER_ABI_JSON:-contracts/abi/UniversalRouter.json}"
export DEMO_PANCAKE_PERMIT2_ADDRESS="${DEMO_PANCAKE_PERMIT2_ADDRESS:-0x31c2F6fcFf4F8759b3Bd5Bf0e1084A055615c768}"

# Swap path: logical tokens (symbols or addresses)
export DEMO_PANCAKE_SWAP_PATH="${DEMO_PANCAKE_SWAP_PATH:-WBNB,USDT}"

# Trader private key for local signing on fork
# Default is Hardhat/Anvil test key (NOT a real secret).
export DEMO_PANCAKE_TRADER_PRIVATE_KEY="${DEMO_PANCAKE_TRADER_PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}"

# Market data hint (DEX price from MarketDataClient)
export MDC_HINT_EXCHANGE="${MDC_HINT_EXCHANGE:-PancakeSwapV3}"
export MDC_HINT_SYMBOL="${MDC_HINT_SYMBOL:-USDTWBNB}"
export MDC_HINT_INSTRUMENT="${MDC_HINT_INSTRUMENT:-swap}"
export MDC_HINT_TIMEOUT_SEC="${MDC_HINT_TIMEOUT_SEC:-10}"
export MDC_HINT_USE_JS="${MDC_HINT_USE_JS:-0}"

# Fork engine: "anvil" or "hardhat"
export DEMO_PANCAKE_FORK_ENGINE="${DEMO_PANCAKE_FORK_ENGINE:-anvil}"

# USDT whale for funding (impersonated on fork)
export DEMO_PANCAKE_USDT_WHALE="${DEMO_PANCAKE_USDT_WHALE:-0xba99D0A2016F43dA2c8AeB581b6076C8b487401A}"

echo "  DEMO_PANCAKE_FORK_RPC_URL       = ${DEMO_PANCAKE_FORK_RPC_URL}"
echo "  DEMO_PANCAKE_UPSTREAM_RPC_URL   = ${DEMO_PANCAKE_UPSTREAM_RPC_URL}"
echo "  DEMO_PANCAKE_ROUTER_ADDRESS     = ${DEMO_PANCAKE_ROUTER_ADDRESS}"
echo "  DEMO_PANCAKE_ROUTER_ABI_JSON    = ${DEMO_PANCAKE_ROUTER_ABI_JSON}"
echo "  DEMO_PANCAKE_SWAP_PATH          = ${DEMO_PANCAKE_SWAP_PATH}"
echo "  DEMO_PANCAKE_FORK_ENGINE        = ${DEMO_PANCAKE_FORK_ENGINE}"
echo "  DEMO_PANCAKE_PERMIT2_ADDRESS    = ${DEMO_PANCAKE_PERMIT2_ADDRESS}"
echo "  DEMO_PANCAKE_USDT_WHALE         = ${DEMO_PANCAKE_USDT_WHALE}"

if [ -n "${DEMO_PANCAKE_TRADER_PRIVATE_KEY:-}" ]; then
  echo "  DEMO_PANCAKE_TRADER_PRIVATE_KEY = ***hardhat_default_or_env***"
else
  echo "  DEMO_PANCAKE_TRADER_PRIVATE_KEY = (not set)"
fi

echo "  MDC_HINT_EXCHANGE               = ${MDC_HINT_EXCHANGE}"
echo "  MDC_HINT_SYMBOL                 = ${MDC_HINT_SYMBOL}"
echo "  MDC_HINT_INSTRUMENT             = ${MDC_HINT_INSTRUMENT}"
echo "  MDC_HINT_TIMEOUT_SEC            = ${MDC_HINT_TIMEOUT_SEC}"
echo "  MDC_HINT_USE_JS                 = ${MDC_HINT_USE_JS}"

pytest -s -rs tests/test_pancake_demo_real_fork.py
