#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[Binance demo test runner]"
echo "  ROOT_DIR                        = ${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  echo "[env] loading .env from ${ROOT_DIR}/.env"
  set -a
  # shellcheck source=/dev/null
  . "${ROOT_DIR}/.env"
  set +a
fi

# NATS / Market data configuration
export NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"

# Market data hint envs for Binance
export MDC_HINT_EXCHANGE="${MDC_HINT_EXCHANGE:-Binance}"
export MDC_HINT_SYMBOL="${MDC_HINT_SYMBOL:-BNBUSDT}"
# You can choose "spot" or "perpetual" depending on how your MarketDataClient is configured.
export MDC_HINT_INSTRUMENT="${MDC_HINT_INSTRUMENT:-perpetual}"
export MDC_HINT_TIMEOUT_SEC="${MDC_HINT_TIMEOUT_SEC:-20}"
# Use JetStream=0 by default; set to 1 if you want to consume from JS.
export MDC_HINT_USE_JS="${MDC_HINT_USE_JS:-0}"

# Demo trading parameters
export DEMO_BINANCE_SYMBOL="${DEMO_BINANCE_SYMBOL:-BNBUSDT}"
export DEMO_BINANCE_INSTRUMENT="${DEMO_BINANCE_INSTRUMENT:-perpetual}"
export DEMO_BINANCE_TEST_QUANTITY="${DEMO_BINANCE_TEST_QUANTITY:-0.1}"
export DEMO_BINANCE_TEST_PRICE_HINT="${DEMO_BINANCE_TEST_PRICE_HINT:-300.0}"
export DEMO_BINANCE_STARTING_QUOTE_BALANCE="${DEMO_BINANCE_STARTING_QUOTE_BALANCE:-10000.0}"
export DEMO_BINANCE_FEE_RATE="${DEMO_BINANCE_FEE_RATE:-0.0004}"
export DEMO_BINANCE_DEFAULT_SLIPPAGE_BPS="${DEMO_BINANCE_DEFAULT_SLIPPAGE_BPS:-1.0}"

export DEMO_BINANCE_SPOT_API_KEY="${DEMO_BINANCE_SPOT_API_KEY:-}"
export DEMO_BINANCE_API_SECRET="${DEMO_BINANCE_API_SECRET:-}"
export DEMO_BINANCE_BASE_URL="${DEMO_BINANCE_BASE_URL:-https://testnet.binance.vision}"
export DEMO_BINANCE_RECV_WINDOW_MS="${DEMO_BINANCE_RECV_WINDOW_MS:-5000}"
export DEMO_BINANCE_HTTP_TIMEOUT_SEC="${DEMO_BINANCE_HTTP_TIMEOUT_SEC:-10.0}"
export DEMO_BINANCE_USE_TESTNET_EXECUTION="${DEMO_BINANCE_USE_TESTNET_EXECUTION:-0}"

# Optional: Spot Testnet credentials (not used by the current demo client,
# but kept for future wiring)
if [[ -n "${DEMO_BINANCE_SPOT_API_KEY:-}" && -n "${DEMO_BINANCE_API_SECRET:-}" ]]; then
  echo "[Binance demo] Testnet API key/secret detected → will try testnet REST calls"
else
  echo "[Binance demo] DEMO_BINANCE_SPOT_API_KEY/SECRET not set → running in pure in-memory mode"
fi

export DEMO_BINANCE_BASE_URL="${DEMO_BINANCE_BASE_URL:-https://testnet.binance.vision}"

cd "${ROOT_DIR}"

pytest -q tests/test_binance_demo.py -s
