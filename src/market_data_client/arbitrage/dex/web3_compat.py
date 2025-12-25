from __future__ import annotations

from typing import Any


def disable_poa_extra_data_validation() -> None:
    try:
        import web3._utils.method_formatters as mf

        if hasattr(mf, "BLOCK_FORMATTERS") and "extraData" in mf.BLOCK_FORMATTERS:

            def _keep_extra_data(value: Any) -> Any:
                return value

            mf.BLOCK_FORMATTERS["extraData"] = _keep_extra_data
    except Exception:
        return
