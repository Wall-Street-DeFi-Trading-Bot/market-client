import re
from typing import Iterable, Tuple, List

def norm_symbol(sym: str) -> str:
    return re.sub(r"[/\-\s]", "", sym).upper()

def build_symbol_splitter(assets: Iterable[str]):
    # Longest suffix wins (e.g., BTCB before BTC)
    quotes: List[str] = sorted({a.upper() for a in assets}, key=len, reverse=True)

    def split(symbol: str) -> Tuple[str, str]:
        s = norm_symbol(symbol)
        for q in quotes:
            if s.endswith(q) and len(s) > len(q):
                return s[: -len(q)], q
        raise ValueError(f"Cannot split symbol={symbol}. quotes={quotes}")

    return split
