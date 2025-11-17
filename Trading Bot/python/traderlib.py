"""
DEPRECATED: Alpaca-based trader module

This file was part of the original Alpaca-based implementation. The active
project has migrated to Binance and optimized modules. The original Alpaca
implementation has been moved to `legacy_alpaca/traderlib_alpaca.py` for
archival purposes.

Any attempt to import or use this module will raise an ImportError so that
the codebase fails fast and you notice leftover references.
"""

raise ImportError(
    "traderlib.py has been deprecated and moved to 'legacy_alpaca/traderlib_alpaca.py'. "
    "Use the Binance modules (e.g. binance_connector.py, binance_bot.py) instead."
)
