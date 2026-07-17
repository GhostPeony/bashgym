"""Compatibility helpers for BashGym's supported Python versions."""

from datetime import timezone

# ``datetime.UTC`` was added in Python 3.11. ``timezone.utc`` is the same
# canonical fixed-offset timezone and keeps the public Python 3.10 contract.
UTC = timezone.utc
