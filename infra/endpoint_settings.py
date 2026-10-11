"""Validated numeric settings for the Modal/SGLang scoring endpoint.

This module does not import ``modal`` so the validation can be unit-tested
without Modal installed. ``deploy_endpoint.py`` reads its numeric
``SCORING_*`` variables through ``load_endpoint_settings`` so an out-of-range
value fails when the deployment is loaded instead of reaching
``sglang.launch_server`` or the Modal autoscaler.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

# SGLang treats --chunked-prefill-size -1 as "disable chunked prefill".
CHUNKED_PREFILL_DISABLED = -1

_POSITIVE_INT = "an integer >= 1"
_CHUNKED_PREFILL = "an integer >= 1, or -1 to disable chunked prefill"
_FRACTION = "a number in (0, 1]"


@dataclass(frozen=True)
class EndpointSettings:
    tensor_parallel: int = 1
    max_containers: int = 3
    mem_fraction_static: float = 0.75
    chunked_prefill_size: int = 4096
    max_running_requests: int = 1


def _read_int(
    environ: Mapping[str, str], name: str, default: int, accepted: str
) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be {accepted}, got {raw!r}") from None


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = _read_int(environ, name, default, _POSITIVE_INT)
    if value < 1:
        raise ValueError(f"{name} must be {_POSITIVE_INT}, got {value}")
    return value


def _chunked_prefill(environ: Mapping[str, str], name: str, default: int) -> int:
    value = _read_int(environ, name, default, _CHUNKED_PREFILL)
    if value != CHUNKED_PREFILL_DISABLED and value < 1:
        raise ValueError(f"{name} must be {_CHUNKED_PREFILL}, got {value}")
    return value


def _fraction(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be {_FRACTION}, got {raw!r}") from None
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be {_FRACTION}, got {raw!r}")
    return value


def load_endpoint_settings(
    environ: Mapping[str, str],
    defaults: EndpointSettings | None = None,
) -> EndpointSettings:
    """Read the endpoint's numeric settings from ``environ``.

    Unset variables keep the value from ``defaults``. Raises ``ValueError``
    naming the variable and its accepted range for a non-numeric,
    non-finite or out-of-range value.
    """
    if defaults is None:
        defaults = EndpointSettings()
    return EndpointSettings(
        tensor_parallel=_positive_int(
            environ, "SCORING_TP", defaults.tensor_parallel
        ),
        max_containers=_positive_int(
            environ, "SCORING_MAX_CONTAINERS", defaults.max_containers
        ),
        mem_fraction_static=_fraction(
            environ, "SCORING_MEM_FRACTION", defaults.mem_fraction_static
        ),
        chunked_prefill_size=_chunked_prefill(
            environ, "SCORING_CHUNKED_PREFILL", defaults.chunked_prefill_size
        ),
        max_running_requests=_positive_int(
            environ, "SCORING_MAX_REQS", defaults.max_running_requests
        ),
    )
