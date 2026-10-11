import dataclasses

import pytest

from infra.endpoint_settings import (
    CHUNKED_PREFILL_DISABLED,
    EndpointSettings,
    load_endpoint_settings,
)
from tests.test_deploy_endpoint import (
    _flag_value,
    _load_deploy_endpoint,
    _start_server_command,
)


POSITIVE_INT = "an integer >= 1"
CHUNKED_PREFILL = "an integer >= 1, or -1 to disable chunked prefill"
FRACTION = "a number in (0, 1]"

REJECTED = [
    ("SCORING_TP", "0", POSITIVE_INT),
    ("SCORING_TP", "-1", POSITIVE_INT),
    ("SCORING_TP", "abc", POSITIVE_INT),
    ("SCORING_TP", "", POSITIVE_INT),
    ("SCORING_MAX_CONTAINERS", "0", POSITIVE_INT),
    ("SCORING_MAX_CONTAINERS", "-1", POSITIVE_INT),
    ("SCORING_MAX_CONTAINERS", "abc", POSITIVE_INT),
    ("SCORING_MAX_REQS", "0", POSITIVE_INT),
    ("SCORING_MAX_REQS", "-1", POSITIVE_INT),
    ("SCORING_MAX_REQS", "abc", POSITIVE_INT),
    ("SCORING_CHUNKED_PREFILL", "0", CHUNKED_PREFILL),
    ("SCORING_CHUNKED_PREFILL", "-2", CHUNKED_PREFILL),
    ("SCORING_CHUNKED_PREFILL", "abc", CHUNKED_PREFILL),
    ("SCORING_MEM_FRACTION", "0", FRACTION),
    ("SCORING_MEM_FRACTION", "-1", FRACTION),
    ("SCORING_MEM_FRACTION", "1.5", FRACTION),
    ("SCORING_MEM_FRACTION", "nan", FRACTION),
    ("SCORING_MEM_FRACTION", "inf", FRACTION),
    ("SCORING_MEM_FRACTION", "-inf", FRACTION),
    ("SCORING_MEM_FRACTION", "abc", FRACTION),
]


def test_defaults_are_unchanged():
    settings = load_endpoint_settings({})

    assert settings == EndpointSettings()
    assert settings.tensor_parallel == 1
    assert settings.max_containers == 3
    assert settings.mem_fraction_static == 0.75
    assert settings.chunked_prefill_size == 4096
    assert settings.max_running_requests == 1


def test_settings_are_frozen():
    settings = load_endpoint_settings({})

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.tensor_parallel = 2


@pytest.mark.parametrize(("name", "value", "accepted"), REJECTED)
def test_out_of_range_values_are_rejected(name, value, accepted):
    with pytest.raises(ValueError) as excinfo:
        load_endpoint_settings({name: value})

    message = str(excinfo.value)
    assert name in message
    assert accepted in message


@pytest.mark.parametrize(
    ("name", "value", "field", "expected"),
    [
        ("SCORING_TP", "1", "tensor_parallel", 1),
        ("SCORING_MAX_CONTAINERS", "1", "max_containers", 1),
        ("SCORING_MAX_REQS", "1", "max_running_requests", 1),
        ("SCORING_CHUNKED_PREFILL", "1", "chunked_prefill_size", 1),
        ("SCORING_CHUNKED_PREFILL", "-1", "chunked_prefill_size", -1),
        ("SCORING_MEM_FRACTION", "1", "mem_fraction_static", 1.0),
        ("SCORING_MEM_FRACTION", "0.01", "mem_fraction_static", 0.01),
    ],
)
def test_boundary_values_are_accepted(name, value, field, expected):
    settings = load_endpoint_settings({name: value})

    assert getattr(settings, field) == expected


def test_chunked_prefill_disable_sentinel_is_minus_one():
    assert CHUNKED_PREFILL_DISABLED == -1
    settings = load_endpoint_settings({"SCORING_CHUNKED_PREFILL": "-1"})
    assert settings.chunked_prefill_size == CHUNKED_PREFILL_DISABLED


def test_values_are_read_from_environ_mapping():
    settings = load_endpoint_settings({
        "SCORING_TP": "2",
        "SCORING_MAX_CONTAINERS": "5",
        "SCORING_MEM_FRACTION": "0.85",
        "SCORING_CHUNKED_PREFILL": "8192",
        "SCORING_MAX_REQS": "4",
        "SCORING_UNRELATED": "ignored",
    })

    assert settings == EndpointSettings(
        tensor_parallel=2,
        max_containers=5,
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        max_running_requests=4,
    )


def test_unset_values_keep_supplied_defaults():
    defaults = EndpointSettings(
        tensor_parallel=2,
        max_containers=5,
        mem_fraction_static=0.85,
        chunked_prefill_size=-1,
        max_running_requests=4,
    )

    assert load_endpoint_settings({}, defaults=defaults) == defaults
    assert load_endpoint_settings(
        {"SCORING_TP": "4"}, defaults=defaults
    ) == dataclasses.replace(defaults, tensor_parallel=4)


def test_deploy_endpoint_defaults_are_unchanged(monkeypatch):
    module, image = _load_deploy_endpoint(monkeypatch)

    assert module.TENSOR_PARALLEL == 1
    assert module.MAX_CONTAINERS == 3
    assert module.MEM_FRACTION_STATIC == 0.75
    assert module.CHUNKED_PREFILL_SIZE == 4096
    assert module.MAX_RUNNING_REQUESTS == 1
    assert module.app.cls_kwargs["max_containers"] == 3
    assert image.env_values["SCORING_TP"] == "1"
    assert image.env_values["SCORING_MAX_CONTAINERS"] == "3"
    assert image.env_values["SCORING_MEM_FRACTION"] == "0.75"
    assert image.env_values["SCORING_CHUNKED_PREFILL"] == "4096"
    assert image.env_values["SCORING_MAX_REQS"] == "1"


@pytest.mark.parametrize(("name", "value", "accepted"), REJECTED)
def test_deploy_endpoint_rejects_out_of_range_values_at_import(
    monkeypatch, name, value, accepted
):
    with pytest.raises(ValueError) as excinfo:
        _load_deploy_endpoint(monkeypatch, **{name: value})

    assert name in str(excinfo.value)
    assert accepted in str(excinfo.value)


def test_deploy_endpoint_launch_args_for_valid_overrides(monkeypatch):
    command = _start_server_command(
        monkeypatch,
        SCORING_TP="2",
        SCORING_MEM_FRACTION="0.85",
        SCORING_CHUNKED_PREFILL="-1",
        SCORING_MAX_REQS="4",
    )

    assert _flag_value(command, "--tp") == "2"
    assert _flag_value(command, "--mem-fraction-static") == "0.85"
    assert _flag_value(command, "--chunked-prefill-size") == "-1"
    assert _flag_value(command, "--max-running-requests") == "4"


def test_deploy_endpoint_launch_args_for_defaults(monkeypatch):
    command = _start_server_command(monkeypatch)

    assert _flag_value(command, "--tp") == "1"
    assert _flag_value(command, "--mem-fraction-static") == "0.75"
    assert _flag_value(command, "--chunked-prefill-size") == "4096"
    assert _flag_value(command, "--max-running-requests") == "1"


def test_start_server_rejects_out_of_range_container_env(monkeypatch):
    module, _ = _load_deploy_endpoint(monkeypatch)
    monkeypatch.setattr(module, "wait_for_server", lambda: None)
    monkeypatch.setenv("SCORING_MEM_FRACTION", "nan")

    with pytest.raises(ValueError, match="SCORING_MEM_FRACTION"):
        module.ScoringEndpoint().start_server()
