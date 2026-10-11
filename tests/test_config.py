import pytest
from pydantic import ValidationError

from scoring_service.config import QWEN_NON_THINKING_EXTRA_BODY, Settings
from scoring_service.server_version import ServerVersion


def test_vl_effective_lookahead_hours_rejects_negative():
    # A negative lookahead would place activation before publication and defeat
    # the pending-blob mechanism, so it must fail closed at config load.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vl_effective_lookahead_hours=-1)


def test_vl_effective_lookahead_hours_allows_zero():
    settings = Settings(_env_file=None, vl_effective_lookahead_hours=0)
    assert settings.vl_effective_lookahead_hours == 0


def test_scoring_defaults_use_qwen36_contract():
    settings = Settings(_env_file=None)

    assert settings.scoring_model_id == "Qwen/Qwen3.6-27B-FP8"
    assert settings.scoring_model_name == "qwen36-27b-fp8"
    assert settings.scoring_model_revision == ""
    assert settings.scoring_service_git_commit == ""
    assert settings.scoring_sglang_image_tag.endswith(
        "@sha256:5d9ec71597ade6b8237d61ae6f01b976cb3d5ad2c1e3cf4e0acaf27a9ff49a65"
    )
    assert settings.scoring_gpu_type == "H100"
    assert settings.scoring_quantization == ""
    assert settings.scoring_attention_backend == ""
    assert settings.scoring_tp == 1
    assert settings.scoring_mem_fraction == "0.75"
    assert settings.scoring_chunked_prefill == 4096
    assert settings.scoring_max_reqs == 1
    assert settings.scoring_reasoning_parser == "qwen3"
    assert settings.sglang_flashinfer_workspace_size == "2147483648"
    assert settings.scoring_disable_thinking is True
    assert settings.modal_key == ""
    assert settings.modal_secret == ""
    assert settings.modal_request_timeout_seconds == 2100
    assert QWEN_NON_THINKING_EXTRA_BODY == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize("value", [0, -1])
def test_unl_max_size_below_one_is_rejected(value):
    with pytest.raises(ValidationError, match="unl_max_size"):
        Settings(_env_file=None, unl_max_size=value)


def test_unl_max_size_of_one_is_allowed():
    assert Settings(_env_file=None, unl_max_size=1).unl_max_size == 1


# Numeric settings that used to load any integer or float and only misbehave
# where they are consumed: a non-positive request timeout or retry count means
# no request can succeed (max_retries=0 skips the request loop entirely), a
# negative backoff base raises inside time.sleep on the first retry, zero
# ingestion pages leave the watcher enabled but never reading the chain, a
# negative seal grace can seal a convergence report before reveals close and
# an infinite grace fraction overflows timedelta, a score cutoff outside the
# 0-100 score range or a negative churn gap changes UNL selection, a VL
# expiring in zero days is already expired when it activates, and a default
# page size outside the endpoint's own Query(ge=1, le=100) bounds makes every
# GET /api/scoring/rounds without ?limit= fail with 422. The SGLang manifest
# fields describe a launch that cannot serve with zero GPUs or zero requests.
# (field, lowest accepted value, rejected values)
MODEL_INGESTION_AND_CLIENT_BOUNDS = [
    ("modal_request_timeout_seconds", 1, [0, -1]),
    ("scoring_tp", 1, [0, -1]),
    ("scoring_max_reqs", 1, [0, -1]),
    ("convergence_ingestion_page_limit", 1, [0, -1]),
    ("convergence_ingestion_max_pages", 1, [0, -1]),
    ("convergence_seal_grace_fraction", 0.0, [-0.5]),
    ("convergence_seal_grace_floor_seconds", 0, [-1]),
    ("unl_score_cutoff", 0, [-1, 101]),
    ("unl_min_score_gap", 0, [-1]),
    ("vl_expiration_days", 1, [0, -1]),
    ("http_request_timeout", 1, [0, -1]),
    ("http_max_retries", 1, [0, -1]),
    ("http_retry_base_delay", 0, [-1]),
    ("default_page_limit", 1, [0, -1, 101]),
]


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field, _, rejected in MODEL_INGESTION_AND_CLIENT_BOUNDS for value in rejected],
)
def test_model_ingestion_and_client_settings_reject_out_of_range_values(field, value):
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [(field, lowest) for field, lowest, _ in MODEL_INGESTION_AND_CLIENT_BOUNDS],
)
def test_model_ingestion_and_client_settings_accept_the_lowest_valid_value(field, value):
    assert getattr(Settings(_env_file=None, **{field: value}), field) == value


@pytest.mark.parametrize("field", ["unl_score_cutoff", "default_page_limit"])
def test_upper_bounded_settings_accept_one_hundred(field):
    assert getattr(Settings(_env_file=None, **{field: 100}), field) == 100


@pytest.mark.parametrize("value", [float("inf"), "inf", float("nan"), "nan", float("-inf")])
def test_convergence_seal_grace_fraction_rejects_non_finite_values(value):
    # ge=0 alone admits +inf, and timedelta(seconds=inf) overflows in _seal_grace.
    with pytest.raises(ValidationError, match="convergence_seal_grace_fraction"):
        Settings(_env_file=None, convergence_seal_grace_fraction=value)


def test_model_ingestion_and_client_defaults_are_unchanged():
    settings = Settings(_env_file=None)
    assert [getattr(settings, field) for field, _, _ in MODEL_INGESTION_AND_CLIENT_BOUNDS] == [
        2100, 1, 1, 200, 20, 0.5, 120, 40, 3, 500, 30, 3, 2, 20,
    ]


@pytest.mark.parametrize(
    "env_name,value",
    [("CONVERGENCE_INGESTION_MAX_PAGES", "0"), ("CONVERGENCE_SEAL_GRACE_FRACTION", "inf")],
)
def test_out_of_range_model_ingestion_and_client_settings_are_rejected_from_the_environment(
    monkeypatch, env_name, value
):
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValidationError, match=env_name.lower()):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "field,value",
    [("convergence_ingestion_start_ledger", -1), ("scoring_chunked_prefill", -1)],
)
def test_sentinel_values_stay_accepted(field, value):
    # Left unbounded on purpose: -1 is the account_tx "earliest ledger" value
    # (PFTLClient.account_tx's own default) and SGLang's "disable chunked
    # prefill" value for --chunked-prefill-size.
    assert getattr(Settings(_env_file=None, **{field: value}), field) == value


def test_default_excluded_validator_server_versions():
    settings = Settings(_env_file=None)

    assert settings.excluded_validator_server_versions == "3.0.0"
    assert settings.excluded_validator_server_version_set == frozenset({"3.0.0"})


def test_excluded_validator_server_versions_parse_comma_separated_values():
    settings = Settings(
        _env_file=None,
        excluded_validator_server_versions="3.0.0, 2.9.0, ,1.0.0 ",
    )

    assert settings.excluded_validator_server_version_set == frozenset({
        "3.0.0",
        "2.9.0",
        "1.0.0",
    })


def test_minimum_safe_version_is_disabled_by_default():
    settings = Settings(_env_file=None)

    assert settings.minimum_safe_version == ""
    assert settings.minimum_safe_server_version is None


def test_minimum_safe_version_is_parsed():
    settings = Settings(_env_file=None, minimum_safe_version=" 1.0.8 ")

    assert settings.minimum_safe_version == "1.0.8"
    assert settings.minimum_safe_server_version == ServerVersion(
        release=(1, 0, 8), is_final=True
    )


@pytest.mark.parametrize("value", ["1.0.x", "1", "1.0.8-rc1", "1.0.8+DEBUG", "1.0.8.1"])
def test_minimum_safe_version_must_be_a_plain_final_release(value):
    with pytest.raises(ValidationError, match="MINIMUM_SAFE_VERSION"):
        Settings(_env_file=None, minimum_safe_version=value)


def test_minimum_safe_version_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MINIMUM_SAFE_VERSION", "1.0.8")

    assert Settings(_env_file=None).minimum_safe_version == "1.0.8"
