"""Tests for the RPC manifest client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from scoring_service.clients.rpc import RPCClient


@pytest.fixture
def client():
    return RPCClient(rpc_url="https://rpc.test.example.com")


MANIFEST_RESPONSE = {
    "result": {
        "details": {
            "master_key": "nHUkhbZe9ncdmhn6dbd5x7391ymwCS3YZEMWjysP9fSiDtau9YEe",
            "ephemeral_key": "n9Kc1swwT6uHYMv5feRTSTwtXtQgBWxDZrWDuHQj7fBTnQaoC9ux",
            "seq": 2,
        },
        "manifest": "JAAAAAJxIe23EdIq5b/pUIYmS3t0y/Irqy94rM01tnBCJNsXOax0CHMh",
        "requested": "nHUkhbZe9ncdmhn6dbd5x7391ymwCS3YZEMWjysP9fSiDtau9YEe",
        "status": "success",
    }
}

# JSON bodies a misbehaving node or proxy can return that are not the
# {"result": {...}} object the client expects, with the type the log names.
NON_OBJECT_BODIES = [
    pytest.param(None, "non-object body (NoneType)", id="body-null"),
    pytest.param([], "non-object body (list)", id="body-list"),
    pytest.param("string", "non-object body (str)", id="body-string"),
    pytest.param({"result": None}, "non-object result (NoneType)", id="result-null"),
    pytest.param({"result": "busy"}, "non-object result (str)", id="result-string"),
    pytest.param({"result": []}, "non-object result (list)", id="result-list"),
]


def _mock_post(mock_client_cls, body):
    mock_response = MagicMock()
    mock_response.json.return_value = body
    mock_response.raise_for_status = MagicMock()
    mock_http = MagicMock(post=MagicMock(return_value=mock_response))
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_http)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    return mock_http


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


class TestInit:
    def test_raises_when_rpc_url_missing(self):
        with patch("scoring_service.clients.rpc.settings") as mock_settings:
            mock_settings.rpc_url = ""
            with pytest.raises(ValueError, match="RPC_URL is required"):
                RPCClient()

    def test_uses_explicit_url(self):
        client = RPCClient(rpc_url="https://custom.example.com")
        assert client.rpc_url == "https://custom.example.com"

    def test_uses_settings_url(self):
        with patch("scoring_service.clients.rpc.settings") as mock_settings:
            mock_settings.rpc_url = "https://from-settings.example.com"
            client = RPCClient()
            assert client.rpc_url == "https://from-settings.example.com"


class TestFetchManifest:
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_returns_manifest_on_success(self, mock_client_cls, client):
        mock_response = MagicMock()
        mock_response.json.return_value = MANIFEST_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = client.fetch_manifest("nHUkhbZe9ncdmhn6dbd5x7391ymwCS3YZEMWjysP9fSiDtau9YEe")
        assert result == "JAAAAAJxIe23EdIq5b/pUIYmS3t0y/Irqy94rM01tnBCJNsXOax0CHMh"

    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_returns_none_on_rpc_error(self, mock_client_cls, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"error": "invalidParams", "error_message": "bad key"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = client.fetch_manifest("bad_key")
        assert result is None

    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_returns_none_on_missing_manifest_field(self, mock_client_cls, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"details": {}, "status": "success"}}
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = client.fetch_manifest("nHUkhbZe9ncdmhn6dbd5x7391ymwCS3YZEMWjysP9fSiDtau9YEe")
        assert result is None

    @patch("scoring_service.clients.rpc.httpx.Client")
    @patch("scoring_service.clients.rpc.time.sleep")
    def test_retries_on_http_error(self, mock_sleep, mock_client_cls, client):
        mock_client_instance = MagicMock()
        mock_client_instance.post.side_effect = [
            httpx.ConnectError("connection refused"),
            MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value=MANIFEST_RESPONSE),
            ),
        ]
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = client.fetch_manifest("nHUkhbZe9ncdmhn6dbd5x7391ymwCS3YZEMWjysP9fSiDtau9YEe")
        assert result == "JAAAAAJxIe23EdIq5b/pUIYmS3t0y/Irqy94rM01tnBCJNsXOax0CHMh"
        assert mock_sleep.called

    @pytest.mark.parametrize("body, logged", NON_OBJECT_BODIES)
    @patch("scoring_service.clients.rpc.time.sleep")
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_returns_none_on_non_object_body(
        self, mock_client_cls, mock_sleep, client, caplog, body, logged
    ):
        mock_http = _mock_post(mock_client_cls, body)

        with caplog.at_level("ERROR", logger="scoring_service.clients.rpc"):
            result = client.fetch_manifest("nHkey")

        assert result is None
        assert _errors(caplog) == [f"RPC manifest returned a {logged}"]
        # Terminal like an RPC error: no retry.
        assert mock_http.post.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.parametrize(
        "manifest, type_name",
        [
            pytest.param({"x": 1}, "dict", id="manifest-dict"),
            pytest.param(["a"], "list", id="manifest-list"),
            pytest.param(7, "int", id="manifest-int"),
            pytest.param("", "str", id="manifest-empty-string"),
        ],
    )
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_returns_none_on_non_string_manifest(
        self, mock_client_cls, client, caplog, manifest, type_name
    ):
        _mock_post(mock_client_cls, {"result": {"manifest": manifest}})

        with caplog.at_level("ERROR", logger="scoring_service.clients.rpc"):
            result = client.fetch_manifest("nHkey")

        assert result is None
        assert _errors(caplog) == [
            f"RPC manifest for nHkey is not a non-empty string ({type_name})"
        ]

    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_fetch_manifests_skips_key_with_non_object_body(self, mock_client_cls, client):
        _mock_post(mock_client_cls, [])

        assert client.fetch_manifests(["nHkey"]) == {}


class TestFetchManifests:
    @patch.object(RPCClient, "fetch_manifest")
    def test_returns_dict_of_manifests(self, mock_fetch, client):
        mock_fetch.side_effect = ["manifest_a_b64", "manifest_b_b64"]
        result = client.fetch_manifests(["key_a", "key_b"])
        assert result == {"key_a": "manifest_a_b64", "key_b": "manifest_b_b64"}

    @patch.object(RPCClient, "fetch_manifest")
    def test_omits_keys_with_no_manifest(self, mock_fetch, client):
        mock_fetch.side_effect = ["manifest_a_b64", None]
        result = client.fetch_manifests(["key_a", "key_b"])
        assert result == {"key_a": "manifest_a_b64"}

    @patch.object(RPCClient, "fetch_manifest")
    def test_returns_empty_dict_for_empty_input(self, mock_fetch, client):
        result = client.fetch_manifests([])
        assert result == {}
        mock_fetch.assert_not_called()


class TestIsReachable:
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_true_when_server_info_answers(self, mock_client_cls, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"info": {"build_version": "1.0.6"}, "status": "success"}}
        mock_response.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_response)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        assert client.is_reachable() is True

    @patch("scoring_service.clients.rpc.time.sleep")
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_false_when_transport_fails(self, mock_client_cls, _sleep, client):
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(post=MagicMock(side_effect=httpx.ConnectError("down")))
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        assert client.is_reachable() is False

    @pytest.mark.parametrize("body, logged", NON_OBJECT_BODIES)
    @patch("scoring_service.clients.rpc.time.sleep")
    @patch("scoring_service.clients.rpc.httpx.Client")
    def test_false_on_non_object_body(
        self, mock_client_cls, mock_sleep, client, caplog, body, logged
    ):
        mock_http = _mock_post(mock_client_cls, body)

        with caplog.at_level("ERROR", logger="scoring_service.clients.rpc"):
            assert client.is_reachable() is False

        assert _errors(caplog) == [f"RPC server_info returned a {logged}"]
        assert mock_http.post.call_count == 1
        mock_sleep.assert_not_called()
