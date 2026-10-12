"""Tests for the HTTPS audit trail fallback endpoint."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from urllib.parse import unquote

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from scoring_service.api._helpers import round_outputs_available
from scoring_service.services.orchestrator import RoundState


class TestServeAuditTrailFile:
    def test_returns_file_when_present(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ({"round_number": 1, "validators": []},)

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=True),
        ):
            response = client.get("/api/scoring/rounds/1/snapshot.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"round_number": 1, "validators": []}
        mock_conn.close.assert_called_once()

    def test_returns_404_when_file_missing(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=True),
        ):
            response = client.get("/api/scoring/rounds/1/nonexistent.json")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "error" in response.json()
        mock_conn.close.assert_called_once()

    def test_returns_404_for_missing_round(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=False),
        ):
            response = client.get("/api/scoring/rounds/999/metadata.json")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_output_row_is_not_served_before_commit_close(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        mock_cursor.fetchone.return_value = (
            RoundState.AWAITING_COMMIT_CLOSE.value,
            None,
            future,
        )

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch(
                "scoring_service.api.audit_trail.get_audit_trail_file",
                return_value={"model_response_hash": "1" * 64},
            ) as get_file,
        ):
            response = client.get(
                "/api/scoring/rounds/1/outputs/verification_hashes.json"
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        get_file.assert_not_called()
        mock_conn.close.assert_called_once()

    def test_returns_404_for_historical_dry_run_round(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=False),
        ):
            response = client.get("/api/scoring/rounds/1/metadata.json")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_handles_nested_file_paths(self, client):
        raw_data = {"validators": [{"master_key": "nHU..."}]}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (raw_data,)

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=True),
        ):
            response = client.get("/api/scoring/rounds/1/raw/vhs_validators.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == raw_data

        call_args = mock_cursor.execute.call_args[0]
        assert call_args[1] == (1, "raw/vhs_validators.json")

    def test_content_type_is_json(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ({"data": True},)

        with (
            patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn),
            patch("scoring_service.api.audit_trail.round_outputs_available", return_value=True),
        ):
            response = client.get("/api/scoring/rounds/1/metadata.json")

        assert "application/json" in response.headers["content-type"]

    def test_single_segment_round_path_does_not_hit_audit_handler(self, client):
        """A single-segment URL like `/api/scoring/rounds/14` must route to the
        round-detail handler in `scoring.py`, not the audit-trail handler.

        This guards against a routing collision after the audit-trail router
        was moved under the shared `/api/scoring` prefix in M1.12.3.
        FastAPI distinguishes the two routes by segment count, but this test
        locks that behavior so a future refactor cannot silently reintroduce
        ambiguity.
        """
        audit_conn = MagicMock()
        scoring_conn = MagicMock()
        scoring_cursor = MagicMock()
        scoring_conn.cursor.return_value = scoring_cursor
        scoring_cursor.fetchone.return_value = None

        with (
            patch(
                "scoring_service.api.audit_trail.get_db",
                return_value=audit_conn,
            ) as audit_db,
            patch(
                "scoring_service.api.scoring.get_db",
                return_value=scoring_conn,
            ) as scoring_db,
        ):
            response = client.get("/api/scoring/rounds/14")

        audit_db.assert_not_called()
        scoring_db.assert_called_once()
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestRoundOutputsAvailable:
    def _conn(self, row):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.return_value = row
        return conn

    def test_fails_closed_when_boundary_missing(self):
        conn = self._conn((RoundState.COMPLETE.value, None, None))

        assert round_outputs_available(conn, 1) is False

    def test_false_before_commit_close(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        conn = self._conn((RoundState.AWAITING_COMMIT_CLOSE.value, None, future))

        assert round_outputs_available(conn, 1) is False

    def test_true_after_commit_close(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        conn = self._conn((RoundState.AWAITING_COMMIT_CLOSE.value, None, past))

        assert round_outputs_available(conn, 1) is True

    def test_override_outputs_available_after_publication(self):
        conn = self._conn((RoundState.IPFS_PUBLISHED.value, "custom", None))

        assert round_outputs_available(conn, 1) is True


class TestServeInputPackageFile:
    def test_returns_input_package_file_when_present(self, client):
        input_bundle = {
            "package_kind": "input",
            "entrypoints": {"model_request": "inputs/model_request.json"},
        }
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(1,), (input_bundle,)]

        with patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn):
            response = client.get("/api/scoring/rounds/1/input/bundle.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == input_bundle
        executed_sql = mock_cursor.execute.call_args_list[-1].args[0]
        executed_params = mock_cursor.execute.call_args_list[-1].args[1]
        assert "input_package_files" in executed_sql
        assert executed_params == (1, "bundle.json")
        mock_conn.close.assert_called_once()

    def test_returns_404_when_input_package_file_missing(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(1,), None]

        with patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn):
            response = client.get(
                "/api/scoring/rounds/1/input/inputs/model_request.json"
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Input package file not found" in response.json()["error"]

    def test_returns_404_for_historical_dry_run_round(self, client):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        with patch("scoring_service.api.audit_trail.get_db", return_value=mock_conn):
            response = client.get("/api/scoring/rounds/1/input/bundle.json")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        query_sql = mock_cursor.execute.call_args.args[0]
        query_params = mock_cursor.execute.call_args.args[1]
        assert "status != %s" in query_sql
        assert query_params == (1, "DRY_RUN_COMPLETE")


def _nul_rejecting_conn(fetchone_value):
    """Mock connection whose cursor refuses NUL in string params, as psycopg2 does."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor

    def execute(sql, params=None):
        for param in params or ():
            if isinstance(param, str) and "\x00" in param:
                raise ValueError(
                    "A string literal cannot contain NUL (0x00) characters."
                )

    cursor.execute.side_effect = execute
    cursor.fetchone.return_value = fetchone_value
    return conn


class TestNulBytePaths:
    """A NUL byte in file_path can never match a stored row (Postgres text
    cannot hold NUL), so both public file routes answer their usual 404
    without opening a database connection."""

    @pytest.fixture
    def lenient_client(self, client):
        return TestClient(client.app, raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "raw_path",
        [
            "scores.json%00",
            "%00",
            "%00scores.json",
            "sco%00res.json",
            "outputs/%00/x.json",
        ],
    )
    def test_audit_trail_nul_path_returns_404_without_db(
        self, lenient_client, raw_path
    ):
        conn = _nul_rejecting_conn(({"ok": 1},))

        with (
            patch(
                "scoring_service.api.audit_trail.get_db", return_value=conn
            ) as get_db,
            patch(
                "scoring_service.api.audit_trail.round_outputs_available",
                return_value=True,
            ),
        ):
            response = lenient_client.get(f"/api/scoring/rounds/1/{raw_path}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json() == {
            "error": f"File not found: round 1, path {unquote(raw_path)}"
        }
        get_db.assert_not_called()

    @pytest.mark.parametrize(
        "raw_path",
        [
            "validators.json%00",
            "%00",
            "%00validators.json",
            "vali%00dators.json",
            "inputs/%00/x.json",
        ],
    )
    def test_input_package_nul_path_returns_404_without_db(
        self, lenient_client, raw_path
    ):
        conn = _nul_rejecting_conn(({"ok": 1},))

        with (
            patch(
                "scoring_service.api.audit_trail.get_db", return_value=conn
            ) as get_db,
            patch(
                "scoring_service.api.audit_trail.public_round_exists",
                return_value=True,
            ),
        ):
            response = lenient_client.get(
                f"/api/scoring/rounds/1/input/{raw_path}"
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json() == {
            "error": (
                "Input package file not found: round 1, "
                f"path {unquote(raw_path)}"
            )
        }
        get_db.assert_not_called()

    def test_audit_trail_normal_path_still_reaches_query(self, lenient_client):
        conn = _nul_rejecting_conn(({"ok": 1},))

        with (
            patch(
                "scoring_service.api.audit_trail.get_db", return_value=conn
            ) as get_db,
            patch(
                "scoring_service.api.audit_trail.round_outputs_available",
                return_value=True,
            ),
        ):
            response = lenient_client.get("/api/scoring/rounds/1/scores.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"ok": 1}
        get_db.assert_called_once()
        assert conn.cursor.return_value.execute.call_args.args[1] == (
            1,
            "scores.json",
        )
        conn.close.assert_called_once()

    def test_input_package_normal_path_still_reaches_query(self, lenient_client):
        conn = _nul_rejecting_conn(({"ok": 1},))

        with (
            patch(
                "scoring_service.api.audit_trail.get_db", return_value=conn
            ) as get_db,
            patch(
                "scoring_service.api.audit_trail.public_round_exists",
                return_value=True,
            ),
        ):
            response = lenient_client.get(
                "/api/scoring/rounds/1/input/validators.json"
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"ok": 1}
        get_db.assert_called_once()
        assert conn.cursor.return_value.execute.call_args.args[1] == (
            1,
            "validators.json",
        )
        conn.close.assert_called_once()
