"""Tests for admin dry-run review endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status


class TestAdminDryRuns:
    def test_requires_admin_auth(self, client):
        with patch("scoring_service.api._helpers.settings") as mock_settings:
            mock_settings.admin_api_key = "the_key"

            response = client.get("/api/scoring/admin/dry-runs")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_lists_private_dry_runs(self, client):
        conn = MagicMock()
        dry_runs = [
            {
                "dry_run_id": 7,
                "status": "DRY_RUN_COMPLETE",
                "snapshot_hash": "snap",
                "scores_hash": None,
                "error_message": None,
                "started_at": "2026-05-08T10:00:00+00:00",
                "completed_at": "2026-05-08T10:05:00+00:00",
                "created_at": "2026-05-08T10:00:00+00:00",
            }
        ]

        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db", return_value=conn),
            patch("scoring_service.api.admin.list_dry_runs", return_value=(dry_runs, 1)),
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                "/api/scoring/admin/dry-runs?limit=5&offset=10",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["dry_runs"] == dry_runs
        assert data["total"] == 1
        assert data["limit"] == 5
        assert data["offset"] == 10
        conn.close.assert_called_once()

    @pytest.mark.parametrize(
        "offset",
        ["9223372036854775808", pytest.param("9" * 400, id="9x400"), "-1", "1.5", "abc"],
    )
    def test_rejects_offset_outside_bigint_range(self, client, offset):
        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db") as mock_get_db,
            patch("scoring_service.api.admin.list_dry_runs") as mock_list,
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                f"/api/scoring/admin/dry-runs?offset={offset}",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["detail"][0]["loc"] == ["query", "offset"]
        mock_get_db.assert_not_called()
        mock_list.assert_not_called()

    @pytest.mark.parametrize("offset", [0, 5, 9223372036854775807])
    def test_accepts_offset_within_bigint_range(self, client, offset):
        conn = MagicMock()

        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db", return_value=conn),
            patch("scoring_service.api.admin.list_dry_runs", return_value=([], 0)) as mock_list,
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                f"/api/scoring/admin/dry-runs?offset={offset}",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["offset"] == offset
        mock_list.assert_called_once_with(conn, 20, offset)

    def test_gets_private_dry_run_detail(self, client):
        conn = MagicMock()
        dry_run = {
            "dry_run_id": 7,
            "status": "SELECTED",
            "snapshot_hash": "snap",
            "scores_hash": None,
            "error_message": None,
            "started_at": "2026-05-08T10:00:00+00:00",
            "completed_at": None,
            "created_at": "2026-05-08T10:00:00+00:00",
        }

        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db", return_value=conn),
            patch("scoring_service.api.admin.get_dry_run", return_value=dry_run),
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                "/api/scoring/admin/dry-runs/7",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["dry_run_id"] == 7

    def test_returns_404_for_missing_dry_run(self, client):
        conn = MagicMock()

        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db", return_value=conn),
            patch("scoring_service.api.admin.get_dry_run", return_value=None),
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                "/api/scoring/admin/dry-runs/999",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_gets_private_dry_run_artifact(self, client):
        conn = MagicMock()
        artifact = {"dry_run_id": 7, "dry_run": True}

        with (
            patch("scoring_service.api._helpers.settings") as mock_settings,
            patch("scoring_service.api.admin.get_db", return_value=conn),
            patch(
                "scoring_service.api.admin.get_dry_run_artifact",
                return_value=artifact,
            ) as mock_get_artifact,
        ):
            mock_settings.admin_api_key = "the_key"

            response = client.get(
                "/api/scoring/admin/dry-runs/7/bundle.json",
                headers={"X-API-Key": "the_key"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == artifact
        mock_get_artifact.assert_called_once_with(conn, 7, "bundle.json")
