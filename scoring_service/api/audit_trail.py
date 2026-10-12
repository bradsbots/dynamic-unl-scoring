"""HTTPS fallback endpoint for audit trail files."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from scoring_service.api._helpers import public_round_exists, round_outputs_available
from scoring_service.database import get_db
from scoring_service.services.ipfs_publisher import (
    get_audit_trail_file,
    get_input_package_file,
)

router = APIRouter(prefix="/api/scoring")


def _input_file_not_found(round_number: int, file_path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": (
                f"Input package file not found: round {round_number}, "
                f"path {file_path}"
            )
        },
    )


def _audit_file_not_found(round_number: int, file_path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": f"File not found: round {round_number}, path {file_path}"},
    )


@router.get("/rounds/{round_number}/input/{file_path:path}")
def serve_input_package_file(round_number: int, file_path: str):
    """Serve a single frozen input package file for a scoring round."""
    if "\x00" in file_path:
        # Postgres text cannot hold NUL, so no stored file can match; psycopg2
        # would raise ValueError on the query parameter instead.
        return _input_file_not_found(round_number, file_path)

    connection = get_db()
    try:
        content = (
            get_input_package_file(connection, round_number, file_path)
            if public_round_exists(connection, round_number)
            else None
        )
    finally:
        connection.close()

    if content is None:
        return _input_file_not_found(round_number, file_path)

    return JSONResponse(content=content, media_type="application/json")


@router.get("/rounds/{round_number}/{file_path:path}")
def serve_audit_trail_file(round_number: int, file_path: str):
    """Serve a single audit trail file for a scoring round."""
    if "\x00" in file_path:
        return _audit_file_not_found(round_number, file_path)

    connection = get_db()
    try:
        content = (
            get_audit_trail_file(connection, round_number, file_path)
            if round_outputs_available(connection, round_number)
            else None
        )
    finally:
        connection.close()

    if content is None:
        return _audit_file_not_found(round_number, file_path)

    return JSONResponse(content=content, media_type="application/json")
