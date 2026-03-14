"""Service layer."""

from __future__ import annotations

from client import send_request


def fetch_user(user_id: int) -> dict:
    return send_request(f"/users/{user_id}", {"id": user_id})
