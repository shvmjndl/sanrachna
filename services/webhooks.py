"""Webhook delivery for ingest and query events."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Any

import httpx
from sqlalchemy import select

from app.db.models import WebhookSubscription


class WebhookService:
    def __init__(self, session_factory):
        self._session = session_factory

    def register(self, url: str, events: list[str], secret: str | None = None) -> dict:
        sub_id = f"wh_{uuid.uuid4().hex[:12]}"
        secret = secret or uuid.uuid4().hex
        with self._session() as session:
            session.merge(
                WebhookSubscription(
                    id=sub_id,
                    url=url,
                    events=events,
                    secret=secret,
                    active=True,
                )
            )
            session.commit()
        return {"subscription_id": sub_id, "url": url, "events": events, "secret": secret}

    def list_subscriptions(self) -> list[dict]:
        with self._session() as session:
            subs = session.execute(select(WebhookSubscription)).scalars().all()
        return [
            {"subscription_id": s.id, "url": s.url, "events": s.events, "active": s.active}
            for s in subs
        ]

    def emit(self, event: str, payload: dict[str, Any]) -> int:
        with self._session() as session:
            subs = session.execute(
                select(WebhookSubscription).where(WebhookSubscription.active.is_(True))
            ).scalars().all()
            delivered = 0
            for sub in subs:
                if event not in (sub.events or []):
                    continue
                body = json.dumps({"event": event, "payload": payload}, ensure_ascii=False)
                headers = {"Content-Type": "application/json"}
                if sub.secret:
                    sig = hmac.new(sub.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                    headers["X-Sanrachna-Signature"] = sig
                try:
                    httpx.post(sub.url, content=body, headers=headers, timeout=10.0)
                    delivered += 1
                except Exception:
                    pass
            session.commit()
        return delivered
