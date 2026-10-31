"""Complete the task with ordinary code once the category is known (0 AI tokens).

WASAT opens a ticket for the right team, replies from a template, and then re-reads the
tickets file to verify the ticket was really saved. A task only counts as successful if
this check passes.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TICKETS = Path(__file__).resolve().parent / "data" / "tickets.json"
_lock = threading.Lock()

REPLIES = {
    "account": "We've passed this to our account team. You can also manage your account in Settings.",
    "cancel": "Our cancellation fee details are on the Help page; our team will confirm your case shortly.",
    "contact": "A member of our support team will get back to you shortly.",
    "delivery": "Our delivery team will confirm the options and timing for you shortly.",
    "feedback": "Thank you for your feedback. It has been passed to our customer experience team.",
    "invoice": "Our billing team will send your invoice shortly. Invoices are also in your account area.",
    "order": "Our orders team is looking into this and will update you shortly.",
    "payment": "Our payments team will help you with this shortly.",
    "refund": "Our refunds team will review your request. Refunds are always checked by a person.",
    "shipping": "Our shipping team will update your address details shortly.",
    "subscription": "Your newsletter request has been passed to our team.",
}


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("tickets", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def run_task(category: str | None, message: str, *, tickets_path: Path | str | None = None) -> dict[str, Any]:
    """Open a ticket for the category's team, then verify it was saved."""
    if not category or category not in REPLIES:
        return {"ok": False, "verified": False, "action": "none", "ticket": None, "reply": ""}

    path = Path(tickets_path) if tickets_path else DEFAULT_TICKETS
    ticket = {
        "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
        "team": category,
        "message": message,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        tickets = _read(path)
        tickets.append(ticket)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tickets": tickets}, indent=2), encoding="utf-8")
        verified = any(t.get("ticket_id") == ticket["ticket_id"] for t in _read(path))

    return {"ok": verified, "verified": verified, "action": f"ticket for the {category} team",
            "ticket": ticket["ticket_id"], "reply": REPLIES[category]}
