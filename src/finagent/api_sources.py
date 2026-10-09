"""`GET /api/sources/{session_id}`: the evidence records of one chat, for the web app's source chips.

The answer text cites sources as `[E1]`; the UI turns each marker into a chip and needs the record
behind it (title, publisher, date, trust tier, a snippet and the link). The stream only carries
`{id, title, url, publisher, tier}` with a tool result, and history carries nothing, so the page
reads this endpoint when a chat is opened and after every turn.

The router is built like `api_chat.router` (same prefix, `api_common.GuardedRoute` and the 503
dependency when there is no chat runtime) and is included next to it, so `LocalApiGuard` covers it
too. It only reads: `EvidenceStore.list(session_id)`, in the order the sources were recorded. An
unknown session is a 404. Every field is as stored (connectors sanitize external text before they
store it); the snippet is cut to `SNIPPET_MAX` characters. The records are external content: the
page renders them as text and links only `http(s)` URLs.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .api_chat import SessionId, _session_or_404
from .api_common import GuardedRoute, Runtime, respond, runtime

SNIPPET_MAX = 500

router = APIRouter(prefix="/api", route_class=GuardedRoute, dependencies=[Depends(runtime)])


def _source_json(record) -> dict:
    return {
        "id": record.id,
        "title": record.title,
        "url": record.url,
        "publisher": record.publisher,
        "tier": record.tier,
        "published_at": record.published_at,
        "connector": record.connector,
        "untrusted": bool(record.untrusted),
        "snippet": (record.snippet or "")[:SNIPPET_MAX],
    }


@router.get("/sources/{session_id}")
async def list_sources(session_id: SessionId, rt: Runtime):
    _session_or_404(rt, session_id)
    return respond({"sources": [_source_json(record) for record in rt.evidence.list(session_id)]})
