"""Shared prompt state for Spotify prompt pipelines."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone


@dataclass(frozen=True)
class PromptOverlayState:
    prompt: str = ""
    update_kind: str = ""
    track_title: str = ""
    track_artists: str = ""
    updated_at_iso: str = ""


_STATE_LOCK = threading.Lock()
_LATEST_STATE = PromptOverlayState()


def set_latest_prompt_state(
    *,
    prompt: str,
    update_kind: str,
    track_title: str,
    track_artists: str,
) -> None:
    global _LATEST_STATE
    updated_at_iso = datetime.now(tz=timezone.utc).isoformat()
    with _STATE_LOCK:
        _LATEST_STATE = PromptOverlayState(
            prompt=prompt.strip(),
            update_kind=update_kind.strip(),
            track_title=track_title.strip(),
            track_artists=track_artists.strip(),
            updated_at_iso=updated_at_iso,
        )


def get_latest_prompt_state() -> PromptOverlayState:
    with _STATE_LOCK:
        return replace(_LATEST_STATE)
