"""Spotify prompts preprocessor pipeline for Daydream Scope."""

from __future__ import annotations

import base64
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import requests
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import SpotifyPromptOverlayConfig, SpotifyPromptsConfig
from ..state import get_latest_prompt_state, set_latest_prompt_state

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except Exception:
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)
_DOTENV_LOCK = threading.Lock()
_DOTENV_LOADED = False
PromptUpdateKind = Literal["track_change", "random", "manual"]

_RANDOM_VARIATION_HINTS: tuple[str, ...] = (
    "shift the color palette and lighting mood while keeping scene coherence",
    "reinterpret the same song as a different visual environment",
    "change the dominant subject and motion pattern",
    "move from abstract textures to grounded cinematic scenery",
    "move from grounded scenery to abstract rhythmic forms",
    "change scale dramatically: intimate micro-world to vast landscape",
)


def _load_dotenv() -> None:
    """Load a .env file from the Scope working directory if present."""
    global _DOTENV_LOADED
    with _DOTENV_LOCK:
        if _DOTENV_LOADED:
            return

        dotenv_path = Path.cwd() / ".env"
        if not dotenv_path.exists():
            _DOTENV_LOADED = True
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

        _DOTENV_LOADED = True


class SpotifyAuthError(RuntimeError):
    """Spotify authentication/configuration error."""


@dataclass
class SpotifyTrack:
    track_id: str
    title: str
    artists: str
    album: str
    is_playing: bool


class SpotifyClient:
    """Spotify now-playing client using refresh-token flow."""

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token: str | None = None
        self._expires_at = 0.0

    def _refresh_access_token(self) -> str:
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("utf-8")
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            headers={"Authorization": f"Basic {basic}"},
            timeout=20,
        )
        if response.status_code >= 400:
            raise SpotifyAuthError(
                f"Spotify token refresh failed: {response.status_code} {response.text}"
            )
        data = response.json()
        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        self._expires_at = time.time() + expires_in - 30
        return self._access_token

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._expires_at:
            return self._access_token
        return self._refresh_access_token()

    def get_current_track(self) -> SpotifyTrack | None:
        token = self._get_access_token()
        response = requests.get(
            "https://api.spotify.com/v1/me/player/currently-playing",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

        if response.status_code == 204:
            return None
        if response.status_code == 401:
            token = self._refresh_access_token()
            response = requests.get(
                "https://api.spotify.com/v1/me/player/currently-playing",
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Spotify now-playing failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        item = payload.get("item")
        if not item:
            return None

        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return SpotifyTrack(
            track_id=item.get("id") or "",
            title=item.get("name") or "Unknown title",
            artists=artists or "Unknown artist",
            album=(item.get("album") or {}).get("name", "Unknown album"),
            is_playing=bool(payload.get("is_playing", False)),
        )


class OpenAIPromptClient:
    """OpenAI client for generating visual prompts from track metadata."""

    SYSTEM_PROMPT = (
        "You create concise, high-quality prompts for real-time video diffusion. "
        "Return exactly one prompt only. No markdown, no bullets, no explanation."
    )

    def __init__(self, api_key: str, model: str, timeout_seconds: int, base_url: str):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")

    def _extract_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

        if parts:
            return " ".join(parts).strip()
        raise RuntimeError("OpenAI response did not include prompt text")

    def generate_visual_prompt(
        self,
        track: SpotifyTrack,
        user_idea: str,
        max_words: int,
        variation_hint: str | None = None,
        previous_prompt: str | None = None,
    ) -> str:
        variation_block = (
            f"Variation request: {variation_hint}\n" if variation_hint else ""
        )
        previous_prompt_block = (
            f"Previous prompt to avoid copying verbatim: {previous_prompt}\n"
            if previous_prompt
            else ""
        )
        user_block = (
            "Generate a single cinematic video prompt for a diffusion model.\n"
            f"Song title: {track.title}\n"
            f"Artist(s): {track.artists}\n"
            f"Album: {track.album}\n"
            f"User creative direction: {user_idea or 'None'}\n"
            f"{variation_block}"
            f"{previous_prompt_block}"
            f"Constraints: vivid visual language, present tense, <= {max_words} words, "
            "safe for work, no artist names, no camera metadata unless useful."
        )

        max_output_tokens = max(90, min(220, int(max_words * 3.0) + 18))

        request_body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_block}],
                },
            ],
            "max_output_tokens": max_output_tokens,
        }

        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI prompt request failed: {response.status_code} {response.text}"
            )
        return self._extract_text(response.json())


class SpotifyPromptsPipeline(Pipeline):
    """Poll Spotify and inject generated prompts for track, random, or manual updates."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return SpotifyPromptsConfig

    def __init__(
        self,
        poll_interval_seconds: float = 8.0,
        openai_model: str = "gpt-5.2",
        openai_timeout_seconds: int = 45,
        user_idea: str = "",
        prompt_weight: float = 1.0,
        reset_cache_on_track_change: bool = True,
        enable_random_prompt_switches: bool = True,
        random_switch_min_seconds: int = 20,
        random_switch_max_seconds: int = 45,
        soft_transition_steps: int = 4,
        soft_transition_method: str = "slerp",
        manual_prompt_refresh_counter: int = 0,
        prompt_max_words: int = 40,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.poll_interval_seconds = max(2.0, float(poll_interval_seconds))
        self.prompt_weight = float(prompt_weight)
        self.reset_cache_on_track_change = bool(reset_cache_on_track_change)
        self.enable_random_prompt_switches = bool(enable_random_prompt_switches)
        self.random_switch_min_seconds = max(3, int(random_switch_min_seconds))
        self.random_switch_max_seconds = max(
            self.random_switch_min_seconds, int(random_switch_max_seconds)
        )
        self.soft_transition_steps = max(0, int(soft_transition_steps))
        self.soft_transition_method = self._normalize_transition_method(
            soft_transition_method
        )
        self.manual_prompt_refresh_counter = int(manual_prompt_refresh_counter)
        self.prompt_max_words = max(12, min(80, int(prompt_max_words)))

        self._state_lock = threading.Lock()
        self._user_idea = user_idea.strip()
        self._pending_manual_refresh = False
        self._last_track_id: str | None = None
        self._last_prompt: str = ""
        self._last_random_hint: str | None = None
        self._next_random_prompt_at: float | None = None

        self._prompt_updates: queue.Queue[dict[str, str]] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

        _load_dotenv()

        self.spotify_client = self._build_spotify_client()
        self.openai_client = self._build_openai_client(
            model=openai_model,
            timeout_seconds=max(10, int(openai_timeout_seconds)),
        )

        if self.spotify_client is None:
            logger.warning(
                "SPOTIFY-PROMPTS: disabled; set SPOTIFY_CLIENT_ID, "
                "SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN."
            )
            return

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="spotify-prompts-poller",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(
            "SPOTIFY-PROMPTS: started polling Spotify every %.1fs",
            self.poll_interval_seconds,
        )

    @staticmethod
    def _normalize_transition_method(value: str) -> Literal["linear", "slerp"]:
        normalized = str(value).strip().lower()
        return "linear" if normalized == "linear" else "slerp"

    def _mark_manual_refresh_requested(self) -> None:
        with self._state_lock:
            self._pending_manual_refresh = True
        self._wake_event.set()

    def _consume_manual_refresh_requested(self) -> bool:
        with self._state_lock:
            if not self._pending_manual_refresh:
                return False
            self._pending_manual_refresh = False
            return True

    def _schedule_next_random_prompt(self, now: float | None = None) -> None:
        if not self.enable_random_prompt_switches:
            self._next_random_prompt_at = None
            return
        base_time = now if now is not None else time.monotonic()
        interval = random.uniform(
            float(self.random_switch_min_seconds),
            float(self.random_switch_max_seconds),
        )
        self._next_random_prompt_at = base_time + interval

    def _choose_random_variation_hint(self) -> str:
        hint = random.choice(_RANDOM_VARIATION_HINTS)
        if hint == self._last_random_hint and len(_RANDOM_VARIATION_HINTS) > 1:
            hint = random.choice(
                tuple(item for item in _RANDOM_VARIATION_HINTS if item != hint)
            )
        self._last_random_hint = hint
        return hint

    def _wait_for_next_poll(self) -> None:
        deadline = time.monotonic() + self.poll_interval_seconds
        while not self._stop_event.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._stop_event.wait(min(0.4, remaining))

    def _build_spotify_client(self) -> SpotifyClient | None:
        client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
        refresh_token = os.getenv("SPOTIFY_REFRESH_TOKEN", "").strip()
        if not client_id or not client_secret or not refresh_token:
            return None
        return SpotifyClient(client_id, client_secret, refresh_token)

    def _build_openai_client(
        self, model: str, timeout_seconds: int
    ) -> OpenAIPromptClient | None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        if not api_key:
            logger.warning(
                "SPOTIFY-PROMPTS: OPENAI_API_KEY not set; using fallback prompt template."
            )
            return None
        return OpenAIPromptClient(
            api_key=api_key,
            model=model.strip() or "gpt-5.2",
            timeout_seconds=timeout_seconds,
            base_url=base_url,
        )

    @staticmethod
    def _truncate_prompt_to_max_words(prompt: str, max_words: int) -> str:
        text = " ".join(prompt.strip().split())
        if not text:
            return ""
        words = text.split(" ")
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip(" ,;:.")

    def _fallback_prompt(
        self,
        track: SpotifyTrack,
        variation_hint: str | None = None,
    ) -> str:
        with self._state_lock:
            user_idea = self._user_idea

        suffix = f" Extra direction: {user_idea}." if user_idea else ""
        variation_suffix = (
            f" Alternate pass: {variation_hint}."
            if variation_hint
            else ""
        )
        prompt = (
            f"Cinematic abstract reinterpretation of '{track.title}' by {track.artists}, "
            f"rhythmic motion, luminous particles, rich contrast, immersive atmosphere, "
            f"smooth temporal continuity, detailed textures, dynamic but coherent scene evolution."
            f"{suffix}"
            f"{variation_suffix}"
        )
        return self._truncate_prompt_to_max_words(prompt, self.prompt_max_words)

    def _prompt_for_track(self, track: SpotifyTrack, update_kind: PromptUpdateKind) -> str:
        variation_hint = (
            self._choose_random_variation_hint()
            if update_kind in {"random", "manual"}
            else None
        )
        if self.openai_client is None:
            return self._fallback_prompt(track, variation_hint=variation_hint)

        with self._state_lock:
            user_idea = self._user_idea
        previous_prompt = self._last_prompt if self._last_prompt else None
        try:
            prompt = self.openai_client.generate_visual_prompt(
                track,
                user_idea,
                max_words=self.prompt_max_words,
                variation_hint=variation_hint,
                previous_prompt=previous_prompt,
            )
            return self._truncate_prompt_to_max_words(prompt, self.prompt_max_words)
        except Exception as exc:
            logger.error(
                "SPOTIFY-PROMPTS: OpenAI generation failed, using fallback: %s", exc
            )
            return self._fallback_prompt(track, variation_hint=variation_hint)

    def _enqueue_prompt_update(self, prompt: str, update_kind: PromptUpdateKind) -> None:
        payload = {"prompt": prompt, "update_kind": update_kind}
        try:
            self._prompt_updates.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self._prompt_updates.get_nowait()
        except queue.Empty:
            pass

        try:
            self._prompt_updates.put_nowait(payload)
        except queue.Full:
            pass

    def _process_prompt_update(self, track: SpotifyTrack, update_kind: PromptUpdateKind) -> None:
        prompt = self._prompt_for_track(track, update_kind=update_kind)
        if not prompt:
            return
        if prompt == self._last_prompt and update_kind != "track_change":
            return
        self._last_prompt = prompt
        self._enqueue_prompt_update(prompt, update_kind=update_kind)
        set_latest_prompt_state(
            prompt=prompt,
            update_kind=update_kind,
            track_title=track.title,
            track_artists=track.artists,
        )
        logger.info(
            "SPOTIFY-PROMPTS: update=%s track='%s' artists='%s'",
            update_kind,
            track.title,
            track.artists,
        )
        logger.info(
            "SPOTIFY-PROMPTS: >>> NEW PROMPT (%s): '%s'",
            update_kind,
            prompt,
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.spotify_client is None:
                    return

                track = self.spotify_client.get_current_track()
                if track and track.is_playing and track.track_id:
                    if track.track_id != self._last_track_id:
                        self._last_track_id = track.track_id
                        self._process_prompt_update(track, update_kind="track_change")
                        with self._state_lock:
                            self._pending_manual_refresh = False
                        self._schedule_next_random_prompt()
                    elif self._consume_manual_refresh_requested():
                        self._process_prompt_update(track, update_kind="manual")
                        self._schedule_next_random_prompt()
                    elif (
                        self.enable_random_prompt_switches
                        and self._next_random_prompt_at is not None
                        and time.monotonic() >= self._next_random_prompt_at
                    ):
                        self._process_prompt_update(track, update_kind="random")
                        self._schedule_next_random_prompt()
                    elif (
                        self.enable_random_prompt_switches
                        and self._next_random_prompt_at is None
                    ):
                        self._schedule_next_random_prompt()
            except Exception as exc:
                logger.error("SPOTIFY-PROMPTS: poll error: %s", exc)

            self._wait_for_next_poll()

    def prepare(self, **kwargs) -> Requirements | None:
        if kwargs.get("video") is not None:
            return Requirements(input_size=1)
        return None

    def _apply_runtime_overrides(self, **kwargs) -> None:
        if "user_idea" in kwargs and isinstance(kwargs["user_idea"], str):
            with self._state_lock:
                self._user_idea = kwargs["user_idea"].strip()

        if "prompt_weight" in kwargs:
            try:
                self.prompt_weight = float(kwargs["prompt_weight"])
            except (TypeError, ValueError):
                pass

        if "reset_cache_on_track_change" in kwargs:
            self.reset_cache_on_track_change = bool(kwargs["reset_cache_on_track_change"])

        if "enable_random_prompt_switches" in kwargs:
            new_enabled = bool(kwargs["enable_random_prompt_switches"])
            if new_enabled != self.enable_random_prompt_switches:
                self.enable_random_prompt_switches = new_enabled
                if not self.enable_random_prompt_switches:
                    self._next_random_prompt_at = None
                else:
                    self._schedule_next_random_prompt()

        if "random_switch_min_seconds" in kwargs:
            try:
                new_min = max(
                    3, int(kwargs["random_switch_min_seconds"])
                )
                if new_min != self.random_switch_min_seconds:
                    self.random_switch_min_seconds = new_min
                    self.random_switch_max_seconds = max(
                        self.random_switch_max_seconds, self.random_switch_min_seconds
                    )
                    self._schedule_next_random_prompt()
            except (TypeError, ValueError):
                pass

        if "random_switch_max_seconds" in kwargs:
            try:
                new_max = max(
                    self.random_switch_min_seconds,
                    int(kwargs["random_switch_max_seconds"]),
                )
                if new_max != self.random_switch_max_seconds:
                    self.random_switch_max_seconds = new_max
                    self._schedule_next_random_prompt()
            except (TypeError, ValueError):
                pass

        if "soft_transition_steps" in kwargs:
            try:
                self.soft_transition_steps = max(0, int(kwargs["soft_transition_steps"]))
            except (TypeError, ValueError):
                pass

        if "soft_transition_method" in kwargs:
            self.soft_transition_method = self._normalize_transition_method(
                str(kwargs["soft_transition_method"])
            )

        if "manual_prompt_refresh_counter" in kwargs:
            try:
                refresh_counter = int(kwargs["manual_prompt_refresh_counter"])
            except (TypeError, ValueError):
                refresh_counter = self.manual_prompt_refresh_counter

            if refresh_counter != self.manual_prompt_refresh_counter:
                self.manual_prompt_refresh_counter = refresh_counter
                self._mark_manual_refresh_requested()
                logger.info("SPOTIFY-PROMPTS: manual refresh requested")

        if "prompt_max_words" in kwargs:
            try:
                self.prompt_max_words = max(12, min(80, int(kwargs["prompt_max_words"])))
            except (TypeError, ValueError):
                pass

    def __call__(self, **kwargs) -> dict:
        self._apply_runtime_overrides(**kwargs)

        video_input = kwargs.get("video")
        if video_input is None:
            return {}

        frame = video_input[0] if isinstance(video_input, list) else video_input
        if frame.dim() == 4:
            frame = frame.squeeze(0)

        frames = frame.unsqueeze(0).to(device=self.device, dtype=torch.float32) / 255.0
        output: dict[str, Any] = {"video": frames.clamp(0, 1)}

        try:
            update = self._prompt_updates.get_nowait()
        except queue.Empty:
            return output

        prompt = update["prompt"].strip()
        if not prompt:
            return output

        prompt_payload = [{"text": prompt, "weight": self.prompt_weight}]
        update_kind = update.get("update_kind", "track_change")

        if update_kind == "track_change":
            output["prompts"] = prompt_payload
            if self.reset_cache_on_track_change:
                output["reset_cache"] = True
            return output

        if self.soft_transition_steps > 0:
            output["transition"] = {
                "target_prompts": prompt_payload,
                "num_steps": self.soft_transition_steps,
                "temporal_interpolation_method": self.soft_transition_method,
            }
        else:
            output["prompts"] = prompt_payload
        return output

    def __del__(self):
        self._stop_event.set()
        self._wake_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3)


class SpotifyPromptOverlayPipeline(Pipeline):
    """Postprocessor that overlays the latest Spotify prompt on output frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return SpotifyPromptOverlayConfig

    def __init__(
        self,
        overlay_enabled: bool = True,
        overlay_opacity: float = 0.72,
        overlay_height_ratio: float = 0.12,
        overlay_font_size: int = 18,
        overlay_max_prompt_chars: int = 160,
        overlay_show_track: bool = True,
        overlay_show_update_kind: bool = True,
        overlay_ticker_speed_px_per_sec: float = 90.0,
        overlay_stale_after_seconds: int = 120,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.overlay_enabled = bool(overlay_enabled)
        self.overlay_opacity = float(max(0.0, min(1.0, overlay_opacity)))
        self.overlay_height_ratio = float(max(0.07, min(0.25, overlay_height_ratio)))
        self.overlay_font_size = int(max(10, min(64, overlay_font_size)))
        self.overlay_max_prompt_chars = int(max(30, min(600, overlay_max_prompt_chars)))
        self.overlay_show_track = bool(overlay_show_track)
        self.overlay_show_update_kind = bool(overlay_show_update_kind)
        self.overlay_ticker_speed_px_per_sec = float(
            max(10.0, min(420.0, overlay_ticker_speed_px_per_sec))
        )
        self.overlay_stale_after_seconds = int(max(1, overlay_stale_after_seconds))
        self._font: Any = None
        self._pil_warned = False
        self._ticker_offset_px = 0.0
        self._ticker_last_tick_time = time.monotonic()
        self._ticker_state_key = ""
        self._ticker_strip_image: Any = None
        self._ticker_strip_key = ""
        self._ticker_strip_width = 1
        self._panel_bg_image: Any = None
        self._panel_bg_key = ""

    def prepare(self, **kwargs) -> Requirements | None:
        if kwargs.get("video") is not None:
            return Requirements(input_size=1)
        return None

    def _apply_runtime_overrides(self, **kwargs) -> None:
        if "overlay_enabled" in kwargs:
            self.overlay_enabled = bool(kwargs["overlay_enabled"])
        if "overlay_opacity" in kwargs:
            try:
                self.overlay_opacity = float(max(0.0, min(1.0, kwargs["overlay_opacity"])))
            except (TypeError, ValueError):
                pass
        if "overlay_height_ratio" in kwargs:
            try:
                self.overlay_height_ratio = float(
                    max(0.07, min(0.25, kwargs["overlay_height_ratio"]))
                )
            except (TypeError, ValueError):
                pass
        if "overlay_font_size" in kwargs:
            try:
                self.overlay_font_size = int(max(10, min(64, kwargs["overlay_font_size"])))
                self._font = None
            except (TypeError, ValueError):
                pass
        if "overlay_max_prompt_chars" in kwargs:
            try:
                self.overlay_max_prompt_chars = int(
                    max(30, min(600, kwargs["overlay_max_prompt_chars"]))
                )
            except (TypeError, ValueError):
                pass
        if "overlay_show_track" in kwargs:
            self.overlay_show_track = bool(kwargs["overlay_show_track"])
        if "overlay_show_update_kind" in kwargs:
            self.overlay_show_update_kind = bool(kwargs["overlay_show_update_kind"])
        if "overlay_ticker_speed_px_per_sec" in kwargs:
            try:
                self.overlay_ticker_speed_px_per_sec = float(
                    max(10.0, min(420.0, kwargs["overlay_ticker_speed_px_per_sec"]))
                )
            except (TypeError, ValueError):
                pass
        if "overlay_stale_after_seconds" in kwargs:
            try:
                self.overlay_stale_after_seconds = int(
                    max(1, kwargs["overlay_stale_after_seconds"])
                )
            except (TypeError, ValueError):
                pass

    def _get_font(self) -> Any:
        if self._font is not None:
            return self._font
        if not _PIL_AVAILABLE:
            return None
        try:
            self._font = ImageFont.truetype("DejaVuSans.ttf", self.overlay_font_size)
        except Exception:
            self._font = ImageFont.load_default()
        return self._font

    @staticmethod
    def _to_uint8_frame(frame: torch.Tensor) -> np.ndarray:
        frame_cpu = frame.detach().to("cpu")
        if frame_cpu.dtype.is_floating_point:
            data = (frame_cpu.clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            data = frame_cpu.to(torch.uint8)
        return data.contiguous().numpy()

    @staticmethod
    def _to_output_video(frame: torch.Tensor) -> torch.Tensor:
        if frame.dtype.is_floating_point:
            tensor = frame.to(dtype=torch.float32).clamp(0, 1)
        else:
            tensor = frame.to(dtype=torch.float32) / 255.0
        return tensor.unsqueeze(0)

    def _is_stale(self, updated_at_iso: str) -> bool:
        if not updated_at_iso:
            return True
        try:
            updated_at = datetime.fromisoformat(updated_at_iso)
            age_seconds = (datetime.now(tz=timezone.utc) - updated_at).total_seconds()
            return age_seconds > self.overlay_stale_after_seconds
        except Exception:
            return True

    def _build_ticker_text(self) -> tuple[str, str]:
        state = get_latest_prompt_state()
        if not state.prompt or self._is_stale(state.updated_at_iso):
            return "", ""

        parts: list[str] = []
        if self.overlay_show_update_kind:
            kind = state.update_kind.replace("_", " ").strip().upper() or "PROMPT"
            parts.append(f"SPOTIFY PROMPT [{kind}]")

        if self.overlay_show_track and state.track_title:
            track = state.track_title
            if state.track_artists:
                track = f"{track} - {state.track_artists}"
            parts.append(track)

        raw_prompt = state.prompt.strip()
        prompt_text = raw_prompt[: self.overlay_max_prompt_chars]
        if len(raw_prompt) > self.overlay_max_prompt_chars:
            prompt_text = prompt_text.rstrip() + "..."
        if prompt_text:
            parts.append(prompt_text)

        ticker_text = " | ".join(p for p in parts if p).strip()
        state_key = f"{state.updated_at_iso}|{ticker_text}"
        return ticker_text, state_key

    @staticmethod
    def _text_width(draw: Any, text: str, font: Any) -> int:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return max(1, int(bbox[2] - bbox[0]))
        except Exception:
            return max(1, int(draw.textlength(text, font=font)))

    def _get_panel_background(self, width: int, panel_height: int) -> Any:
        alpha = int(255 * self.overlay_opacity)
        cache_key = f"{width}|{panel_height}|{alpha}"
        if self._panel_bg_image is not None and cache_key == self._panel_bg_key:
            return self._panel_bg_image

        self._panel_bg_image = Image.new("RGBA", (width, panel_height), (0, 0, 0, alpha))
        self._panel_bg_key = cache_key
        return self._panel_bg_image

    def _ensure_ticker_strip(
        self,
        ticker_text: str,
        state_key: str,
        width: int,
        panel_height: int,
        font: Any,
    ) -> tuple[Any, int]:
        strip_key = (
            f"{state_key}|{width}|{panel_height}|{self.overlay_font_size}"
            f"|{self.overlay_max_prompt_chars}"
        )
        if self._ticker_strip_image is not None and strip_key == self._ticker_strip_key:
            return self._ticker_strip_image, self._ticker_strip_width

        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)
        text_width = self._text_width(probe_draw, ticker_text, font=font)
        gap = max(40, int(self.overlay_font_size * 1.8))
        strip_width = max(1, text_width + gap)

        strip = Image.new("RGBA", (strip_width, panel_height), (0, 0, 0, 0))
        strip_draw = ImageDraw.Draw(strip)
        text_y = max(2, (panel_height - self.overlay_font_size) // 2 - 1)
        strip_draw.text((0, text_y), ticker_text, font=font, fill=(255, 255, 255, 235))

        self._ticker_strip_image = strip
        self._ticker_strip_key = strip_key
        self._ticker_strip_width = strip_width
        return strip, strip_width

    def __call__(self, **kwargs) -> dict:
        self._apply_runtime_overrides(**kwargs)

        video_input = kwargs.get("video")
        if video_input is None:
            return {}

        frame = video_input[0] if isinstance(video_input, list) else video_input
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        passthrough = {"video": self._to_output_video(frame)}

        if not self.overlay_enabled:
            return passthrough

        if not _PIL_AVAILABLE:
            if not self._pil_warned:
                logger.warning(
                    "SPOTIFY-PROMPT-OVERLAY: Pillow is unavailable; overlay disabled."
                )
                self._pil_warned = True
            return passthrough

        ticker_text, state_key = self._build_ticker_text()
        if not ticker_text:
            return passthrough

        now = time.monotonic()
        elapsed = max(0.0, now - self._ticker_last_tick_time)
        self._ticker_last_tick_time = now
        if state_key != self._ticker_state_key:
            self._ticker_state_key = state_key
            self._ticker_offset_px = 0.0

        frame_np = self._to_uint8_frame(frame)
        input_device = frame.device
        height, width, _ = frame_np.shape
        panel_height = max(42, int(height * self.overlay_height_ratio))

        font = self._get_font()
        image = Image.fromarray(frame_np, mode="RGB").convert("RGBA")
        panel = self._get_panel_background(width=width, panel_height=panel_height).copy()
        ticker_strip, strip_width = self._ensure_ticker_strip(
            ticker_text=ticker_text,
            state_key=state_key,
            width=width,
            panel_height=panel_height,
            font=font,
        )

        self._ticker_offset_px += self.overlay_ticker_speed_px_per_sec * elapsed
        if strip_width > 0:
            self._ticker_offset_px %= float(strip_width)
        x = int(width - self._ticker_offset_px)
        while x < width:
            panel.paste(ticker_strip, (x, 0), ticker_strip)
            x += strip_width

        image.alpha_composite(panel, dest=(0, height - panel_height))
        composited = image.convert("RGB")
        composited_np = np.asarray(composited, dtype=np.uint8)
        output_frame = (
            torch.from_numpy(composited_np)
            .unsqueeze(0)
            .to(device=input_device, dtype=torch.float32)
            / 255.0
        )
        return {"video": output_frame.clamp(0, 1)}
