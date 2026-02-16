"""Spotify prompts preprocessor pipeline for Daydream Scope."""

from __future__ import annotations

import base64
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import SpotifyPromptsConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)
_DOTENV_LOCK = threading.Lock()
_DOTENV_LOADED = False


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

    def generate_visual_prompt(self, track: SpotifyTrack, user_idea: str) -> str:
        user_block = (
            "Generate a single cinematic video prompt for a diffusion model.\n"
            f"Song title: {track.title}\n"
            f"Artist(s): {track.artists}\n"
            f"Album: {track.album}\n"
            f"User creative direction: {user_idea or 'None'}\n"
            "Constraints: vivid visual language, present tense, <= 80 words, "
            "safe for work, no artist names, no camera metadata unless useful."
        )

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
            "max_output_tokens": 220,
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
    """Poll Spotify and inject generated prompts on track change."""

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

        self._state_lock = threading.Lock()
        self._user_idea = user_idea.strip()
        self._last_track_id: str | None = None
        self._last_prompt: str = ""

        self._prompt_updates: queue.Queue[dict[str, str]] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
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

    def _fallback_prompt(self, track: SpotifyTrack) -> str:
        with self._state_lock:
            user_idea = self._user_idea

        suffix = f" Extra direction: {user_idea}." if user_idea else ""
        return (
            f"Cinematic abstract reinterpretation of '{track.title}' by {track.artists}, "
            f"rhythmic motion, luminous particles, rich contrast, immersive atmosphere, "
            f"smooth temporal continuity, detailed textures, dynamic but coherent scene evolution."
            f"{suffix}"
        )

    def _prompt_for_track(self, track: SpotifyTrack) -> str:
        if self.openai_client is None:
            return self._fallback_prompt(track)

        with self._state_lock:
            user_idea = self._user_idea
        try:
            return self.openai_client.generate_visual_prompt(track, user_idea)
        except Exception as exc:
            logger.error(
                "SPOTIFY-PROMPTS: OpenAI generation failed, using fallback: %s", exc
            )
            return self._fallback_prompt(track)

    def _enqueue_prompt_update(self, prompt: str) -> None:
        payload = {"prompt": prompt}
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

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.spotify_client is None:
                    return

                track = self.spotify_client.get_current_track()
                if track and track.is_playing and track.track_id:
                    if track.track_id != self._last_track_id:
                        self._last_track_id = track.track_id
                        prompt = self._prompt_for_track(track)
                        if prompt and prompt != self._last_prompt:
                            self._last_prompt = prompt
                            self._enqueue_prompt_update(prompt)
                            logger.info(
                                "SPOTIFY-PROMPTS: track='%s' artists='%s'",
                                track.title,
                                track.artists,
                            )
            except Exception as exc:
                logger.error("SPOTIFY-PROMPTS: poll error: %s", exc)

            self._stop_event.wait(self.poll_interval_seconds)

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

        output["prompts"] = [{"text": prompt, "weight": self.prompt_weight}]
        if self.reset_cache_on_track_change:
            output["reset_cache"] = True
        return output

    def __del__(self):
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3)
