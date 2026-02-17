"""Scope Spotify Prompts - Active Prompt Monitor
Always-on-top overlay showing the current prompt injected by the Spotify preprocessor.

Reads Scope log lines from:
  %APPDATA%\\Daydream Scope\\logs\\main.log

Looks for lines in this format:
  SPOTIFY-PROMPTS: >>> NEW PROMPT (<kind>): '<text>'
"""

from __future__ import annotations

import os
import re
import threading
import time
import tkinter as tk

LOG_PATH = os.path.expandvars(r"%APPDATA%\Daydream Scope\logs\main.log")

RE_NEW_PROMPT = re.compile(r"SPOTIFY-PROMPTS: >>> NEW PROMPT \(([^)]+)\): '(.+?)'")
RE_UPDATE = re.compile(
    r"SPOTIFY-PROMPTS: update=([a-z_]+) track='(.+?)' artists='(.+?)'"
)

KIND_COLORS = {
    "track_change": "#ff6b6b",
    "random": "#4ecdc4",
    "manual": "#ffd166",
}


class SpotifyPromptMonitor:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Scope Spotify Prompt")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#111827")
        self.root.geometry("650x170+20+20")
        self.root.resizable(True, True)

        self.status_var = tk.StringVar(value="waiting for spotify prompt updates...")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#9ca3af",
            bg="#111827",
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=10, pady=(8, 0))

        self.track_var = tk.StringVar(value="")
        self.track_label = tk.Label(
            self.root,
            textvariable=self.track_var,
            font=("Segoe UI", 9),
            fg="#6ee7b7",
            bg="#111827",
            anchor="w",
        )
        self.track_label.pack(fill="x", padx=10, pady=(2, 0))

        self.prompt_var = tk.StringVar(value="(no prompt seen yet)")
        self.prompt_label = tk.Label(
            self.root,
            textvariable=self.prompt_var,
            font=("Segoe UI Semibold", 13),
            fg="#e5e7eb",
            bg="#111827",
            anchor="w",
            justify="left",
            wraplength=620,
        )
        self.prompt_label.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        self._running = True
        self._thread = threading.Thread(target=self._tail_log, daemon=True)
        self._thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._running = False
        self.root.destroy()

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_track(self, text: str) -> None:
        self.root.after(0, lambda: self.track_var.set(text))

    def _set_prompt(self, text: str, kind: str) -> None:
        def _apply() -> None:
            self.prompt_var.set(text)
            self.prompt_label.configure(fg=KIND_COLORS.get(kind, "#e5e7eb"))

        self.root.after(0, _apply)

    def _tail_log(self) -> None:
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while self._running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.25)
                        continue

                    if "SPOTIFY-PROMPTS" not in line:
                        continue

                    m_update = RE_UPDATE.search(line)
                    if m_update:
                        kind, track, artists = m_update.groups()
                        self._set_status(f"update={kind}")
                        self._set_track(f"{track} — {artists}")
                        continue

                    m_prompt = RE_NEW_PROMPT.search(line)
                    if m_prompt:
                        kind, text = m_prompt.groups()
                        self._set_status(f"new prompt ({kind})")
                        self._set_prompt(text, kind.strip().lower())
        except FileNotFoundError:
            self._set_status("log file not found")
            self._set_prompt(LOG_PATH, "track_change")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    SpotifyPromptMonitor().run()
