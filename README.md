# Scope Spotify Prompts Plugin

Spotify-driven prompt generation plugin for Daydream Scope.

This plugin runs as a Scope preprocessor:

1. Poll Spotify for the currently playing track.
2. Generate a visual prompt using OpenAI Responses API.
3. Inject hard-cut prompt updates on track change (with optional `reset_cache=true`).
4. Optionally inject random or manual prompt refreshes during the same track.
5. Use soft prompt transitions for random/manual refreshes (no hard cache clear).

## Setup

Recommended: create a `.env` file in the Scope working directory (Scope root).
The plugin auto-loads that file on startup.

Example `.env`:

```dotenv
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REFRESH_TOKEN=...
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

You can also set environment variables directly where the Scope backend process can read them:

```powershell
$env:SPOTIFY_CLIENT_ID="..."
$env:SPOTIFY_CLIENT_SECRET="..."
$env:SPOTIFY_REFRESH_TOKEN="..."

$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

## Install (local editable)

From the Scope environment:

```powershell
uv run daydream-scope install -e D:\scope-spotify\scope-spotify-prompts
```

Or from Scope UI plugin settings, install the same local folder path.

## Usage

1. Add `Spotify Prompts` as a preprocessor.
2. Choose your main generation pipeline as downstream.
3. Keep stream input mode as `video` when using video passthrough pipelines.
4. Configure:
   - `Spotify Poll Interval (s)`
   - `OpenAI Model`
   - `Creative Direction` (runtime)
   - `Prompt Weight` (runtime)
   - `Reset Cache On Track Change` (runtime)
   - `Enable Random Prompt Switches` (runtime)
   - `Random Switch Min (s)` / `Random Switch Max (s)` (runtime)
   - `Random/Manual Transition Steps` + `Random/Manual Transition Method` (runtime)
   - `Generate New Prompt (+/-)` (runtime manual trigger)

## Manual Trigger

Scope currently renders plugin runtime params as generic schema controls.
For this plugin, the `Generate New Prompt (+/-)` runtime field is the manual trigger:

- Click `+` (or `-`) once to request a new prompt immediately.
- The plugin detects the value change and generates a fresh prompt without a hard cut.

## Notes

- If Spotify credentials are missing, the plugin logs a warning and stays idle.
- If OpenAI credentials are missing or API calls fail, a fallback prompt template is used.
