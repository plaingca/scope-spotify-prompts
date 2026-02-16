"""Scope plugin registration for Spotify prompt generation."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register pipelines provided by this plugin."""
    from .pipelines.pipeline import SpotifyPromptsPipeline

    register(SpotifyPromptsPipeline)
