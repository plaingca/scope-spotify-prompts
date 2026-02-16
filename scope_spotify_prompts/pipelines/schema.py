"""Configuration schema for the Spotify prompts preprocessor pipeline."""

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class SpotifyPromptsConfig(BasePipelineConfig):
    """Configuration for Spotify-driven prompt generation."""

    pipeline_id = "spotify-prompts"
    pipeline_name = "Spotify Prompts"
    pipeline_description = (
        "Polls Spotify now-playing, generates a visual prompt with OpenAI, and injects "
        "that prompt into the downstream pipeline when tracks change."
    )
    pipeline_version = "0.1.0"

    supports_prompts = True
    modes = {"video": ModeDefaults(input_size=1, default=True)}
    usage = [UsageType.PREPROCESSOR]

    requires_models = False
    estimated_vram_gb = 0.1

    poll_interval_seconds: float = Field(
        default=8.0,
        ge=2.0,
        le=120.0,
        description="How often to poll Spotify currently-playing.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Spotify Poll Interval (s)",
            is_load_param=True,
        ),
    )
    openai_model: str = Field(
        default="gpt-5.2",
        min_length=1,
        description="OpenAI model used for prompt generation.",
        json_schema_extra=ui_field_config(
            order=2,
            label="OpenAI Model",
            is_load_param=True,
        ),
    )
    openai_timeout_seconds: int = Field(
        default=45,
        ge=10,
        le=180,
        description="Timeout for OpenAI prompt requests.",
        json_schema_extra=ui_field_config(
            order=3,
            label="OpenAI Timeout (s)",
            is_load_param=True,
        ),
    )
    user_idea: str = Field(
        default="",
        description="Optional creative direction mixed into generated prompts.",
        json_schema_extra=ui_field_config(
            order=4,
            label="Creative Direction",
            category="input",
        ),
    )
    prompt_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Weight assigned to injected prompt text.",
        json_schema_extra=ui_field_config(
            order=5,
            label="Prompt Weight",
            category="input",
        ),
    )
    reset_cache_on_track_change: bool = Field(
        default=True,
        description="Send reset_cache when track changes to force a clean transition.",
        json_schema_extra=ui_field_config(
            order=6,
            label="Reset Cache On Track Change",
            category="input",
        ),
    )
