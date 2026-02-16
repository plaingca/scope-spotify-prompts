"""Configuration schema for the Spotify prompts preprocessor pipeline."""

from typing import Literal

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
        "that prompt into the downstream pipeline on track changes and optional random/manual refreshes."
    )
    pipeline_version = "0.2.0"

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
        description="Send reset_cache only on track-change prompt updates.",
        json_schema_extra=ui_field_config(
            order=6,
            label="Reset Cache On Track Change",
            category="input",
        ),
    )
    enable_random_prompt_switches: bool = Field(
        default=True,
        description="Generate fresh prompts during a song at random intervals.",
        json_schema_extra=ui_field_config(
            order=7,
            label="Enable Random Prompt Switches",
            category="input",
        ),
    )
    random_switch_min_seconds: int = Field(
        default=20,
        ge=3,
        le=300,
        description="Minimum seconds between random prompt switches.",
        json_schema_extra=ui_field_config(
            order=8,
            label="Random Switch Min (s)",
            category="input",
        ),
    )
    random_switch_max_seconds: int = Field(
        default=45,
        ge=3,
        le=600,
        description="Maximum seconds between random prompt switches.",
        json_schema_extra=ui_field_config(
            order=9,
            label="Random Switch Max (s)",
            category="input",
        ),
    )
    soft_transition_steps: int = Field(
        default=4,
        ge=0,
        le=32,
        description="Transition length for random/manual prompt updates (0 = immediate).",
        json_schema_extra=ui_field_config(
            order=10,
            label="Random/Manual Transition Steps",
            category="input",
        ),
    )
    soft_transition_method: Literal["linear", "slerp"] = Field(
        default="slerp",
        description="Interpolation method for random/manual prompt transitions.",
        json_schema_extra=ui_field_config(
            order=11,
            label="Random/Manual Transition Method",
            category="input",
        ),
    )
    manual_prompt_refresh_counter: int = Field(
        default=0,
        ge=0,
        description="Manual trigger: click +/- to request a new prompt immediately.",
        json_schema_extra=ui_field_config(
            order=12,
            label="Generate New Prompt (+/-)",
            category="input",
        ),
    )
