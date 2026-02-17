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
    pipeline_version = "0.2.2"

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
            category="configuration",
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
            category="configuration",
        ),
    )
    reset_cache_on_track_change: bool = Field(
        default=True,
        description="Send reset_cache only on track-change prompt updates.",
        json_schema_extra=ui_field_config(
            order=6,
            label="Reset Cache On Track Change",
            category="configuration",
        ),
    )
    enable_random_prompt_switches: bool = Field(
        default=True,
        description="Generate fresh prompts during a song at random intervals.",
        json_schema_extra=ui_field_config(
            order=7,
            label="Enable Random Prompt Switches",
            category="configuration",
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
            category="configuration",
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
            category="configuration",
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
            category="configuration",
        ),
    )
    soft_transition_method: Literal["linear", "slerp"] = Field(
        default="slerp",
        description="Interpolation method for random/manual prompt transitions.",
        json_schema_extra=ui_field_config(
            order=11,
            label="Random/Manual Transition Method",
            category="configuration",
        ),
    )
    manual_prompt_refresh_counter: int = Field(
        default=0,
        ge=0,
        description="Manual trigger: click +/- to request a new prompt immediately.",
        json_schema_extra=ui_field_config(
            order=12,
            label="Generate New Prompt (+/-)",
            category="configuration",
        ),
    )


class SpotifyPromptOverlayConfig(BasePipelineConfig):
    """Configuration for Spotify prompt visualization overlay postprocessor."""

    pipeline_id = "spotify-prompt-overlay"
    pipeline_name = "Spotify Prompt Overlay"
    pipeline_description = (
        "Draws the most recent Spotify prompt at the bottom of output frames."
    )
    pipeline_version = "0.1.0"

    supports_prompts = False
    modes = {"video": ModeDefaults(input_size=1, default=True)}
    usage = [UsageType.POSTPROCESSOR]

    requires_models = False
    estimated_vram_gb = 0.1

    overlay_enabled: bool = Field(
        default=True,
        description="Enable overlay rendering on output frames.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Enable Overlay",
            category="configuration",
        ),
    )
    overlay_opacity: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Background opacity of the bottom overlay panel.",
        json_schema_extra=ui_field_config(
            order=2,
            label="Overlay Opacity",
            category="configuration",
        ),
    )
    overlay_height_ratio: float = Field(
        default=0.22,
        ge=0.1,
        le=0.45,
        description="Height of the overlay panel as a fraction of frame height.",
        json_schema_extra=ui_field_config(
            order=3,
            label="Overlay Height Ratio",
            category="configuration",
        ),
    )
    overlay_font_size: int = Field(
        default=18,
        ge=10,
        le=64,
        description="Overlay font size in pixels.",
        json_schema_extra=ui_field_config(
            order=4,
            label="Overlay Font Size",
            category="configuration",
        ),
    )
    overlay_max_prompt_chars: int = Field(
        default=220,
        ge=30,
        le=600,
        description="Maximum prompt characters to render before truncation.",
        json_schema_extra=ui_field_config(
            order=5,
            label="Overlay Max Prompt Chars",
            category="configuration",
        ),
    )
    overlay_show_track: bool = Field(
        default=True,
        description="Show track title and artists in the overlay.",
        json_schema_extra=ui_field_config(
            order=6,
            label="Show Track Info",
            category="configuration",
        ),
    )
    overlay_show_update_kind: bool = Field(
        default=True,
        description="Show prompt update source (track change, random, manual).",
        json_schema_extra=ui_field_config(
            order=7,
            label="Show Update Kind",
            category="configuration",
        ),
    )
    overlay_stale_after_seconds: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="Hide overlay when last prompt update is older than this age.",
        json_schema_extra=ui_field_config(
            order=8,
            label="Overlay Stale Timeout (s)",
            category="configuration",
        ),
    )
