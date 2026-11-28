"""Settings read from environment variables (or a .env file loaded by the app)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class ModelConfig:
    """Which model to call and what it costs.

    provider:
      "anthropic"  - Claude models (needs ANTHROPIC_API_KEY)
      "openai"     - OpenAI models, or ANY OpenAI-compatible server such as Ollama or vLLM
                     running an open-weight vision model (set base_url)
      "mock"       - offline fake model for tests and UI work without spending credits
    Prices are per million tokens. Fill them in from your provider's pricing page so cost
    per figure is reported; leave them unset and only token counts are reported.
    """

    provider: str = "mock"
    model: str = "mock-model"
    base_url: str | None = None
    api_key: str | None = None
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    max_tokens: int = 2000
    temperature: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def model_from_env(prefix: str = "CHARTSIGHT") -> ModelConfig:
    def f(name):
        v = _env(f"{prefix}_{name}")
        return float(v) if v is not None else None

    provider = _env(f"{prefix}_PROVIDER", "mock")
    return ModelConfig(
        provider=provider,
        model=_env(f"{prefix}_MODEL", "mock-model" if provider == "mock" else ""),
        base_url=_env(f"{prefix}_BASE_URL"),
        api_key=_env(f"{prefix}_API_KEY"),
        price_in_per_mtok=f("PRICE_IN"),
        price_out_per_mtok=f("PRICE_OUT"),
    )


@dataclass
class PipelineConfig:
    """Switches used by the app and by the ablation study."""

    use_extraction: bool = True  # chart image -> structured data first (off = direct image-to-text)
    use_context: bool = True  # retrieve paper paragraphs about the figure
    use_verification: bool = True  # check numbers and claims against extracted data
    repair_rounds: int = 1  # how many times to ask the model to fix unsupported sentences
    rel_tolerance: float = 0.05  # numbers within 5% of a data value count as supported
    axis_tolerance: float = 0.02  # ... or within 2% of the y-axis range
    context_k: int = 3
    model: ModelConfig = field(default_factory=model_from_env)

    @classmethod
    def ablation(cls, name: str, model: ModelConfig | None = None) -> "PipelineConfig":
        presets = {
            "full": {},
            "no_verify": {"use_verification": False},
            "no_context": {"use_context": False},
            "no_extraction": {"use_extraction": False},
            "direct": {"use_extraction": False, "use_context": False, "use_verification": False},
        }
        if name not in presets:
            raise ValueError(f"unknown system '{name}', choose from {sorted(presets)}")
        cfg = cls(**presets[name])
        if model is not None:
            cfg.model = model
        return cfg
