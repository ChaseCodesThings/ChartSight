"""ChartSight: verified, layered alt text for charts in scientific papers."""
from .config import ModelConfig, PipelineConfig, model_from_env
from .pipeline import describe_chart
from .schema import AltText, ChartData, ChartResult

__all__ = ["ModelConfig", "PipelineConfig", "model_from_env", "describe_chart", "AltText", "ChartData", "ChartResult"]
