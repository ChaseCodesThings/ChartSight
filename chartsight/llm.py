"""One small interface over the model providers, with token, latency and cost logging."""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import ModelConfig
from .schema import Usage


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._client = None

    # ---- public -------------------------------------------------------------
    def complete(self, stage: str, system: str, prompt: str, images: list[bytes] | None = None) -> LLMResponse:
        start = time.time()
        text, tin, tout = self._call(stage, system, prompt, images or [])
        usage = Usage(stage=stage, model=self.cfg.label, input_tokens=tin, output_tokens=tout,
                      seconds=round(time.time() - start, 3), cost_usd=self._cost(tin, tout))
        return LLMResponse(text=text, usage=usage)

    def complete_json(self, stage: str, system: str, prompt: str, images: list[bytes] | None = None,
                      retries: int = 1) -> tuple[dict, list[Usage]]:
        """Ask for JSON; on a parse failure, ask once more with the error shown to the model."""
        usages = []
        p = prompt
        for attempt in range(retries + 1):
            r = self.complete(stage, system, p, images)
            usages.append(r.usage)
            try:
                return parse_json(r.text), usages
            except ValueError as e:
                p = (prompt + f"\n\nYour previous reply could not be parsed as JSON ({e}). "
                     "Reply with ONLY the JSON object, no prose and no code fences.")
        raise ValueError(f"model did not return valid JSON after {retries + 1} attempts")

    # ---- internals ----------------------------------------------------------
    def _cost(self, tin: int, tout: int) -> Optional[float]:
        if self.cfg.price_in_per_mtok is None or self.cfg.price_out_per_mtok is None:
            return None
        return round(tin / 1e6 * self.cfg.price_in_per_mtok + tout / 1e6 * self.cfg.price_out_per_mtok, 6)

    def _call(self, stage: str, system: str, prompt: str, images: list[bytes]) -> tuple[str, int, int]:
        p = self.cfg.provider
        if p == "anthropic":
            return self._anthropic(system, prompt, images)
        if p == "openai":
            return self._openai(system, prompt, images)
        if p == "mock":
            return MOCK_HANDLER(stage, system, prompt, images)
        raise ValueError(f"unknown provider '{p}'")

    def _anthropic(self, system, prompt, images):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.cfg.api_key) if self.cfg.api_key else anthropic.Anthropic()
        content = [{"type": "image", "source": {"type": "base64", "media_type": _mime(b),
                                                 "data": base64.b64encode(b).decode()}} for b in images]
        content.append({"type": "text", "text": prompt})
        msg = self._client.messages.create(model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                                           temperature=self.cfg.temperature, system=system,
                                           messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens

    def _openai(self, system, prompt, images):
        import openai

        if self._client is None:
            kw = {}
            if self.cfg.base_url:
                kw["base_url"] = self.cfg.base_url
            if self.cfg.api_key:
                kw["api_key"] = self.cfg.api_key
            elif self.cfg.base_url:
                kw["api_key"] = "not-needed"  # local servers such as Ollama ignore the key
            self._client = openai.OpenAI(**kw)
        content = [{"type": "image_url", "image_url": {"url": f"data:{_mime(b)};base64,{base64.b64encode(b).decode()}"}}
                   for b in images]
        content.append({"type": "text", "text": prompt})
        r = self._client.chat.completions.create(model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                                                 temperature=self.cfg.temperature,
                                                 messages=[{"role": "system", "content": system},
                                                           {"role": "user", "content": content}])
        u = r.usage
        return r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)


def _mime(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and stray prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(str(e))
    raise ValueError("unbalanced JSON object")


# ---- offline mock -------------------------------------------------------------
# The mock lets you run the whole app and evaluation harness with no API key.
# Tests replace MOCK_HANDLER to script specific replies.

def _default_mock(stage: str, system: str, prompt: str, images: list[bytes]) -> tuple[str, int, int]:
    if stage == "extract":
        out = {"chart_type": "bar", "title": "Mock chart", "x_axis": {"title": "Year", "scale": "categorical"},
               "y_axis": {"title": "Value", "unit": "%", "scale": "linear", "min": 0, "max": 50},
               "series": [{"name": None, "points": [{"x": "2019", "y": 12}, {"x": "2020", "y": 30}, {"x": "2021", "y": 45}]}],
               "confidence": "high", "notes": "mock extraction"}
        return json.dumps(out), 800, 150
    if stage == "repair":
        return json.dumps({"sentences": ["Values rise from 12% in 2019 to 45% in 2021."]}), 300, 40
    if stage in ("generate", "direct"):
        out = {"summary": "Bar chart showing the value rising each year from 2019 to 2021.",
               "construction": ["A bar chart titled Mock chart with years 2019 to 2021 on the x-axis and value in percent on the y-axis, ranging from 0 to 50."],
               "statistics": ["The lowest value is 12% in 2019 and the highest is 45% in 2021."],
               "trends": ["Values rise steadily, more than tripling over the three years."]}
        return json.dumps(out), 900, 160
    if stage == "judge":
        return json.dumps({"accuracy": 4, "completeness": 4, "clarity": 5, "trend": 4, "reason": "mock"}), 600, 40
    return "{}", 10, 2


MOCK_HANDLER: Callable[[str, str, str, list[bytes]], tuple[str, int, int]] = _default_mock
