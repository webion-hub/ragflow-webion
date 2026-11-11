#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" AND BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Request settings for the OpenAI GPT-6 family.

RAGFlow never sends a reasoning effort, so every model runs at the API
default. GPT-6 models default to ``medium``, which is slower and spends far
more output tokens than the GPT-5 models they replace (gpt-5.2 defaults to
``none``). OpenAI's migration guide says to keep the effort you had, so each
model gets an explicit effort here.

The bundled LiteLLM does not know these models either: it drops
``reasoning_effort`` and keeps ``temperature`` (which the API rejects once
reasoning is on), so the effort goes through ``extra_body`` and sampling
parameters are removed when needed.
"""

import json
import logging
import os

# Per-model effort. Sol and Astra have no "none"; "low" is their minimum.
# Override without a rebuild via OPENAI_REASONING_EFFORT='{"gpt-6-luna": "medium"}'.
DEFAULT_REASONING_EFFORT = {
    "gpt-6.1-terra": "none",
    "gpt-6-luna": "low",
    "gpt-6.1-sol": "low",
    "gpt-6-astra": "low",
}

# USD per 1M tokens: (input, cached input, output), Standard tier.
_PRICES_PER_MILLION = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
}

# Rejected by the API whenever reasoning effort is not "none".
_SAMPLING_PARAMS = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs")


def _base_name(model_name: str) -> str:
    # LiteLLM model names carry a provider prefix, e.g. "openai/gpt-6-luna".
    return (model_name or "").split("/", 1)[-1]


def reasoning_effort(model_name: str) -> str | None:
    efforts = DEFAULT_REASONING_EFFORT
    override = os.environ.get("OPENAI_REASONING_EFFORT")
    if override:
        try:
            efforts = {**DEFAULT_REASONING_EFFORT, **json.loads(override)}
        except (ValueError, TypeError):
            logging.warning("Ignoring malformed OPENAI_REASONING_EFFORT: %r", override)
    return efforts.get(_base_name(model_name))


def apply_reasoning_effort(model_name: str, request: dict) -> dict:
    """Add the model's reasoning effort to an OpenAI chat request, in place."""
    effort = reasoning_effort(model_name)
    if effort is None:
        return request
    request["extra_body"] = {**(request.get("extra_body") or {}), "reasoning_effort": effort}
    if effort != "none":
        for param in _SAMPLING_PARAMS:
            request.pop(param, None)
    return request


def register_litellm_prices() -> None:
    """Make LiteLLM compute the cost of GPT-6 calls without its remote cost map."""
    import litellm

    litellm.register_model(
        {
            name: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": inp / 1e6,
                "cache_read_input_token_cost": cached / 1e6,
                "output_cost_per_token": out / 1e6,
            }
            for name, (inp, cached, out) in _PRICES_PER_MILLION.items()
        }
    )
