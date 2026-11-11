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
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import pytest

pytest.importorskip("litellm")

from common.openai_models import register_litellm_prices
from common.token_utils import cost_from_usage

register_litellm_prices()

LUNA_IN, LUNA_CACHED, LUNA_OUT = 0.10e-6, 0.01e-6, 0.50e-6


def _resp(prompt, completion, cached=0, reasoning=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return {"usage": usage}


def test_bills_every_completion_token_even_with_reasoning_details():
    # LiteLLM 1.80.x bills only the 14 reasoning tokens here; all 39 are output.
    cost = cost_from_usage("openai/gpt-6-luna", _resp(244, 39, reasoning=14))
    assert cost == pytest.approx(244 * LUNA_IN + 39 * LUNA_OUT)


def test_cached_prompt_tokens_use_the_cache_price():
    cost = cost_from_usage("openai/gpt-6-luna", _resp(2000, 100, cached=1500))
    assert cost == pytest.approx(500 * LUNA_IN + 1500 * LUNA_CACHED + 100 * LUNA_OUT)


def test_works_on_usage_objects_like_stream_chunks():
    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 1000, 10, 1010
        prompt_tokens_details = None
        completion_tokens_details = None

    class Chunk:
        usage = Usage()

    assert cost_from_usage("openai/gpt-6-luna", Chunk()) == pytest.approx(1000 * LUNA_IN + 10 * LUNA_OUT)


def test_models_from_litellm_price_map_are_priced_too():
    # Bare-name entries (LiteLLM's own map) resolve from the "openai/" prefixed name.
    assert cost_from_usage("openai/gpt-5-mini", _resp(1_000_000, 0)) == pytest.approx(0.25)


@pytest.mark.parametrize("model, resp", [
    ("openai/not-a-real-model", _resp(100, 10)),
    ("openai/gpt-6-luna", {"usage": None}),
    ("openai/gpt-6-luna", {}),
    ("openai/gpt-6-luna", _resp(0, 0)),
])
def test_returns_zero_when_cost_is_unknown(model, resp):
    assert cost_from_usage(model, resp) == 0.0
