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
import json

import pytest

from common.openai_models import apply_reasoning_effort, reasoning_effort


class TestReasoningEffort:
    def test_defaults_for_gpt6_models(self):
        assert reasoning_effort("gpt-6.1-terra") == "none"
        assert reasoning_effort("gpt-6-luna") == "low"
        assert reasoning_effort("gpt-6.1-sol") == "low"
        assert reasoning_effort("gpt-6-astra") == "low"

    def test_litellm_provider_prefix_is_ignored(self):
        assert reasoning_effort("openai/gpt-6-luna") == "low"

    def test_unknown_models_are_left_alone(self):
        assert reasoning_effort("gpt-5.2") is None
        assert reasoning_effort("openai/gpt-5-nano") is None

    def test_env_override_merges_with_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", json.dumps({"gpt-6-luna": "medium"}))
        assert reasoning_effort("gpt-6-luna") == "medium"
        assert reasoning_effort("gpt-6.1-terra") == "none"

    def test_malformed_env_override_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "not json")
        assert reasoning_effort("gpt-6-luna") == "low"


class TestApplyReasoningEffort:
    def test_reasoning_model_gets_effort_and_loses_sampling_params(self):
        req = {"model": "openai/gpt-6-luna", "temperature": 0.9, "top_p": 0.3, "presence_penalty": 0.4, "frequency_penalty": 0.7, "stream": True}
        out = apply_reasoning_effort("openai/gpt-6-luna", req)
        assert out["extra_body"] == {"reasoning_effort": "low"}
        for p in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            assert p not in out
        assert out["stream"] is True

    def test_effort_none_keeps_sampling_params(self):
        req = {"model": "openai/gpt-6.1-terra", "temperature": 0.1}
        out = apply_reasoning_effort("openai/gpt-6.1-terra", req)
        assert out["extra_body"] == {"reasoning_effort": "none"}
        assert out["temperature"] == 0.1

    def test_existing_extra_body_is_preserved(self):
        caller_extra_body = {"foo": "bar"}
        out = apply_reasoning_effort("gpt-6-luna", {"extra_body": caller_extra_body})
        assert out["extra_body"] == {"foo": "bar", "reasoning_effort": "low"}
        assert caller_extra_body == {"foo": "bar"}

    def test_unknown_model_is_untouched(self):
        req = {"model": "openai/gpt-5.2", "temperature": 0.2}
        assert apply_reasoning_effort("openai/gpt-5.2", dict(req)) == req


def test_litellm_prices_are_registered():
    litellm = pytest.importorskip("litellm")
    from common.openai_models import register_litellm_prices

    register_litellm_prices()
    cost = litellm.completion_cost(
        model="openai/gpt-6-luna",
        prompt="",
        completion="",
        custom_llm_provider="openai",
        completion_response={"model": "gpt-6-luna", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000}, "choices": []},
    )
    assert cost == pytest.approx(0.10 + 0.50)
