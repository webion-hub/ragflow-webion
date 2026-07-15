#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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


import os
import tiktoken

from common.file_utils import get_project_base_directory

tiktoken_cache_dir = get_project_base_directory()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        code_list = encoder.encode(string)
        return len(code_list)
    except Exception:
        return 0

def total_token_count_from_response(resp):
    """
    Extract token count from LLM response in various formats.

    Handles None responses and different response structures from various LLM providers.
    Returns 0 if token count cannot be determined.
    """
    if resp is None:
        return 0

    try:
        if hasattr(resp, "usage") and hasattr(resp.usage, "total_tokens"):
            return resp.usage.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "usage_metadata") and hasattr(resp.usage_metadata, "total_tokens"):
            return resp.usage_metadata.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "meta") and hasattr(resp.meta, "billed_units") and hasattr(resp.meta.billed_units, "input_tokens"):
            return resp.meta.billed_units.input_tokens
    except Exception:
        pass

    if isinstance(resp, dict) and 'usage' in resp and 'total_tokens' in resp['usage']:
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'usage' in resp and 'input_tokens' in resp['usage'] and 'output_tokens' in resp['usage']:
        try:
            return resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'meta' in resp and 'tokens' in resp['meta'] and 'input_tokens' in resp['meta']['tokens'] and 'output_tokens' in resp['meta']['tokens']:
        try:
            return resp["meta"]["tokens"]["input_tokens"] + resp["meta"]["tokens"]["output_tokens"]
        except Exception:
            pass
    return 0


def usage_from_response(resp) -> dict:
    """Extract a {prompt_tokens, completion_tokens, total_tokens} split from an LLM response.

    Handles OpenAI/OpenRouter-style ``resp.usage`` objects and dict variants. Missing
    fields default to 0; ``total_tokens`` falls back to prompt+completion when absent.
    (Backported to match upstream RAGFlow >=0.26 so future merges stay clean.)
    """
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resp is None:
        return out

    usage = None
    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
    except Exception:
        usage = None
    if usage is None:
        return out

    def _get(obj, *names):
        for n in names:
            try:
                v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            except Exception:
                v = None
            if v:
                return int(v)
        return 0

    out["prompt_tokens"] = _get(usage, "prompt_tokens", "input_tokens")
    out["completion_tokens"] = _get(usage, "completion_tokens", "output_tokens")
    out["total_tokens"] = _get(usage, "total_tokens")
    if not out["total_tokens"]:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def cost_from_response(resp) -> float:
    """Best-effort real USD cost of a completion.

    OpenRouter returns ``usage.cost`` when the request carries ``usage:{include:true}``;
    LiteLLM may also expose a computed cost via ``_hidden_params.response_cost``.
    Returns 0.0 when no cost is available.
    """
    if resp is None:
        return 0.0

    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
        if usage is not None:
            c = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
            if c:
                return float(c)
    except Exception:
        pass

    try:
        hp = getattr(resp, "_hidden_params", None)
        if isinstance(hp, dict) and hp.get("response_cost"):
            return float(hp["response_cost"])
    except Exception:
        pass

    return 0.0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])
