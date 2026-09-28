"""Run the session on the model the user picked, not only the deploy-time one.

## Why a middleware

The deployed model is fixed when `define_deep_agent(model=...)` is compiled.
The two overrides that look like they should change it are accepted and
ignored: `configurable.model` is only read on the `mda eval` path, and
`context.model` reaches `runtime.context` but nothing in the stock stack reads
it. Swapping the model per call inside `wrap_model_call` is the documented route
(docs.langchain.com/oss/python/deepagents/models#select-a-model-at-runtime).

## Why it is on the editor too

Subagents keep the deploy-time model unless they carry the same middleware, and
the editor does most of the actual writing. Picking GPT and getting a GPT
coordinator briefing Gemini editors would be the choice half-honoured, so
`agent.py` gives this to both. The editor reads the same `runtime.context` the
coordinator was started with.

## What stays at the deploy-time model regardless

The Deep Agents harness profile (provider-tuned prompt and tool-description
tweaks) is chosen from `CCE_MODEL` at deploy. A session on another provider
runs on that profile. Test a provider here before offering it, rather than
assuming it behaves as it would on its own deployment.

## Switching mid-thread

Not supported, and not guarded here: the page fixes the model when a thread is
created. Provider-specific reasoning blocks from one model in the history can
fail on another.
"""

from __future__ import annotations

from functools import cache

from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.chat_models import init_chat_model

# The only models a caller may name. Keys are what `context.model` carries;
# the page's picker lists the same keys. Each provider's API key must be in
# `.env`, which `mda deploy` forwards as a secret, even when CCE_MODEL is
# another provider.
_ALLOWED: dict[str, dict] = {
    "google_genai:gemini-3.8-flash": {},
    # Explicit, because plain ChatOpenAI falls back to Chat Completions for any
    # model not on its Responses-only list. deepagents sets this itself for an
    # `openai:` deploy-time model; a model built here does not get that.
    # Effort is explicit because the default reasons very little: in the
    # 2026-09-24 eval it averaged ~100 reasoning tokens a call against Gemini's
    # ~1,400, and deleted whole slide roles the house rules say to keep.
    "openai:gpt-6-sol": {"use_responses_api": True, "reasoning": {"effort": "medium"}},
    "anthropic:claude-sonnet-5": {},
}


@cache
def _model(spec: str):
    # Built on first use rather than at import, so a missing key for one
    # provider fails the session that picked it, not the whole deployment.
    return init_chat_model(spec, **_ALLOWED[spec])


@wrap_model_call
async def select_model(request: ModelRequest, handler):
    """Swap in the model named by `runtime.context.model`, if one was named.

    Async for the same reason as `editor_identity`: the Agent Server drives the
    graph with `astream`, and a sync hook raises there.
    """
    context = getattr(request.runtime, "context", None) or {}
    if isinstance(context, dict):
        spec = context.get("model")
    else:
        spec = getattr(context, "model", None)

    if not spec:
        # No choice made (direct API calls, older threads): deploy-time model.
        return await handler(request)
    if spec not in _ALLOWED:
        # Refused, not ignored. Falling back would run the session on a model
        # the user did not pick while the page says it is on the one they did.
        raise ValueError(f"Model {spec!r} is not offered by this deployment.")
    return await handler(request.override(model=_model(spec)))
