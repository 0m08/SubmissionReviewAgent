import os
import json
import requests
from typing import Optional


GENERATOR_MODEL_NAME: str = "microsoft/mai-image-2.5"
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
MAX_IMAGE_ROUNDS: int = 3
ASPECT_RATIO: str = "16:9"
IMAGE_PIXEL_SIZE: str = "1792x1024"


def get_openrouter_api_key() -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.session_state.get("openrouter_api_key")
        except (ImportError, Exception):
            pass
    return api_key


def call_openrouter(messages: list, extra_body: Optional[dict] = None) -> dict:
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY.")

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "loaded"
    print(f"[ModelConfig] Calling OpenRouter → model: {GENERATOR_MODEL_NAME} (key: {masked_key})")

    payload: dict = {
        "model": GENERATOR_MODEL_NAME,
        "messages": messages,
        "modalities": ["image"],
        "image_config": {
            "aspect_ratio": ASPECT_RATIO,
            "image_size": "2K",
        },
        "provider_options": {
            "openai": {
                "image_config": {
                    "size": IMAGE_PIXEL_SIZE,
                    "aspect_ratio": ASPECT_RATIO,
                }
            }
        },
    }
    if extra_body:
        payload.update(extra_body)

    response = requests.post(
        url=OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter request failed [{response.status_code}]: {response.text}"
        )

    return response.json()
