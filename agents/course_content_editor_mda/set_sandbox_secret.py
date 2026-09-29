"""Store the service account JSON as a LangSmith workspace secret. Run once.

    python set_sandbox_secret.py            # store it
    python set_sandbox_secret.py --list     # show which keys exist (names only)

This is the one step that sends a credential anywhere, so it is a separate,
explicit script rather than something a deploy does quietly.

It reads `GDRIVE_SA_B64` from `.env`, decodes it, and POSTs it to
`/workspaces/current/secrets` under the name `sandbox/__init__.py` refers to.
From there the auth proxy resolves it when a sandbox makes a Google API call —
the secret itself never enters the sandbox and is never returned by the API.

Nothing here prints the key, or any part of it.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT))

from langsmith import Client  # noqa: E402

from sandbox import SA_SECRET  # noqa: E402


def existing_keys(client: Client) -> list[str]:
    response = client.request_with_retries("GET", "/workspaces/current/secrets")
    response.raise_for_status()
    return sorted(str(s.get("key")) for s in response.json())


def service_account_json() -> str:
    """The decoded SA JSON from `.env`, validated as JSON before it is sent."""
    raw = os.environ.get("GDRIVE_SA_B64")
    if not raw:
        raise SystemExit("ERROR: GDRIVE_SA_B64 not set in .env")
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as e:
        raise SystemExit(f"ERROR: GDRIVE_SA_B64 is not valid base64 utf-8 ({e})") from e
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: decoded GDRIVE_SA_B64 is not JSON ({e})") from e
    # Fail here rather than at the first sheet call inside a sandbox, where the
    # error surfaces as an opaque 401 with no hint about which end is wrong.
    if parsed.get("type") != "service_account":
        raise SystemExit(f"ERROR: expected a service_account key, got type={parsed.get('type')!r}")
    print(f"  service account: {parsed.get('client_email', '(no client_email)')}")
    return decoded


def main() -> int:
    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        return int(bool(print("ERROR: LANGSMITH_API_KEY not set in .env"))) or 2
    client = Client(api_key=key)

    if "--list" in sys.argv:
        print(f"workspace secrets: {existing_keys(client)}")
        return 0

    before = existing_keys(client)
    print(f"workspace secrets before: {before}")
    if SA_SECRET in before:
        print(f"  {SA_SECRET} already set — POSTing again overwrites it.")

    value = service_account_json()
    response = client.request_with_retries(
        "POST", "/workspaces/current/secrets", json=[{"key": SA_SECRET, "value": value}]
    )
    response.raise_for_status()

    after = existing_keys(client)
    print(f"workspace secrets after:  {after}")
    ok = SA_SECRET in after
    print("stored" if ok else "FAILED: secret did not appear")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
