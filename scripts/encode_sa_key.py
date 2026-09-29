"""Base64-encode a Google service account JSON key for `GDRIVE_SA_B64` & friends.

    python scripts/encode_sa_key.py new-sa-key.json              # base64 on stdout
    python scripts/encode_sa_key.py new-sa-key.json --env        # GDRIVE_SA_B64="..."
    python scripts/encode_sa_key.py new-sa-key.json --env --var VERTEX_AI_SA_B64
    python scripts/encode_sa_key.py --verify "$GDRIVE_SA_B64"    # check an existing value

Every consumer in this repo reads these vars with `base64.b64decode(...)` and then
`json.loads`, so the encoding has to be one unwrapped line — `certutil -encode`
and `base64` without `-w0` both wrap, and the decode then fails far from here.
This validates the key before encoding and round-trips the result afterwards, so
a bad paste is caught now rather than as an opaque 401 inside a sandbox.

The base64 IS the private key. It goes to stdout so it can be piped or
redirected; everything else goes to stderr, and the key is never echoed back.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ("client_email", "private_key", "project_id")


def describe(decoded: str, source: str) -> dict:
    """Parse the decoded JSON and fail loudly if it is not a usable SA key."""
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: {source} is not JSON ({e})") from e
    if parsed.get("type") != "service_account":
        raise SystemExit(f"ERROR: expected a service_account key, got type={parsed.get('type')!r}")
    missing = [f for f in REQUIRED_FIELDS if not parsed.get(f)]
    if missing:
        raise SystemExit(f"ERROR: {source} is missing {', '.join(missing)}")
    return parsed


def encode(path: Path) -> tuple[str, dict]:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise SystemExit(f"ERROR: cannot read {path} ({e})") from e
    try:
        parsed = describe(raw.decode("utf-8"), str(path))
    except UnicodeDecodeError as e:
        raise SystemExit(f"ERROR: {path} is not utf-8 text ({e})") from e
    b64 = base64.b64encode(raw).decode("ascii")
    # Round-trip, because the whole point of this script is that the consumers
    # only ever see the encoded form.
    assert json.loads(base64.b64decode(b64).decode("utf-8")) == parsed
    return b64, parsed


def verify(value: str) -> dict:
    stripped = "".join(value.split())
    if stripped != value.strip():
        print("note: value contained whitespace/newlines - strip it before use", file=sys.stderr)
    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"ERROR: not valid base64 utf-8 ({e})") from e
    return describe(decoded, "the decoded value")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key_file", nargs="?", type=Path, help="path to the service account JSON")
    parser.add_argument("--verify", metavar="B64", help="check an existing base64 value instead")
    parser.add_argument("--env", action="store_true", help="print as a ready-to-paste .env line")
    parser.add_argument("--var", default="GDRIVE_SA_B64", help="env var name for --env")
    args = parser.parse_args()

    if args.verify:
        parsed = verify(args.verify)
        print(f"ok: {parsed['client_email']} (project {parsed['project_id']})", file=sys.stderr)
        return 0

    if not args.key_file:
        parser.error("give a key file, or --verify to check an existing value")

    b64, parsed = encode(args.key_file)
    print(f"service account: {parsed['client_email']} (project {parsed['project_id']})", file=sys.stderr)
    print(f"encoded length:  {len(b64)} chars", file=sys.stderr)
    print(f'{args.var}="{b64}"' if args.env else b64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
