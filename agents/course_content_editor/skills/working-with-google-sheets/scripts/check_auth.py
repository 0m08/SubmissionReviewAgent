"""Verify the service account can open a Google Sheet.

Usage:
    python check_auth.py --sheet-url "<URL>"

Exits 0 with "OK" on success. On failure, exits non-zero with a stderr
message that tells the agent exactly what to fix (share with SA email,
fix credential path, etc.).
"""

from __future__ import annotations

import argparse
import sys

from _common import open_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", required=True, help="Google Sheet URL")
    args = parser.parse_args()

    sheet = open_sheet(args.sheet_url)
    # If open_sheet succeeded, auth + share permissions are fine.
    title = sheet.title
    print(f"OK — opened '{title}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
