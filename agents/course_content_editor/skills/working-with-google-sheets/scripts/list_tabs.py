"""List all tab names in a Google Sheet, one per line.

Usage:
    python list_tabs.py --sheet-url "<URL>"

Used when the agent needs to verify which tabs exist before invoking
prepare_workspace.py with --source-tab / --outline-tab. The default tab
names ("Slide Chunks", "Final Outline") sometimes vary by course.
"""

from __future__ import annotations

import argparse
import sys

from _common import open_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", required=True)
    args = parser.parse_args()

    sheet = open_sheet(args.sheet_url)
    for ws in sheet.worksheets():
        print(ws.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
