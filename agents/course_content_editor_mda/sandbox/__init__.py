"""The execution sandbox, and how its scripts reach Google without holding a key.

Declaring this is what gives the agent a shell, which is what lets the sheet
skill ship real `scripts/` instead of the four in-process tools. Without a shell
the agent can read a script but has no way to run it.

## Why the credential is not in here

Deploy-forwarded `.env` values reach `setup.sh` at bake time, but **thread
sandboxes do not inherit them** — so `os.environ["GDRIVE_SA_B64"]` inside a
script raises `KeyError`, and baking the key into the snapshot would put it in
every thread's image. Both roads end with a credential sitting on a disk the
model can read, which is the one thing worth avoiding: an attacker who controls
part of the agent's input (a malicious cell in a course sheet) can read files
and run commands in here.

The auth proxy sidesteps the whole question. It intercepts sandbox egress and
injects credentials resolved from LangSmith workspace secrets, *outside* the
box. The script sends a bare, unauthenticated request; the proxy fills in the
`Authorization` header on the way out. Nothing to read, because nothing arrives.

`gcp_auth` is purpose-built for this: it holds the service account JSON, mints
OAuth tokens, and matches Google API hosts on its own, so `match_hosts` is not
needed. Store the JSON as the workspace secret named below — plaintext is
rejected, it must be wrapped in `workspace_secret()`.

## What the scripts must do differently

Authorize gspread with `AnonymousCredentials`, not a service account file:

    import gspread
    from google.auth.credentials import AnonymousCredentials

    gc = gspread.authorize(AnonymousCredentials())
    ws = gc.open_by_key(sheet_id).worksheet(tab)

`AnonymousCredentials` reports `valid=True`, never refreshes, and its
`before_request` is a no-op, so `AuthorizedSession` adds no `Authorization` of
its own and leaves the header free for the proxy to set.

Prefer `open_by_key` over `open(name)`: the latter runs a Drive search, which is
the only reason the Drive scope would be needed at all.

The proxy handles authentication, not authorization — the service account email
still has to be shared on the sheet, exactly as it is today.
"""

from __future__ import annotations

from langsmith.sandbox import gcp_auth, proxy_config, workspace_secret
from managed_deepagents import define_sandbox

#: LangSmith workspace secret holding the service account JSON. Set it once with
#: `POST /workspaces/current/secrets`; it is never returned by the API and never
#: enters the sandbox.
SA_SECRET = "GCP_SERVICE_ACCOUNT_JSON"

sandbox = define_sandbox(
    # One box per thread, reaped after ten idle minutes. A conversation that
    # pauses while someone reads a diff should not pay to keep a VM warm.
    idle_ttl_seconds=600,
    proxy_config=proxy_config(
        rules=[
            gcp_auth(
                service_account_json=workspace_secret(SA_SECRET),
                # Sheets alone. `open_by_key` avoids the Drive search that would
                # otherwise force a broader grant than the job needs.
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
        ]
    ),
)
