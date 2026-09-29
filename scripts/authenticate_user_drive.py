#!/usr/bin/env python3
"""
One-time OAuth 2.0 authorization script for user Google Drive access.
Authenticates as om@skillcatapp.com (or your personal Google account)
and saves the GOOGLE_OAUTH_REFRESH_TOKEN into .env for persistent automated uploads.

Uses the registered OAUTH_REDIRECT_URI (http://localhost:8501) to prevent
Google 'Error 400: redirect_uri_mismatch'.
"""

import http.server
import os
import re
import sys
import urllib.parse
import webbrowser
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def update_env_file(refresh_token: str, env_path: str = ".env"):
    """Adds or updates GOOGLE_OAUTH_REFRESH_TOKEN in the .env file."""
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f'GOOGLE_OAUTH_REFRESH_TOKEN="{refresh_token}"\n')
        return

    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_line = f'GOOGLE_OAUTH_REFRESH_TOKEN="{refresh_token}"'
    if "GOOGLE_OAUTH_REFRESH_TOKEN" in content:
        content = re.sub(
            r'GOOGLE_OAUTH_REFRESH_TOKEN\s*=.*',
            target_line,
            content
        )
    else:
        content = content.rstrip() + f"\n\n# User OAuth Google Drive Token\n{target_line}\n"

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    auth_code = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
            <html>
            <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #0f172a; color: #f8fafc;">
                <div style="background: #1e293b; padding: 40px 50px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); text-align: center; border: 1px solid #334155; max-width: 480px;">
                    <div style="font-size: 54px; margin-bottom: 12px;">🎉</div>
                    <h2 style="margin: 0 0 10px 0; color: #38bdf8;">Authentication Successful!</h2>
                    <p style="color: #94a3b8; font-size: 15px; line-height: 1.5;">Your Google Drive credentials have been securely captured and saved. You can close this tab and return to your terminal.</p>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Authentication failed: No code received.</h2></body></html>")

    def log_message(self, format, *args):
        # Silence default HTTP server access logs
        return


def authenticate_user():
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501").strip()

    if not client_id or not client_secret:
        print("❌ Error: OAUTH_CLIENT_ID or OAUTH_CLIENT_SECRET is missing in .env.")
        sys.exit(1)

    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    port = parsed_redirect.port or 8501

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }

    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    print("\n========================================================")
    print("🔐 Starting Google Drive OAuth for your Google Account...")
    print(f"   Target Account: om@skillcatapp.com")
    print(f"   Authorized Callback: {redirect_uri}")
    print("========================================================\n")

    # Start local listener on the registered redirect port (8501)
    try:
        server = http.server.HTTPServer(("localhost", port), OAuthCallbackHandler)
    except OSError as e:
        print(f"⚠️ Port {port} is currently in use (e.g. by Streamlit): {e}")
        print(f"   Please stop any running Streamlit process on port {port} and rerun this script.")
        sys.exit(1)

    print(f"🌐 Opening browser for sign-in...")
    print(f"   If it does not open automatically, visit this URL:\n\n{auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for the single callback request
    print("⏳ Waiting for authorization callback in browser...")
    server.handle_request()
    server.server_close()

    code = OAuthCallbackHandler.auth_code
    if not code:
        print("❌ Failed to receive authorization code from Google.")
        sys.exit(1)

    # Exchange code for credentials
    print("🔄 Exchanging authorization code for offline access tokens...")
    flow.fetch_token(code=code)
    creds = flow.credentials

    refresh_token = getattr(creds, "refresh_token", None)
    if not refresh_token:
        print("⚠️ Warning: Google did not return a refresh token.")
        print("   If you have authorized this app before, visit:")
        print("   https://myaccount.google.com/connections to revoke access, then rerun this script.")
        return

    # Verify user email via Drive API
    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        about = service.about().get(fields="user").execute()
        user_info = about.get("user", {})
        email = user_info.get("emailAddress", "Unknown")
        display_name = user_info.get("displayName", "Unknown")
        print(f"\n✅ Authenticated as: {display_name} ({email})")
    except Exception as verify_err:
        print(f"⚠️ Could not query user info from Drive API: {verify_err}")

    # Save to .env
    update_env_file(refresh_token)
    print(f"🎉 Successfully saved GOOGLE_OAUTH_REFRESH_TOKEN to .env!")
    print("   Submissions can now be uploaded directly under om@skillcatapp.com without service accounts.\n")


if __name__ == "__main__":
    authenticate_user()
