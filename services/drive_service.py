from pydrive2.auth import GoogleAuth
import os
import json
import tempfile
import gspread
import re
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from oauth2client.client import OAuth2Credentials

# Common OAuth scopes used across the app
GOOGLE_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


SKILLCAT_SHARED_DRIVE_FOLDER_ID = "1-YOY7Z9kK5_NR2shHbJgRXTJT6D16HIM"


def _iter_service_account_candidates():
    """Yield parsed service-account dict candidates from env in priority order."""
    sa_json = os.environ.get("GDRIVE_SA_JSON")
    if sa_json:
        try:
            yield "GDRIVE_SA_JSON", (json.loads(sa_json) if isinstance(sa_json, str) else sa_json)
        except Exception:
            pass

    sa_b64 = os.environ.get("GDRIVE_SA_B64")
    if sa_b64:
        try:
            yield "GDRIVE_SA_B64", json.loads(base64.b64decode(sa_b64).decode())
        except Exception:
            pass


def build_service_account_drive_service():
    """Build a Google Drive v3 client authenticated with the background-job service account."""
    for _source, sa_dict in _iter_service_account_candidates():
        try:
            creds = service_account.Credentials.from_service_account_info(
                sa_dict, scopes=GOOGLE_OAUTH_SCOPES
            )
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception:
            continue
    return None


def extract_drive_id_from_url(value: str) -> str:
    """Extract a Google Drive file/folder ID from common URL forms or accept a bare ID."""
    if not value:
        return ""
    s = str(value).strip()
    patterns = [
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"/drive/folders/([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, s, re.I)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", s):
        return s
    return ""


def is_inside_skillcat_shared_drive(file_or_folder_id: str, drive_service=None) -> bool:
    """
    Check if a file or folder is inside the Skillcat Shared Drive.
    :param file_or_folder_id: ID of the file or folder to check
    :param drive_service: Google Drive service client (optional, will build one if not provided)
    
    :return: True if the file or folder is inside the Skillcat Shared Drive, False otherwise
    """
   
    if not file_or_folder_id:
        return False
    if drive_service is None:
        drive_service = build_service_account_drive_service()
    if drive_service is None:
        return False

    visited = set()
    current_id = file_or_folder_id
    for _ in range(25):
        if not current_id or current_id in visited:
            return False
        visited.add(current_id)
        if current_id == SKILLCAT_SHARED_DRIVE_FOLDER_ID:
            return True
        try:
            meta = drive_service.files().get(
                fileId=current_id,
                fields="id,parents,driveId,mimeType",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            return False
        if meta.get("driveId") == SKILLCAT_SHARED_DRIVE_FOLDER_ID:
            return True
        parents = meta.get("parents") or []
        if not parents:
            return False
        current_id = parents[0]
    return False


def login_with_service_account(path=None, json_str=None, user_email=None):
    """
    Google Drive service with a service account.
    note: for the service account to work, you need to share the folder or
    files with the service account email.

    :param path: Path to service account JSON file
    :param json_str: Service account JSON as string
    :param user_email: Email of user to impersonate (for domain-wide delegation)
    :return: google auth
    """
    if path:
        settings = {
            "client_config_backend": "service",
            "service_config": {
                "client_json_file_path": path,
            }
        }
    elif json_str:
        settings = {
            "client_config_backend": "service",
            "service_config": {
                "client_json": json_str,
            }
        }

    # Create instance of GoogleAuth
    gauth = GoogleAuth(settings=settings)
    
    # If user_email is provided, use domain-wide delegation
    if user_email:
        # Parse the service account JSON to get the client_email
        if json_str:
            sa_info = json.loads(json_str)
        else:
            with open(path, 'r') as f:
                sa_info = json.load(f)
        
        # Set up delegation
        gauth.ServiceAuth()
        # Create credentials with delegation        
        credentials = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        # Delegate to the specified user
        delegated_credentials = credentials.with_subject(user_email)
        
        # Build the Drive service
        service = build('drive', 'v3', credentials=delegated_credentials)
        
        # Set the service in gauth for compatibility
        gauth.service = service
        
    else:
        # Regular service account authentication
        gauth.ServiceAuth()
    
    return gauth


def login_with_oauth2(client_id=None, client_secret=None, credentials_file=None):
    """
    Google Drive service with OAuth 2.0 user authentication.
    This uses a regular user account with storage quota instead of a service account.
    
    :param client_id: OAuth 2.0 client ID from environment variables
    :param client_secret: OAuth 2.0 client secret from environment variables
    :param credentials_file: Path to store/load user credentials
    :return: google auth
    """
    if not client_id or not client_secret:
        raise ValueError("Both client_id and client_secret must be provided")
    
    # Create config dynamically from environment variables
    config = {
        "installed": {
            "client_id": client_id,
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"]
        }
    }
    
    # Write config to temporary file
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config_file)
    temp_config_file.close()
    
    settings = {
        "client_config_backend": "file",
        "client_config_file": temp_config_file.name,
        "save_credentials": False,  # Don't save credentials
        "get_refresh_token": False,  # Don't get refresh token
        "oauth_scope": ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    }
    
    # Create instance of GoogleAuth
    gauth = GoogleAuth(settings=settings)
    
    # Always authenticate (no saved credentials check)
    gauth.LocalWebserverAuth()
    
    # Clean up temporary file
    try:
        os.unlink(temp_config_file.name)
    except:
        pass
    
    return gauth


def get_oauth_credentials_for_gspread(client_id=None, client_secret=None):
    """
    Get OAuth 2.0 credentials that can be used with gspread.
    Returns the credentials object that can be used with gspread.oauth()
    
    :param client_id: OAuth 2.0 client ID from environment variables
    :param client_secret: OAuth 2.0 client secret from environment variables
    :return: credentials object for gspread
    """
    if not client_id or not client_secret:
        raise ValueError("Both client_id and client_secret must be provided")
    
    # Create config for gspread
    config = {
        "installed": {
            "client_id": client_id,
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"]
        }
    }
    
    # Write config to temporary file
    temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config_file)
    temp_config_file.close()
    
    # Create temporary file for authorized user credentials
    temp_auth_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_auth_file.close()
    
    # Use gspread's oauth method with temporary files
    gc = gspread.oauth(
        credentials_filename=temp_config_file.name,
        authorized_user_filename=temp_auth_file.name
    )
    
    # Clean up temporary files
    try:
        os.unlink(temp_config_file.name)
        os.unlink(temp_auth_file.name)
    except:
        pass
    
    return gc


# ================================
# Web OAuth flow for Streamlit
# ================================

def _build_web_client_config(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """Create a Google OAuth "web" client config dict for google-auth-oauthlib."""
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            # It's fine to include the single redirect URI we intend to use here
            "redirect_uris": [redirect_uri],
        }
    }


def get_google_oauth_authorization_url(client_id: str, client_secret: str, redirect_uri: str):
    """
    Initialize the web OAuth flow and return (authorization_url, state).
    The caller should persist the returned "state" in session and verify it on callback.
    """
    client_config = _build_web_client_config(client_id, client_secret, redirect_uri)
    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=GOOGLE_OAUTH_SCOPES,
        redirect_uri=redirect_uri,
    )
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return authorization_url, state


def exchange_code_for_credentials(client_id: str, client_secret: str, redirect_uri: str, code: str) -> Credentials:
    """
    Exchange an authorization code for Google OAuth credentials.
    """
    client_config = _build_web_client_config(client_id, client_secret, redirect_uri)
    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=GOOGLE_OAUTH_SCOPES,
        redirect_uri=redirect_uri,
    )
    # Fetch tokens using the "code" directly 
    flow.fetch_token(code=code)
    return flow.credentials


def init_clients_from_credentials(creds: Credentials, client_id: str = None, client_secret: str = None):
    """
    Given google.oauth2.credentials.Credentials, initialize:
      - PyDrive2 GoogleAuth + GoogleDrive
      - gspread client
    Returns (gauth, drive, gc)
    
    Args:
        creds: Google OAuth2 credentials object
        client_id: OAuth2 client ID (if not provided, will try to get from env vars)
        client_secret: OAuth2 client secret (if not provided, will try to get from env vars)
    """
    # Get client_id and client_secret from parameters or environment variables
    if not client_id:
        client_id = os.getenv("OAUTH_CLIENT_ID")
    if not client_secret:
        client_secret = os.getenv("OAUTH_CLIENT_SECRET")
    
    # Check if we have valid (non-empty) values
    if not client_id or not client_secret or not client_id.strip() or not client_secret.strip():
        raise ValueError("Missing required setting client_id. Please ensure OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET are set in your environment variables.")
    
    # Convert google.oauth2.credentials.Credentials to oauth2client format
    oauth2_creds = OAuth2Credentials(
        access_token=creds.token,
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=creds.refresh_token,
        token_expiry=creds.expiry,
        token_uri=creds.token_uri,
        user_agent=None,
        revoke_uri=None,
        scopes=creds.scopes
    )
    
    # Create a persistent temporary config file with absolute path
    # This prevents PyDrive2 from looking for client_secrets.json in the current working directory
    # when the working directory changes during long-running workflows
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"]
        }
    }
    
    # Use tempfile.gettempdir() to get a system temp directory with absolute path
    # This ensures the file path doesn't depend on the current working directory
    temp_dir = tempfile.gettempdir()
    config_file_path = os.path.join(temp_dir, "pydrive2_client_config.json")
    
    # Write config to temp file (overwrite if exists)
    with open(config_file_path, 'w') as f:
        json.dump(client_config, f)
    
    settings = {
        "client_config_backend": "file",
        "client_config_file": config_file_path,  # Absolute path - won't break if cwd changes
        "save_credentials_backend": "file",
        "save_credentials_file": os.path.join(temp_dir, "pydrive2_credentials.json"),
        "oauth_scope": creds.scopes,
    }
    gauth = GoogleAuth(settings=settings)
    gauth.credentials = oauth2_creds
    # Ensure HTTP is authorized for PyDrive2 operations
    gauth.http = AuthorizedHttp(creds)

    drive = None
    try:
        from pydrive2.drive import GoogleDrive
        drive = GoogleDrive(gauth)
    except Exception:
        # Defer failures to callers that actually need Drive
        drive = None

    gc = gspread.authorize(creds)
    return gauth, drive, gc


def try_build_user_drive_for_background_jobs():
    """
    Build a PyDrive GoogleDrive client using a user OAuth refresh token from the environment for background jobs.

    - OAUTH_CLIENT_ID / OAUTH_CLIENT_SECRET — same as Streamlit
    - GOOGLE_OAUTH_REFRESH_TOKEN — refresh token for the Google account that should own uploads

    
    """
    refresh = os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    if not refresh:
        return None
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    try:
        from google.auth.transport.requests import Request

        creds = Credentials(
            None,
            refresh_token=refresh,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=GOOGLE_OAUTH_SCOPES,
        )
        creds.refresh(Request())
        _gauth, drive, _gc = init_clients_from_credentials(
            creds, client_id=client_id, client_secret=client_secret
        )
        if drive is None:
            print("[WARN] init_clients_from_credentials returned no GoogleDrive instance.")
            return None
        return drive
    except Exception as e:
        print(f"[WARN] Could not build user Drive from GOOGLE_OAUTH_REFRESH_TOKEN: {e}")
        return None


def get_authenticated_drive_client():
    """
    Get an authenticated PyDrive GoogleDrive client instance.
    Prioritizes user OAuth credentials (from Streamlit session state or background environment refresh token).
    Falls back to the service account.
    """
    # 1. Try Streamlit session state
    try:
        import streamlit as st
        drive = st.session_state.get("drive")
        if drive is not None:
            return drive
    except Exception:
        pass

    # 2. Try background OAuth refresh token
    try:
        drive = try_build_user_drive_for_background_jobs()
        if drive is not None:
            return drive
    except Exception as e:
        print(f"[WARN] Failed to build user drive client from refresh token: {e}")

    # 3. Fallback to service account
    try:
        sa_json = os.environ.get("GDRIVE_SA_JSON")
        if sa_json:
            sa_dict = json.loads(sa_json) if isinstance(sa_json, str) else sa_json
        else:
            sa_b64 = os.environ.get("GDRIVE_SA_B64")
            if sa_b64:
                sa_dict = json.loads(base64.b64decode(sa_b64).decode())
            else:
                sa_dict = None
        if sa_dict:
            from pydrive2.drive import GoogleDrive
            gauth = login_with_service_account(json_str=json.dumps(sa_dict))
            gauth.ServiceAuth()
            return GoogleDrive(gauth)
    except Exception as e:
        print(f"[WARN] Failed to build service account Drive client: {e}")

    return None


def get_authenticated_drive_service():
    """
    Get a Google Drive API v3 service client (googleapiclient.discovery.build).
    Prioritizes user OAuth credentials (from Streamlit session state or background environment refresh token).
    Falls back to the service account.

    NOTE: Uses google.oauth2.credentials.Credentials directly (not oauth2client),
    which is the correct type for googleapiclient.discovery.build.
    """
    from googleapiclient.discovery import build as _api_build

    # 1. Try background OAuth refresh token (works both in UI background jobs
    #    and when GOOGLE_OAUTH_REFRESH_TOKEN is set directly in the environment).
    refresh = os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    if refresh and client_id and client_secret:
        try:
            from google.auth.transport.requests import Request

            creds = Credentials(
                None,
                refresh_token=refresh,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=GOOGLE_OAUTH_SCOPES,
            )
            creds.refresh(Request())
            return _api_build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as e:
            print(f"[WARN] Failed to build user Drive API service from refresh token: {e}")

    # 2. Try Streamlit session state — re-hydrate creds from the stored refresh token
    #    if available so we get a proper google.oauth2.credentials.Credentials object.
    try:
        import streamlit as st
        stored_rt = st.session_state.get("google_oauth_refresh_token")
        if stored_rt and client_id and client_secret:
            from google.auth.transport.requests import Request

            creds = Credentials(
                None,
                refresh_token=stored_rt,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=GOOGLE_OAUTH_SCOPES,
            )
            creds.refresh(Request())
            return _api_build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"[WARN] Failed to build user Drive API service from session refresh token: {e}")

    # 3. Fallback to service account
    return build_service_account_drive_service()



def share_sheet_with_service_account(sheet, service_account_email: str, creds: Credentials):
    """
    Share a Google Sheet with a service account email using OAuth credentials.
    Checks if the sheet is already shared with the service account before sharing.
    
    :param sheet: gspread Spreadsheet object
    :param service_account_email: Email address of the service account
    :param creds: google.oauth2.credentials.Credentials object
    :return: True if shared successfully or already shared, False otherwise
    """
    try:
        # Build Drive service with OAuth credentials
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Get the file ID from the sheet
        file_id = sheet.id
        
        # Check existing permissions
        try:
            permissions = drive_service.permissions().list(
                fileId=file_id,
                fields='permissions(id,emailAddress,role)'
            ).execute()
            
            # Check if service account already has access
            for perm in permissions.get('permissions', []):
                if perm.get('emailAddress') == service_account_email:
                    # Already shared, return True
                    return True
        except Exception as e:
            # If we can't check permissions, try to share anyway
            pass
        
        # Share the file with the service account
        permission = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': service_account_email
        }
        
        drive_service.permissions().create(
            fileId=file_id,
            body=permission,
            sendNotificationEmail=False  
        ).execute()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to share sheet with service account: {e}")
        return False


def get_service_account_email():
    """
    Extract the service account email from environment variables.
    
    :return: Service account email address or None if not found
    """
    try:
        # Try GDRIVE_SA_JSON first
        sa_json = os.environ.get("GDRIVE_SA_JSON")
        if sa_json:
            sa_dict = json.loads(sa_json) if isinstance(sa_json, str) else sa_json
            return sa_dict.get('client_email')
        
        # Try GDRIVE_SA_B64
        sa_b64 = os.environ.get("GDRIVE_SA_B64")
        if sa_b64:
            key_bytes = base64.b64decode(sa_b64)
            sa_json = key_bytes.decode()
            sa_dict = json.loads(sa_json)
            return sa_dict.get('client_email')
        
        return None
    except Exception as e:
        print(f"[ERROR] Failed to extract service account email: {e}")
        return None


# Example helper function to recursively download a folder from Google Drive
# This ensures that the local directory structure mirrors what we have on Drive.
def download_folder_from_drive(folder_id: str, local_path: str, drive) -> None:
    """
    Recursively download all files and subfolders from a given Google Drive folder ID
    into the specified local path.

    :param folder_id: ID of the Google Drive folder to download.
    :param local_path: Path to the local folder where files will be saved.
    :param drive: Authenticated GoogleDrive instance.
    """
    os.makedirs(local_path, exist_ok=True)
    # List all items in the folder
    file_list = drive.ListFile({ 'q': f"'{folder_id}' in parents" }).GetList()
    for item in file_list:
        mime_type = item.get('mimeType')
        title = item.get('title')
        item_id = item.get('id')
        if mime_type == 'application/vnd.google-apps.folder':
            # Create a matching subfolder locally and recurse
            subfolder_path = os.path.join(local_path, title)
            download_folder_from_drive(item_id, subfolder_path, drive)
        else:
            # Download the file into local_path
            local_file_path = os.path.join(local_path, title)
            item.GetContentFile(local_file_path)


def upload_folder_to_drive(local_folder_path: str, parent_folder_id: str, drive) -> None:
    """
    Recursively upload an entire local folder (and its subfolders) to a folder on Google Drive.
    :param local_folder_path: Path to the local folder you want to upload.
    :param parent_folder_id: The ID of the parent folder on Google Drive where this folder should go.
    :param drive: An authenticated GoogleDrive instance.
    """
    for item in os.listdir(local_folder_path):
        item_path = os.path.join(local_folder_path, item)

        # Check if this is a directory or file
        if os.path.isdir(item_path):
            # Create a corresponding folder on Drive
            folder_metadata = {
                'title': item,
                'parents': [{'id': parent_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            new_folder = drive.CreateFile(folder_metadata)
            new_folder.Upload()

            # Recursively upload the subfolder's contents
            upload_folder_to_drive(item_path, new_folder['id'], drive)
        else:
            # Upload a file
            f = drive.CreateFile({'title': item, 'parents': [{'id': parent_folder_id}]})
            f.SetContentFile(item_path)
            f.Upload()


def copy_sheet_from_link(drive, link, new_name, parent_id):
    """
    Creates a copy of a Google Sheet from a shareable link.
    Args:
        drive: Authenticated PyDrive GoogleDrive object
        link (str): Google Sheets shareable link
        new_name (str): Desired name for the copied sheet
        parent_id (str): Google Drive folder ID where the copy will be placed
    Returns:
        str: File ID of the newly created copy
    """
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    if match:
        file_id = match.group(1)
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata()
        new_file = file.Copy()
        new_file['title'] = new_name
        new_file['parents'] = [{'id': parent_id}]
        new_file.Upload()
        return new_file['id']
    else:
        raise ValueError("Invalid link")
