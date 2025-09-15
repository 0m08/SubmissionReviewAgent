from pydrive2.auth import GoogleAuth
import os
import json
import tempfile
import gspread
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


def init_clients_from_credentials(creds: Credentials):
    """
    Given google.oauth2.credentials.Credentials, initialize:
      - PyDrive2 GoogleAuth + GoogleDrive
      - gspread client
    Returns (gauth, drive, gc)
    """
    
    # Convert google.oauth2.credentials.Credentials to oauth2client format
    oauth2_creds = OAuth2Credentials(
        access_token=creds.token,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        refresh_token=creds.refresh_token,
        token_expiry=creds.expiry,
        token_uri=creds.token_uri,
        user_agent=None,
        revoke_uri=None,
        scopes=creds.scopes
    )
    
    gauth = GoogleAuth()
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
