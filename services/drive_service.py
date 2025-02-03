from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def login_with_service_account(path):
    """
    Google Drive service with a service account.
    note: for the service account to work, you need to share the folder or
    files with the service account email.

    :return: google auth
    """
    # Define the settings dict to use a service account
    # We also can use all options available for the settings dict like
    # oauth_scope,save_credentials,etc.
    settings = {
                "client_config_backend": "service",
                "service_config": {
                    "client_json_file_path": path,
                }
            }
    # Create instance of GoogleAuth
    gauth = GoogleAuth(settings=settings)
    # Authenticate
    gauth.ServiceAuth()
    return gauth

gauth = login_with_service_account("content/service-credentials.json")
drive = GoogleDrive(gauth)