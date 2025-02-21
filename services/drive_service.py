from pydrive2.auth import GoogleAuth
import os


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
