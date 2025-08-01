import pickle


def save_as_pickle_pydrive(obj, filename, folder_id, drive):
    """
    Save an object as a pickle file and upload it to a specific folder in Google Drive.

    :param obj: The object to be pickled.
    :param filename: The name of the file to save the pickle to.
    :param folder_id: The ID of the folder to upload the file to in Google Drive.
    :param drive: Authenticated GoogleDrive instance.
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object successfully saved to {filename}")

    # Upload the pickle file to Google Drive
    file_drive = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
    file_drive.SetContentFile(filename)
    file_drive.Upload()
    print(f"Pickle file successfully uploaded to folder {folder_id} on Google Drive as {filename}")


def load_from_pickle_pydrive(filename, folder_id, drive):
    """
    Download a pickle file from a specific folder in Google Drive and load an object from it.

    :param filename: The name of the file to load the pickle from.
    :param folder_id: The ID of the folder to search for the file in Google Drive.
    :param drive: Authenticated GoogleDrive instance.
    :return: The object that was loaded from the pickle file.
    """
    # Search for the file in the specified folder on Google Drive
    query = f"title='{filename}' and '{folder_id}' in parents"
    file_list = drive.ListFile({'q': query}).GetList()
    if not file_list:
        raise FileNotFoundError(f"File {filename} not found in folder {folder_id} on Google Drive.")

    # Download the file
    file_drive = file_list[0]
    file_drive.GetContentFile(filename)
    print(f"Pickle file {filename} successfully downloaded from folder {folder_id} on Google Drive.")

    # Load the object from the pickle file
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    print(f"Object successfully loaded from {filename}")
    return obj