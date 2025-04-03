from pydrive2.drive import GoogleDrive
from typing import List, Dict, Any
import os
import streamlit as st
import pandas as pd
from utils.decorator_helpers import try_n_times
from services.sheets_service import save_to_sheet, create_or_read_worksheet, format_worksheet, resize_column_by_name


def find_images_with_empty_description(folder_id: str, drive: GoogleDrive) -> List[Dict[str, Any]]:
    """
    Recursively search through a Google Drive folder and its subfolders
    to find all image files that have an empty description.
    
    :param folder_id: ID of the Google Drive folder to search in
    :param drive: Authenticated GoogleDrive instance
        
    :return:
        List of dictionaries containing information about each image with empty description:
        [{
            'id': image_id,
            'title': image_title,
            'parentFolder': parent_folder_name,
            'mimeType': mime_type,
            'thumbnailLink': thumbnail_link (if available)
        }]
    """
    results = []
    
    # Define image MIME types to search for
    image_mime_types = [
        'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 
        'image/webp', 'image/tiff', 'image/svg+xml'
    ]
    
    # Helper function to get folder name by ID
    def get_folder_name(folder_id):
        folder = drive.CreateFile({'id': folder_id})
        folder.FetchMetadata()
        return folder.get('title', 'Unknown Folder')
    
    # Recursive function to search folders
    def search_folder(folder_id, parent_name=None):
        if parent_name is None:
            parent_name = get_folder_name(folder_id)
            
        # List all items in the current folder
        query = f"'{folder_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        
        for item in file_list:
            mime_type = item.get('mimeType')
            
            # If it's a folder, recursively search it
            if mime_type == 'application/vnd.google-apps.folder':
                subfolder_name = item.get('title', 'Unknown Folder')
                search_folder(item['id'], subfolder_name)
                
            # If it's an image, check if it has an empty description
            elif any(mime_type == img_type for img_type in image_mime_types):
                # Check if description is empty or doesn't exist
                if not item.get('description'):
                    results.append({
                        'id': item['id'],
                        'title': item.get('title', 'Untitled'),
                        'parentFolder': parent_name,
                        'mimeType': mime_type,
                        'thumbnailLink': item.get('thumbnailLink', None)
                    })
    
    # Start the recursive search
    search_folder(folder_id)
    return results

    


def add_folder_details_in_sheet(sheet, worksheet_name, folder_id, drive):
    """
    Add folder details to a Google Sheet.

    :param sheet: Google Sheet object
    :param worksheet_name: Name of the worksheet to add the folder details to
    """
    folders_sheet, folders_df = create_or_read_worksheet(sheet, worksheet_name)

    # List all folders
    folders = list_all_folders(folder_id, drive)

    folders_df = pd.concat([folders_df, folders])

    save_to_sheet(worksheet = folders_sheet, df = folders_df)

    format_worksheet(worksheet = folders_sheet)
    resize_column_by_name(worksheet = folders_sheet, column_name = "id", pixel_size = 100, wrap = "WRAP")
    resize_column_by_name(worksheet = folders_sheet, column_name = "title", pixel_size = 200, wrap = "WRAP")
    resize_column_by_name(worksheet = folders_sheet, column_name = "parentFolder", pixel_size = 200, wrap = "WRAP")
    resize_column_by_name(worksheet = folders_sheet, column_name = "createdDate", pixel_size = 200)
    resize_column_by_name(worksheet = folders_sheet, column_name = "modifiedDate", pixel_size = 200)
    resize_column_by_name(worksheet = folders_sheet, column_name = "owner", pixel_size = 150)
    # resize_column_by_name(worksheet = folders_sheet, column_name = "isSourceFolder", pixel_size = 100)

    return



def download_images_with_empty_description(sheet: str, 
                                          worksheet_name: str, 
                                          folder_id: str, 
                                          download_path: str = "/tmp/graphics", 
                                          drive: GoogleDrive = None) -> List[Dict[str, Any]]:
    """
    Find all images with empty descriptions in a Google Drive folder and download them.

    :param folder_id: ID of the Google Drive folder to search in
    :param download_path: Local path where images will be downloaded
    :param drive: Authenticated GoogleDrive instance
        
    :return:
        List of dictionaries with information about the downloaded images
    """
    
    # Find all images with empty descriptions
    with st.spinner("Searching for images with empty descriptions...", show_time=True):
        images = find_images_with_empty_description(folder_id, drive)
    
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download each image
    # for image in images:
    #     file_id = image['id']
    #     file_name = image['title']
        
    #     # Create a subfolder for the parent folder
    #     parent_folder_path = os.path.join(download_path, image['parentFolder'])
    #     os.makedirs(parent_folder_path, exist_ok=True)
        
    #     # Download the file
    #     file_path = os.path.join(parent_folder_path, file_name)
    #     drive_file = drive.CreateFile({'id': file_id})
    #     drive_file.GetContentFile(file_path)
        
    #     # Update the local path in the results
    #     image['localPath'] = file_path
    
    # # Print the results
    # print(f"Downloaded {len(images)} images to {download_path}")

    # print(images)

    # Show as df on streamlit
    st.write("Downloaded Images")
    df = pd.DataFrame(images)
    st.dataframe(df)

    add_folder_details_in_sheet(sheet, worksheet_name, folder_id, drive)

    raise Exception("Stop here")
    return images


def list_all_folders(folder_id: str, drive: GoogleDrive) -> pd.DataFrame:
    """
    Recursively search through a Google Drive folder and its subfolders
    to find all folders and return their details as a DataFrame.
    
    :param folder_id: ID of the Google Drive folder to search in
    :param drive: Authenticated GoogleDrive instance
        
    :return:
        DataFrame containing information about each folder:
        - id: folder_id
        - title: folder_name
        - parentFolder: parent_folder_name
        - createdDate: creation date
        - modifiedDate: last modified date
        - owner: owner's email
        - isSourceFolder: boolean indicating if this is the source folder
    """
    results = []
    
    # Helper function to get folder name by ID
    def get_folder_name(folder_id):
        folder = drive.CreateFile({'id': folder_id})
        folder.FetchMetadata()
        return folder.get('title', 'Unknown Folder')
    
    # Get source folder details
    source_folder = drive.CreateFile({'id': folder_id})
    source_folder.FetchMetadata()
    source_folder_info = {
        'id': folder_id,
        'title': source_folder.get('title', 'Untitled'),
        'parentFolder': 'Root',  # Source folder's parent is considered as Root
        'createdDate': source_folder.get('createdDate', 'Unknown'),
        'modifiedDate': source_folder.get('modifiedDate', 'Unknown'),
        'owner': source_folder.get('owners', [{}])[0].get('emailAddress', 'Unknown'),
        'isSourceFolder': True
    }
    results.append(source_folder_info)
    
    # Recursive function to search folders
    @try_n_times(3)
    def search_folder(folder_id, parent_name=None):
        if parent_name is None:
            parent_name = get_folder_name(folder_id)
            
        # List all items in the current folder
        query = f"'{folder_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        
        for item in file_list:
            mime_type = item.get('mimeType')
            
            # If it's a folder, add it to results and recursively search it
            if mime_type == 'application/vnd.google-apps.folder':
                folder_info = {
                    'id': item['id'],
                    'title': item.get('title', 'Untitled'),
                    'parentFolder': parent_name,
                    'createdDate': item.get('createdDate', 'Unknown'),
                    'modifiedDate': item.get('modifiedDate', 'Unknown'),
                    'owner': item.get('owners', [{}])[0].get('emailAddress', 'Unknown'),
                    'isSourceFolder': False
                }
                results.append(folder_info)
                search_folder(item['id'], item.get('title', 'Untitled'))
    
    # Start the recursive search
    search_folder(folder_id)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Show the DataFrame in Streamlit
    st.write("Folder Structure")
    st.dataframe(df)
    
    return df
