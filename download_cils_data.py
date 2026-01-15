import requests
import os
import zipfile
from pathlib import Path

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    output_dir = Path("data/boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("1cLE7l56OnCDoC_kvPjpCQVled3lg4Pya", "cils_mines_polygon.zip"),
        ("1gsJdrFSv27-IhqhnS13bYWkXJY6E24fY", "cils_mines_point.zip")
    ]

    for file_id, filename in files:
        dest_path = output_dir / filename
        print(f"Downloading {filename}...")
        try:
            download_file_from_google_drive(file_id, dest_path)
            print(f"Downloaded {filename}")
            
            # Check if it is a valid zip
            if zipfile.is_zipfile(dest_path):
                print(f"Extracting {filename}...")
                unzip_file(dest_path, output_dir)
                print(f"Extracted {filename}")
            else:
                print(f"Warning: {filename} is not a valid zip file. It might be a direct file.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
