from __future__ import print_function
import pickle
import os.path
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io

# If modifying these scopes, delete the file token.pickle.
#SCOPES = ['https://www.gclearoogleapis.com/auth/drive.metadata.readonly']
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']

def download(file_id, service, destination):
    pass

def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("here")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    # Find items
    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
        return
    
    item_dict = {}
    print('Searching for chexpert files...')
    for item in items:
        if "adjusted_train_images" in item['name']:
            item_dict[item['name']] = item['id']
    
    for k,v in item_dict.items():
        file_id = v
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO('./adjusted_data/%s'%str(k), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Downloading %s at: %d%%." % (str(k), int(status.progress() * 100)))

    

if __name__ == '__main__':
    if not os.path.exists('./adjusted_data'):
        os.mkdir("./adjusted_data")

    main()
