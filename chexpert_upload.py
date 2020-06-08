from __future__ import print_function
import pickle
import os.path
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io

# If modifying these scopes, delete the file token.pickle.
#SCOPES = ['https://www.gclearoogleapis.com/auth/drive.metadata.readonly']
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']

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
    file_metadata = {'name': 'sem_train'}
    media = MediaFileUpload('./semantic_train_data/sem_train',  chunksize=1024 * 1024, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print("SEM-TRAIN Uploaded %d%%." % int(status.progress() * 100))

    file_metadata = {'name': 'sem_labels'}
    media = MediaFileUpload('./semantic_train_data/sem_labels',  chunksize=1024 * 1024, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print("SEM-LABEL Uploaded %d%%." % int(status.progress() * 100))

if __name__ == '__main__':
    if not os.path.exists('./semantic_train_data'):
        exit()
    
    if not os.path.exists('./semantic_train_data/sem_train'):
        print("Unable to get semantic train data...")
        exit()

    if not os.path.exists('./semantic_train_data/sem_labels'):
        print("Unable to get semantic train labels...")
        exit()

    main()
