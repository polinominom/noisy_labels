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
    media = MediaFileUpload('./semantic_train_data/sem_train', mimetype=None, uploadType="resumable")
    file = service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()
    
    print('TRAIN - File ID: %s' % file.get('id'))

    file_metadata = {'name': 'sem_labels'}
    media = MediaFileUpload('./semantic_train_data/sem_labels', mimetype=None, uploadType="resumable")
    file = service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()

    print('LABELS - File ID: %s' % file.get('id'))

if __name__ == '__main__':
    if not os.path.exists('./semantic_train_data'):
        exit()

    main()
