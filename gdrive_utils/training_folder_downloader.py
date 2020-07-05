from __future__ import print_function
import pickle
import os.path
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io
import numpy as np
import datetime

# If modifying these scopes, delete the file token.pickle.
#SCOPES = ['https://www.gclearoogleapis.com/auth/drive.metadata.readonly']
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']

def print_remaining_time(before, currentPosition, totalSize):
  after = datetime.datetime.now()
  elaspsed_time = (after - before).seconds
  estimated_remaining_time = elaspsed_time * (totalSize - currentPosition) / currentPosition
  
  msg = '%i/%i(%.2f%s) finished. Estimated Remaining Time: %.2f seconds.'%(currentPosition, totalSize, (100*currentPosition/totalSize), '%' ,estimated_remaining_time)
  print(msg)

def download(srcv, item_list, type_name, max_length):
    print('Started to download %s...'%(type_name))
    before = datetime.datetime.now()
    for i, item in enumerate(item_list):
        file_id = item["id"]
        name = item["name"]
        request = srcv.files().get_media(fileId=file_id)
        fh = io.FileIO('../buffer/%s/%s'%(str(type_name), str(name)), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            #print("Downloading %s/%s at: %d%%." % (str(type_name), str(name), int(status.progress() * 100)))

        if (i+1) % 100 == 0:
            print_remaining_time(before, i+1, max_length)
        #print('Downloading %s %i/%i (%.2f%s) completed...'%(type_name, ))

def get_folder_files(srvc, folder_id, max_length):
    print('SEARCH STARTED FOR FOLDER: %s'%folder_id)
    item_list = []
    results = srvc.files().list(
        pageSize=1000, 
        fields="nextPageToken, files(id,parents,name,mimeType)",
        q="trashed=false and '%s' in parents"%folder_id).execute()

    items = results.get('files', [])
    for item in items:
        if folder_id in item["parents"]:
            item_list.append(item)
    
    token = results.get('nextPageToken', None)
    print('Search finished: %i/%i (%.2f%s)'%(len(item_list), max_length, float(100*len(item_list)/max_length), '%'))
    while token != None:
        results = srvc.files().list(
            pageSize=1000, 
            pageToken=token,
            fields="nextPageToken, files(id,parents,name,mimeType)",
            q="trashed=false and '%s' in parents"%folder_id).execute()

        items = results.get('files', [])
        log = np.array(list(map(lambda x: x['name'],items)))
        for item in items:
            if folder_id in item["parents"]:
                item_list.append(item)

        if (len(item_list) == max_length):
            break
        token = results.get('nextPageToken', None)
        print('New page search finished: %i/%i (%.2f%s)'%(len(item_list), max_length, float(100*len(item_list)/max_length), '%'))
    print('Folder search finished... ')
    return item_list
    
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
    buffer_fodler_id =  "1TKwDYdODzDfl_VtXlI6sl8XiNDScKOsL"
    train_foler_id   =  "1O3ZQEK9jwGh8XAC1LFgdoOwes41OsDGM"
    val_folder_id    =  "1QsvWWirpE-p5hZE2nHH98_fGOVo-W8sI"
    test_folder_id   =  "1VdkWTV50Kt0Encfemzc_XdL3FqD1AtNY"

    train_items = get_folder_files(service, train_foler_id, 26848)
    val_items   = get_folder_files(service, val_folder_id, 3356)
    #test_items  = get_folder_files(service, test_folder_id, 3356)

    print('ALL FILES HAVE BEEN FOUND')
    download(service, train_items, 'train', 26848)
    download(service, val_items, 'val', 3356)
    #download(service, test_items, 'test', 3356)

    #request = service.files().get_media(file_id, "application/vnd.google-apps.folder")
    #fh = io.FileIO('../buffer/%s'%str(k), 'wb')
    #downloader = MediaIoBaseDownload(fh, request)
    #done = False
    #while done is False:
        #status, done = downloader.next_chunk()
        #print("Downloading %s at: %d%%." % (str(k), int(status.progress() * 100)))

    

if __name__ == '__main__':
    if not os.path.exists('../buffer'):
        os.mkdir("../buffer")
    if not os.path.exists('../buffer/train'):
        os.mkdir("../buffer/train")
    if not os.path.exists('../buffer/val'):
        os.mkdir("../buffer/val")
    if not os.path.exists('../buffer/test'):
        os.mkdir("../buffer/test")
    main()
