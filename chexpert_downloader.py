# GOOGLE DRIVE DOWNLOAD FUNCTIONS ARE TAKEN FROM THIS StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
import os
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        save_response_content(response, destination)    
    else:
        print("Token confirmation failed!")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
        else:
            print("key: %s - value: %s"%(str(key),str(value)))

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id_dict = {}
data = pd.read_csv("./file_ids.csv",  header=None)
for i in range(len(data)):
    k = data[0][i]
    v = data[1][i]
    if type(v) != type("s"):
        continue

    parsed_id = v.split('file/d/')[1].split('/')[0]
    file_id_dict[k] = parsed_id


if not os.path.exists("./adjusted_data"):
    os.mkdir("./adjusted_data")

for k,v in file_id_dict.items():
    destination = "./adjusted_data/%s"%str(k)
    download_file_from_google_drive(k, destination)
