import requests

url = 'https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=6885d7419d5196e2adff99cae21341663d6e4c94c359a1a9f598807e889ec561&v=1&xid=2cdd4dd795&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed'

def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
                print('a chunk is finished...')
    return local_filename

download_file(url, 'test.zip')